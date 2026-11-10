"""Online frozen CSP + LR/SVM control loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.online import (
    AvatarController,
    DecisionGate,
    EMGRejector,
    EmgRejectConfig,
    OnlineLogger,
    OnlinePipeline,
    OnlineStream,
    TrialLoop,
    TrialTiming,
    WindowManager,
    load_frozen_model,
    save_frozen_model,
    train_frozen_model,
)
from pipeline.online.avatar import AvatarConfig
from pipeline.online.gate import DecisionGateConfig
from pipeline.online.types import WindowDecision
from pipeline.online.windows import ImageryWindow
from pipeline.preprocess import FilterSpec


def _synthetic_trials(
    n_per_class: int = 12,
    channels: int = 8,
    samples: int = 500,
    rate: float = 250.0,
) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(0)
    trials: list[np.ndarray] = []
    labels: list[str] = []
    t = np.arange(samples) / rate
    for _ in range(n_per_class):
        left = rng.normal(0.0, 1.0, size=(channels, samples))
        left[0] += 2.5 * np.sin(2 * np.pi * 10 * t)
        trials.append(left)
        labels.append("left")
        right = rng.normal(0.0, 1.0, size=(channels, samples))
        right[1] += 2.5 * np.sin(2 * np.pi * 20 * t)
        trials.append(right)
        labels.append("right")
    return trials, labels


def test_train_save_load_linear_svm(tmp_path: Path) -> None:
    trials, labels = _synthetic_trials()
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_name="linear_svm",
        filter=FilterSpec(notch_hz=50.0, band_low_hz=8.0, band_high_hz=30.0),
        apply_filter=True,
    )
    assert model.offline_accuracy is not None
    assert model.offline_accuracy >= 0.7
    path = save_frozen_model(model, tmp_path / "model.joblib")
    loaded = load_frozen_model(path)
    assert loaded.classifier_name == "linear_svm"
    assert loaded.csp_filters.shape[0] == 8


def test_train_logistic_regression() -> None:
    trials, labels = _synthetic_trials()
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_name="logistic_regression",
        apply_filter=True,
    )
    assert model.classifier_name == "logistic_regression"
    assert model.offline_accuracy is not None


def test_train_multiple_classifiers_share_csp(tmp_path: Path) -> None:
    trials, labels = _synthetic_trials()
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_names=("linear_svm", "logistic_regression"),
        apply_filter=True,
    )
    assert model.classifier_names == ("linear_svm", "logistic_regression")
    assert set(model.offline_accuracies) == {"linear_svm", "logistic_regression"}
    path = save_frozen_model(model, tmp_path / "multi.joblib")
    loaded = load_frozen_model(path)
    assert loaded.classifier_names == ("linear_svm", "logistic_regression")
    preds = loaded.predict_all(trials[0])
    assert {name for name, _label, _score in preds} == {
        "linear_svm",
        "logistic_regression",
    }


def test_window_manager_emits_five_windows() -> None:
    rate = 250.0
    stream = OnlineStream(
        sample_rate_hz=rate,
        n_eeg_channels=8,
        n_emg_channels=2,
        apply_causal_filter=True,
    )
    manager = WindowManager(stream)
    onset = 10.0
    manager.begin(onset)
    count = int(rate * 4.2)
    times = [onset + index / rate for index in range(count)]
    eeg = np.random.default_rng(1).normal(size=(count, 8))
    emg = np.random.default_rng(2).normal(size=(count, 2)) * 0.1
    stream.push_eeg(eeg, times)
    stream.push_emg(emg, times)
    ready = manager.poll_ready()
    assert len(ready) == 5
    assert [window.index for window in ready] == [0, 1, 2, 3, 4]
    assert manager.all_complete()


def test_emg_rejector_flags_contaminated_window() -> None:
    rejector = EMGRejector(EmgRejectConfig(threshold=5.0, metric="rms"))
    clean = ImageryWindow(
        index=0,
        start=0.0,
        end=2.0,
        eeg=np.zeros((100, 8)),
        eeg_times=np.arange(100),
        emg=np.ones((100, 2)) * 0.1,
        emg_times=np.arange(100),
        complete=True,
    )
    dirty = ImageryWindow(
        index=1,
        start=0.5,
        end=2.5,
        eeg=np.zeros((100, 8)),
        eeg_times=np.arange(100),
        emg=np.ones((100, 2)) * 20.0,
        emg_times=np.arange(100),
        complete=True,
    )
    assert not rejector.check(clean).rejected
    assert rejector.check(dirty).rejected


def test_decision_gate_majority_and_hold() -> None:
    gate = DecisionGate(DecisionGateConfig(min_consecutive=3, majority_of=3))
    decisions = [
        WindowDecision(0, 0.0, 2.0, "left", 1.0, False, "classified"),
        WindowDecision(1, 0.5, 2.5, "left", 1.0, False, "classified"),
        WindowDecision(2, 1.0, 3.0, "right", 1.0, False, "classified"),
    ]
    state = gate.extend(decisions)
    assert state.command == "hold"
    gate.reset()
    stable = [
        WindowDecision(0, 0.0, 2.0, "right", 1.0, False, "classified"),
        WindowDecision(1, 0.5, 2.5, "right", 1.0, False, "classified"),
        WindowDecision(2, 1.0, 3.0, "right", 1.0, False, "classified"),
    ]
    state = gate.extend(stable)
    assert state.stable
    assert state.command == "right"


def test_avatar_moves_only_on_stable_correct() -> None:
    avatar = AvatarController(AvatarConfig(steps_to_goal=3))
    avatar.begin_trial("left")
    gate = DecisionGate()
    # Unstable / wrong → hold
    wrong = gate.extend(
        [
            WindowDecision(0, 0, 2, "right", 1.0, False),
            WindowDecision(1, 0.5, 2.5, "right", 1.0, False),
            WindowDecision(2, 1.0, 3.0, "right", 1.0, False),
        ]
    )
    state = avatar.apply(wrong)
    assert state.steps_taken == 0
    assert state.position == 0.0
    gate.reset()
    correct = gate.extend(
        [
            WindowDecision(0, 0, 2, "left", 1.0, False),
            WindowDecision(1, 0.5, 2.5, "left", 1.0, False),
            WindowDecision(2, 1.0, 3.0, "left", 1.0, False),
        ]
    )
    state = avatar.apply(correct)
    assert state.steps_taken == 1
    assert state.position < 0.0


def test_online_pipeline_skips_emg_and_classifies(tmp_path: Path) -> None:
    trials, labels = _synthetic_trials()
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_name="linear_svm",
        apply_filter=False,
        apply_training_ea=True,
    )
    pipeline = OnlinePipeline(
        model,
        EMGRejector(EmgRejectConfig(threshold=5.0)),
    )
    # Rest reference from quiet filtered-like noise
    rest = np.random.default_rng(3).normal(size=(750, 8)) * 0.2
    pipeline.add_rest(rest)
    assert pipeline.ea_ready

    clean_eeg = trials[0].T[:500]
    window = ImageryWindow(
        index=0,
        start=0.0,
        end=2.0,
        eeg=clean_eeg,
        eeg_times=np.arange(clean_eeg.shape[0]) / 250.0,
        emg=np.ones((clean_eeg.shape[0], 2)) * 0.1,
        emg_times=np.arange(clean_eeg.shape[0]) / 250.0,
        complete=True,
    )
    decision = pipeline.process_window(window)
    assert not decision.emg_rejected
    assert decision.label in {"left", "right"}

    dirty = ImageryWindow(
        index=1,
        start=0.5,
        end=2.5,
        eeg=clean_eeg,
        eeg_times=window.eeg_times,
        emg=np.ones((clean_eeg.shape[0], 2)) * 40.0,
        emg_times=window.emg_times,
        complete=True,
    )
    rejected = pipeline.process_window(dirty)
    assert rejected.emg_rejected
    assert rejected.label is None
    assert rejected.reason == "emg_contaminated"


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, float(seconds))


class _ClockFedStream(OnlineStream):
    """Push EEG/EMG up to the fake clock so window onsets stay aligned."""

    def __init__(self, clock: _FakeClock, **kwargs) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._cursor = 0.0
        self._rng = np.random.default_rng(4)

    def poll(self, timeout: float = 0.05) -> int:
        del timeout
        target = float(self._clock())
        if target <= self._cursor:
            return 0
        rate = self.sample_rate_hz
        times = np.arange(self._cursor, target, 1.0 / rate)
        if times.size == 0:
            self._cursor = target
            return 0
        eeg = self._rng.normal(size=(times.size, self.n_eeg_channels)) * 0.3
        emg = self._rng.normal(size=(times.size, self.n_emg_channels)) * 0.05
        self.push_eeg(eeg, times.tolist())
        self.push_emg(emg, times.tolist())
        self._cursor = float(times[-1] + 1.0 / rate)
        return int(times.size)


def test_trial_loop_runs_with_injected_stream(tmp_path: Path) -> None:
    trials, labels = _synthetic_trials(n_per_class=16)
    model = train_frozen_model(
        trials,
        labels,
        sample_rate_hz=250.0,
        classifier_name="logistic_regression",
        apply_filter=True,
    )
    clock = _FakeClock()
    stream = _ClockFedStream(
        clock,
        sample_rate_hz=250.0,
        n_eeg_channels=8,
        n_emg_channels=2,
        filter_spec=model.filter,
        apply_causal_filter=True,
    )
    pipeline = OnlinePipeline(model, EMGRejector(EmgRejectConfig(threshold=10.0)))
    loop = TrialLoop(
        stream,
        WindowManager(stream),
        pipeline,
        DecisionGate(DecisionGateConfig(min_consecutive=3, majority_of=3)),
        AvatarController(AvatarConfig(steps_to_goal=3)),
        OnlineLogger(tmp_path / "online.jsonl"),
        TrialTiming(cue_seconds=2.0, imagery_seconds=4.0, rest_seconds=3.0, poll_seconds=0.2),
        clock=clock,
        sleep=clock.sleep,
    )
    summary = loop.run_session(["left", "right"])
    assert summary["n_trials"] == 2
    assert "online_accuracy" in summary
    assert summary["offline_accuracy"] == model.offline_accuracy
    assert (tmp_path / "online.jsonl").exists()
    text = (tmp_path / "online.jsonl").read_text(encoding="utf-8")
    assert "window" in text
    assert "gate" in text

"""Online motor-imagery control: frozen CSP + LR/SVM with EMG gating."""

from pipeline.online.avatar import AvatarConfig, AvatarController
from pipeline.online.dataset_model import train_frozen_from_dataset
from pipeline.online.emg_reject import EMGRejector, EmgRejectConfig
from pipeline.online.gate import DecisionGate, DecisionGateConfig
from pipeline.online.log import OnlineLogger
from pipeline.online.model import (
    ONLINE_CLASSIFIERS,
    FrozenClassifierHead,
    FrozenOnlineModel,
    load_frozen_model,
    parse_classifier_names,
    save_frozen_model,
    train_frozen_model,
)
from pipeline.online.pipeline import OnlinePipeline
from pipeline.online.registry import (
    OfflineModeSpec,
    OnlineModelRegistry,
    RegistryEntry,
    export_offline_modes,
    load_offline_modes,
)
from pipeline.online.replay_session import ReplayOnlineConfig, ReplayOnlineSession
from pipeline.online.simulator_session import SimulatorOnlineConfig, SimulatorOnlineSession
from pipeline.online.stream import OnlineStream
from pipeline.online.trial_loop import TrialLoop, TrialTiming
from pipeline.online.windows import ImageryWindow, WindowManager

__all__ = [
    "ONLINE_CLASSIFIERS",
    "AvatarConfig",
    "AvatarController",
    "DecisionGate",
    "DecisionGateConfig",
    "EMGRejector",
    "EmgRejectConfig",
    "FrozenClassifierHead",
    "FrozenOnlineModel",
    "ImageryWindow",
    "OfflineModeSpec",
    "OnlineLogger",
    "OnlineModelRegistry",
    "OnlinePipeline",
    "OnlineStream",
    "RegistryEntry",
    "ReplayOnlineConfig",
    "ReplayOnlineSession",
    "SimulatorOnlineConfig",
    "SimulatorOnlineSession",
    "TrialLoop",
    "TrialTiming",
    "WindowManager",
    "export_offline_modes",
    "load_frozen_model",
    "load_offline_modes",
    "parse_classifier_names",
    "save_frozen_model",
    "train_frozen_from_dataset",
    "train_frozen_model",
]

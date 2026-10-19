"""Tunable values for the pre-model steps. Omitted fields keep the current defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelParams:
    notch_q: float = 30.0
    hjorth_degrees: float = 36.0
    laplacian_radius_degrees: float = 70.0
    laplacian_sigma_degrees: float = 30.0
    spline_order: int = 4
    spline_ridge: float = 1e-5
    ts_points: int = 32
    csp_pairs: int = 2
    fbcsp_low_hz: float = 4.0
    fbcsp_high_hz: float = 40.0
    fbcsp_band_hz: float = 4.0
    fbcsp_order: int = 4
    fbcsp_keep: int = 4
    theta_low_hz: float = 4.0
    theta_high_hz: float = 8.0
    mu_low_hz: float = 8.0
    mu_high_hz: float = 12.0
    beta_low_hz: float = 13.0
    beta_high_hz: float = 30.0
    regularization_c: float = 1.0
    neighbors: int = 5
    gamma: float | str = "scale"
    max_depth: int | None = None
    n_estimators: int | None = None
    max_features: float = 0.5
    shrinkage: float | None = None
    folds: int = 5
    test_fraction: float = 0.2
    validation_fraction: float = 0.2
    inner_folds: int = 3
    search_budget: int = 4
    var_smoothing: float | None = None
    min_samples_leaf: int | None = None
    knn_weights: str = "uniform"
    learning_rate: float | None = None

    def psd_bands(self) -> tuple[tuple[str, float, float], ...]:
        return (
            ("theta", self.theta_low_hz, self.theta_high_hz),
            ("mu", self.mu_low_hz, self.mu_high_hz),
            ("beta", self.beta_low_hz, self.beta_high_hz),
        )


def model_params(raw: object) -> ModelParams:
    """Read a request object. Unknown or blank fields stay at the default."""
    if not isinstance(raw, dict):
        return ModelParams()
    base = ModelParams()
    values: dict[str, Any] = {}
    floats = (
        "notch_q",
        "hjorth_degrees",
        "laplacian_radius_degrees",
        "laplacian_sigma_degrees",
        "spline_ridge",
        "fbcsp_low_hz",
        "fbcsp_high_hz",
        "fbcsp_band_hz",
        "theta_low_hz",
        "theta_high_hz",
        "mu_low_hz",
        "mu_high_hz",
        "beta_low_hz",
        "beta_high_hz",
        "regularization_c",
        "max_features",
        "test_fraction",
        "validation_fraction",
        "var_smoothing",
        "learning_rate",
    )
    ints = (
        "spline_order",
        "ts_points",
        "csp_pairs",
        "fbcsp_order",
        "fbcsp_keep",
        "neighbors",
        "folds",
        "inner_folds",
        "search_budget",
    )
    for key in floats:
        parsed = _optional_float(raw.get(key))
        if parsed is not None:
            values[key] = parsed
    for key in ints:
        parsed_int = _optional_int(raw.get(key))
        if parsed_int is not None:
            values[key] = parsed_int
    if "shrinkage" in raw and raw.get("shrinkage") not in (None, ""):
        values["shrinkage"] = float(raw["shrinkage"])
    if "max_depth" in raw and raw.get("max_depth") not in (None, "", 0):
        values["max_depth"] = int(raw["max_depth"])
    if "n_estimators" in raw and raw.get("n_estimators") not in (None, "", 0):
        values["n_estimators"] = int(raw["n_estimators"])
    if "min_samples_leaf" in raw and raw.get("min_samples_leaf") not in (None, "", 0):
        values["min_samples_leaf"] = int(raw["min_samples_leaf"])
    weights = raw.get("knn_weights")
    if isinstance(weights, str) and weights.strip().lower() in {"uniform", "distance"}:
        values["knn_weights"] = weights.strip().lower()
    gamma = raw.get("gamma", base.gamma)
    if isinstance(gamma, str) and gamma.strip().lower() == "scale":
        values["gamma"] = "scale"
    elif gamma not in (None, ""):
        values["gamma"] = float(gamma)
    return ModelParams(**values)


def with_updates(base: ModelParams, updates: dict[str, Any]) -> ModelParams:
    """Copy ``base`` and replace the named fields."""
    data = {name: getattr(base, name) for name in ModelParams.__dataclass_fields__}
    for key, value in updates.items():
        if key not in data:
            raise ValueError(f"Unknown hyperparameter {key}")
        data[key] = value
    return ModelParams(**data)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)

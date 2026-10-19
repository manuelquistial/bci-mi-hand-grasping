"""10-10 positions for the datasets the pre-model can open.

Coordinates are the FieldTrip ``standard_1005`` locations, recentered and
scaled onto the unit sphere. Nose is +Y, right ear is +X, vertex is +Z.
``T3/T4/T5/T6`` are the older names for ``T7/T8/P7/P8``.
"""

from __future__ import annotations

import math

import numpy as np

_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Unit vectors. Local neighbors sit about 30° apart; C3 is about 64° from Cz.
ELECTRODES: dict[str, tuple[float, float, float]] = {
    "FP1": (-0.27423, 0.89375, -0.35497),
    "FPZ": (-0.00590, 0.94988, -0.31255),
    "FP2": (0.26308, 0.89757, -0.35378),
    "AF7": (-0.51161, 0.76421, -0.39274),
    "AF3": (-0.35087, 0.92987, -0.11057),
    "AFZ": (-0.00544, 0.99937, 0.03498),
    "AF4": (0.35275, 0.93012, -0.10218),
    "AF8": (0.50337, 0.76998, -0.39211),
    "F7": (-0.70392, 0.56440, -0.43123),
    "F5": (-0.71202, 0.68236, -0.16554),
    "F3": (-0.59811, 0.79259, 0.11857),
    "F1": (-0.34524, 0.87264, 0.34540),
    "FZ": (-0.00542, 0.90463, 0.42617),
    "F2": (0.34947, 0.87555, 0.33358),
    "F4": (0.59326, 0.79860, 0.10139),
    "F6": (0.71227, 0.68183, -0.16667),
    "F8": (0.70095, 0.57094, -0.42743),
    "FT7": (-0.84396, 0.29591, -0.44740),
    "FT8": (0.83840, 0.30903, -0.44899),
    "FC5": (-0.91675, 0.38929, -0.08964),
    "FC3": (-0.81095, 0.49483, 0.31224),
    "FC1": (-0.48533, 0.56428, 0.66786),
    "FCZ": (-0.00531, 0.59466, 0.80396),
    "FC2": (0.48062, 0.57760, 0.65983),
    "FC4": (0.80809, 0.50140, 0.30918),
    "FC6": (0.91285, 0.39857, -0.08858),
    "T7": (-0.89861, -0.01646, -0.43844),
    "T8": (0.89692, -0.00592, -0.44216),
    "T9": (-0.73312, -0.01155, -0.68000),
    "T10": (0.72582, -0.01624, -0.68769),
    "C5": (-0.99931, 0.00868, -0.03605),
    "C3": (-0.89796, 0.03847, 0.43840),
    "C1": (-0.53790, 0.06529, 0.84048),
    "CZ": (-0.00510, 0.07747, 0.99698),
    "C2": (0.54679, 0.07167, 0.83419),
    "C4": (0.90236, 0.04845, 0.42824),
    "C6": (0.99919, 0.02038, -0.03474),
    "TP7": (-0.86222, -0.31795, -0.39433),
    "TP8": (0.86124, -0.31567, -0.39826),
    "CP5": (-0.92859, -0.37086, -0.01311),
    "CP3": (-0.80893, -0.40940, 0.42192),
    "CP1": (-0.47206, -0.42735, 0.77106),
    "CPZ": (-0.00486, -0.43843, 0.89875),
    "CP2": (0.48933, -0.42400, 0.76209),
    "CP4": (0.81723, -0.39922, 0.41564),
    "CP6": (0.93376, -0.35777, -0.00992),
    "P7": (-0.73072, -0.58899, -0.34517),
    "P5": (-0.73940, -0.67206, -0.04023),
    "P3": (-0.61677, -0.73802, 0.27372),
    "P1": (-0.34841, -0.78366, 0.51428),
    "PZ": (-0.00508, -0.79686, 0.60414),
    "P2": (0.36425, -0.77155, 0.52157),
    "P4": (0.62489, -0.72934, 0.27853),
    "P6": (0.73701, -0.67446, -0.04382),
    "P8": (0.72812, -0.59015, -0.34866),
    "PO7": (-0.53373, -0.79752, -0.28123),
    "PO3": (-0.39547, -0.91689, 0.05396),
    "POZ": (-0.00596, -0.97840, 0.20663),
    "PO4": (0.38455, -0.92195, 0.04604),
    "PO8": (0.52860, -0.80047, -0.28254),
    "O1": (-0.28693, -0.93209, -0.22111),
    "OZ": (-0.00630, -0.98526, -0.17096),
    "O2": (0.27825, -0.93434, -0.22268),
    "IZ": (-0.00633, -0.88360, -0.46820),
}


def canon(name: str) -> str:
    key = "".join(str(name).split()).upper()
    return _ALIASES.get(key, key)


def positions(channels: list[str]) -> tuple[list[str], np.ndarray]:
    """Return ``(labels, unit vectors)`` for channels that have a 10-10 position."""
    labels: list[str] = []
    rows: list[tuple[float, float, float]] = []
    missing: list[str] = []
    for name in channels:
        key = canon(name)
        point = ELECTRODES.get(key)
        if point is None:
            missing.append(name)
            continue
        labels.append(name)
        rows.append(point)
    if missing:
        shown = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
        raise ValueError(
            "This montage has no 10-10 position for "
            f"{shown}{extra}. CAR does not need positions."
        )
    return labels, np.asarray(rows, dtype=np.float64)


def angles(points: np.ndarray) -> np.ndarray:
    """Pairwise geodesic angle in radians. Shape ``(n, n)``."""
    cosine = np.clip(points @ points.T, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    return np.arccos(cosine)


def surrounded(origin: np.ndarray, neighbors: np.ndarray) -> bool:
    """True when neighbors wrap the electrode instead of sitting on one side."""
    if neighbors.shape[0] < 3:
        return False
    helper = np.array([1.0, 0.0, 0.0]) if abs(origin[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(origin, helper)
    first = first / np.linalg.norm(first)
    second = np.cross(origin, first)
    offsets = neighbors - origin
    tangent = np.arctan2(offsets @ second, offsets @ first)
    ordered = np.sort(tangent)
    gaps = np.diff(ordered, append=ordered[0] + 2 * math.pi)
    return float(np.max(gaps)) <= math.pi

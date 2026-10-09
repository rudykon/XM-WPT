"""Derive screening and receiver sensitivity from saved simulation fields.

This module never propagates or optimizes a field. Historical results use the
configuration snapshot stored in ``summary.json``, not the current run config.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .model import coverage
from .paths import RESULTS_DIR


@dataclass(frozen=True)
class StudyData:
    """Validated baseline and optimization result snapshots."""

    directory: Path
    config: dict
    baseline: dict
    optimized: dict


@dataclass(frozen=True)
class FieldData:
    """An already-saved field, detached from its closed NPZ file."""

    side_m: float
    path: Path
    arrays: dict
    radiated_w: float


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved simulation input: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require(mapping, keys, description):
    missing = set(keys).difference(mapping)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{description} is missing: {names}")


def _close(actual, expected, description):
    if not np.isclose(actual, expected, rtol=1e-8, atol=1e-11):
        raise ValueError(
            f"Inconsistent saved results ({description}): "
            f"{actual!r} does not match {expected!r}."
        )


def load_study(data_dir=RESULTS_DIR):
    """Load compatible result snapshots without reading the live config.

    Legacy optimization files have no config snapshot. Their geometric
    starting values and power accounting must match the baseline snapshot.
    This detects incompatible saved runs without inventing missing metadata.
    """
    directory = Path(data_dir)
    baseline = _read_json(directory / "summary.json")
    optimized = _read_json(directory / "optimized_summary.json")
    _require(baseline, ("config", "scan"), "summary.json")
    _require(
        optimized, ("optimized_scan", "best_sampled"),
        "optimized_summary.json",
    )
    config = baseline["config"]
    if not isinstance(config, dict):
        raise ValueError("summary.json config must be a JSON object.")
    required_config = (
        "phone_width_m", "phone_height_m", "phone_envelope_width_m",
        "phone_envelope_height_m", "rx_collection_efficiency",
        "rectifier_efficiency", "pmic_efficiency", "tx_radiation_efficiency",
        "psu_efficiency", "pa_efficiency", "aux_wall_w",
        "reference_distance_m", "target_dc_w", "public_e_rms_limit_v_m",
        "public_h_rms_limit_a_m", "tx_channels_per_axis",
    )
    _require(config, required_config, "Saved config")
    for key in required_config:
        value = config[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"Saved config {key} must be a finite number.")
        if value < 0 or (value == 0 and key != "aux_wall_w"):
            raise ValueError(
                f"Saved config {key} has an invalid value: {value}"
            )

    if "config" in optimized and optimized["config"] != config:
        other = optimized["config"]
        differences = (
            sorted(key for key in set(config) | set(other)
                   if config.get(key) != other.get(key))
            if isinstance(other, dict) else ["config type"]
        )
        raise ValueError(
            "summary.json and optimized_summary.json belong to different "
            f"configuration snapshots: {', '.join(differences)}"
        )

    rows = optimized["optimized_scan"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("optimized_summary.json has no optimized_scan rows.")
    if not isinstance(baseline["scan"], list) or not baseline["scan"]:
        raise ValueError("summary.json has no baseline scan rows.")
    required_row = (
        "tx_side_m", "distance_m", "capture_fraction",
        "envelope_capture_fraction", "rf_feed_to_load_efficiency",
        "rf_feed_for_5w_w", "radiated_for_5w_w", "wall_for_5w_w",
        "wall_efficiency_at_5w", "initial_geometric_capture",
    )
    for index, row in enumerate(rows):
        _require(row, required_row, f"Optimized row {index}")
        side = row["tx_side_m"]
        _close(row["distance_m"], config["reference_distance_m"],
               f"distance for D={side}")
        matches = [
            item for item in baseline["scan"]
            if item.get("architecture") == "144"
            and np.isclose(item.get("tx_side_m", np.nan), side)
            and np.isclose(item.get("distance_m", np.nan), row["distance_m"])
        ]
        if len(matches) != 1:
            raise ValueError(
                f"No unique matching baseline row for optimized D={side} m."
            )
        _close(
            row["initial_geometric_capture"], matches[0]["capture_fraction"],
            f"baseline capture for D={side}",
        )
        eta = (
            config["tx_radiation_efficiency"] * row["capture_fraction"]
            * config["rx_collection_efficiency"]
            * config["rectifier_efficiency"] * config["pmic_efficiency"]
        )
        if eta <= 0:
            raise ValueError(
                f"Nonpositive saved transfer efficiency for D={side}."
            )
        feed = config["target_dc_w"] / eta
        wall = feed / (config["psu_efficiency"] * config["pa_efficiency"])
        wall += config["aux_wall_w"]
        expected = {
            "rf_feed_to_load_efficiency": eta,
            "rf_feed_for_5w_w": feed,
            "radiated_for_5w_w": feed * config["tx_radiation_efficiency"],
            "wall_for_5w_w": wall,
            "wall_efficiency_at_5w": config["target_dc_w"] / wall,
        }
        for key, value in expected.items():
            _close(row[key], value, f"{key} for D={side}")

    best = optimized["best_sampled"]
    _require(best, required_row, "best_sampled")
    matches = [row for row in rows if row == best]
    if len(matches) != 1:
        raise ValueError("best_sampled must match one optimized_scan row.")
    return StudyData(directory, config, baseline, optimized)


def load_fields(study):
    """Load available optimized fields; never synthesize a missing aperture."""
    fields = []
    for row in sorted(study.optimized["optimized_scan"],
                      key=lambda item: item["tx_side_m"]):
        side = row["tx_side_m"]
        path = study.directory / f"optimized_fields_D{side:.3f}.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read saved field {path}: {exc}") from exc
        _require(
            arrays,
            ("x", "z", "slice_x", "target_sz", "slice_e2", "slice_h2",
             "radiated_for_5w_w"),
            str(path),
        )
        for key, array in arrays.items():
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Nonfinite values in {path}: {key}")
        for key in ("x", "z", "slice_x"):
            axis = arrays[key]
            if axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(
                    f"{path}: {key} must be an increasing 1D grid."
                )
        x = arrays["x"]
        if not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-8):
            raise ValueError(f"{path}: target x grid must be evenly spaced.")
        if arrays["target_sz"].shape != (len(x), len(x)):
            raise ValueError(f"{path}: target_sz does not match its x grid.")
        shape = (len(arrays["z"]), len(arrays["slice_x"]))
        for key in ("slice_e2", "slice_h2"):
            if arrays[key].shape != shape or np.any(arrays[key] < 0):
                raise ValueError(
                    f"{path}: invalid {key} dimensions or values."
                )
        radiated = float(arrays["radiated_for_5w_w"])
        _close(
            radiated, row["radiated_for_5w_w"],
            f"field scaling in {path.name}",
        )
        fields.append(FieldData(side, path, arrays, radiated))
    if not fields:
        raise FileNotFoundError(
            f"No saved optimized_fields_D*.npz matching the scan in "
            f"{study.directory}; generate fields explicitly before analysis."
        )
    return fields


def screening_ratio(field, config):
    """Return the larger E/H screening ratio using the saved limit values."""
    arrays = field.arrays
    return np.maximum(
        np.sqrt(arrays["slice_e2"] * field.radiated_w)
        / config["public_e_rms_limit_v_m"],
        np.sqrt(arrays["slice_h2"] * field.radiated_w)
        / config["public_h_rms_limit_a_m"],
    )


def point_screening(fields, config):
    """Sample the existing field grid at the report's four reference points."""
    distance = config["reference_distance_m"]
    positions = (
        (1.0, 0.0, "主轴1m"), (2.0, 0.0, "主轴2m"),
        (distance, 0.0, "手机中心"),
        (distance, 0.25, "同平面横向25cm"),
    )
    points = []
    for field in fields:
        arrays = field.arrays
        ratio = screening_ratio(field, config)
        for z, x, label in positions:
            if not (arrays["z"][0] <= z <= arrays["z"][-1]
                    and arrays["slice_x"][0] <= x <= arrays["slice_x"][-1]):
                raise ValueError(
                    f"{field.path}: screening point {label} is outside "
                    "the saved grid."
                )
            iz = int(np.argmin(abs(arrays["z"] - z)))
            ix = int(np.argmin(abs(arrays["slice_x"] - x)))
            points.append({
                "tx_side_m": field.side_m, "label": label,
                "x_m": float(arrays["slice_x"][ix]),
                "z_m": float(arrays["z"][iz]),
                "e_rms_v_m": float(np.sqrt(
                    arrays["slice_e2"][iz, ix] * field.radiated_w)),
                "h_rms_a_m": float(np.sqrt(
                    arrays["slice_h2"][iz, ix] * field.radiated_w)),
                "limit_ratio": float(ratio[iz, ix]),
            })
    return points


def receiver_sensitivity(study, fields):
    """Integrate the stored signed flux; keep the optimized beam unchanged."""
    config = study.config
    best = study.optimized["best_sampled"]
    matches = [field for field in fields
               if np.isclose(field.side_m, best["tx_side_m"])]
    if len(matches) != 1:
        raise FileNotFoundError(
            "Receiver sensitivity requires the saved best-aperture field: "
            f"optimized_fields_D{best['tx_side_m']:.3f}.npz"
        )
    field = matches[0]
    x = field.arrays["x"]
    dx = float(x[1] - x[0])
    sensitivity = []
    for error in (0, 0.005, 0.01, 0.02, 0.03, 0.05):
        weights = (
            coverage(x, config["phone_width_m"], dx, error)[None, :]
            * coverage(x, config["phone_height_m"], dx)[:, None]
        )
        load = float(
            np.sum(field.arrays["target_sz"] * weights) * dx**2
            * field.radiated_w * config["rx_collection_efficiency"]
            * config["rectifier_efficiency"] * config["pmic_efficiency"]
        )
        sensitivity.append({"lateral_error_m": error, "load_w": load})
    capture = best["capture_fraction"]
    envelope = best["envelope_capture_fraction"]
    if capture <= 0 or envelope <= 0:
        raise ValueError("Saved receiver capture fractions must be positive.")
    return {
        "fixed_best_rf_feed_w": best["rf_feed_for_5w_w"],
        "position_errors_x": sensitivity,
        "same_beam_98cm2_load_w": config["target_dc_w"] * envelope / capture,
        "same_beam_98cm2_rf_feed_for_5w_w": (
            best["rf_feed_for_5w_w"] * capture / envelope
        ),
    }


def run(output_dir: Path = RESULTS_DIR):
    """Write both derived JSON files alongside the validated saved inputs."""
    study = load_study(output_dir)
    fields = load_fields(study)
    points = point_screening(fields, study.config)
    sensitivity = receiver_sensitivity(study, fields)
    outputs = {
        "public_point_screening.json": points,
        "receiver_sensitivity.json": sensitivity,
    }
    for name, result in outputs.items():
        (study.directory / name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return {name: study.directory / name for name in outputs}

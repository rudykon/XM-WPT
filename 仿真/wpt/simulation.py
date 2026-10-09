"""Explicit baseline and optimized simulation entry points.

Importing this module does not create directories or run simulations. Output
names and historical metric keys remain compatible with the existing report.
"""

import csv
import json
from pathlib import Path
import time

import numpy as np

from .config import (
    CFG,
    CHECK_GRIDS,
    CONTROL_TX_SIDES_M,
    DISTANCE_SCAN_M,
    DISTANCE_TX_SIDES_M,
    FIELD_REFERENCE_SIDE_M,
    GRADIENT_CHECK_DIRECTION_SEED,
    GRADIENT_CHECK_PHASE_SEED,
    LONGITUDINAL_SAMPLES,
    LONGITUDINAL_START_M,
    LONGITUDINAL_STOP_M,
    OPTIMIZED_TX_SIDES_M,
    TARGET_FIELD_HALF_WIDTH_M,
)
from .model import Aperture, fresnel_capture, quantize_phase
from .optimization import PhaseOptimizer
from .paths import RESULTS_DIR


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _power_scan(rows, config, include_architecture=False):
    """Apply the same fixed conversion chain to every feed-power sample."""
    output = []
    for row in rows:
        for power in config["rf_feed_power_w"]:
            load = power * row["rf_feed_to_load_efficiency"]
            wall = (
                power / (config["psu_efficiency"] * config["pa_efficiency"])
                + config["aux_wall_w"]
            )
            item = {}
            if include_architecture:
                item["architecture"] = row["architecture"]
            item.update({
                "tx_side_m": row["tx_side_m"],
                "rf_feed_w": power,
                "load_w": load,
                "wall_input_w": wall,
                "wall_efficiency": load / wall,
            })
            output.append(item)
    return output


def _save_fields(output_dir, grid, spectrum, row, prefix):
    """Save the target plane and longitudinal slice in the existing format."""
    side, distance = row["tx_side_m"], row["distance_m"]
    plane = grid.plane(spectrum, distance)
    keep = np.abs(grid.x) <= TARGET_FIELD_HALF_WIDTH_M
    z_values = np.linspace(
        LONGITUDINAL_START_M, LONGITUDINAL_STOP_M, LONGITUDINAL_SAMPLES
    )
    xs, sz, e2, h2 = grid.longitudinal(spectrum, z_values)
    np.savez_compressed(
        output_dir / f"{prefix}_D{side:.3f}.npz",
        x=grid.x[keep],
        target_sz=plane["sz"][np.ix_(keep, keep)],
        target_e2=plane["e2"][np.ix_(keep, keep)],
        target_h2=plane["h2"][np.ix_(keep, keep)],
        z=z_values,
        slice_x=xs,
        slice_sz=sz,
        slice_e2=e2,
        slice_h2=h2,
        radiated_for_5w_w=row["radiated_for_5w_w"],
    )


def _baseline_row(grid, side, distance, architecture, bits):
    spectrum = grid.source(side, distance, architecture, bits)
    return {
        "architecture": architecture,
        "tx_side_m": side,
        "distance_m": distance,
        **grid.summarize(grid.plane(spectrum, distance)),
    }


def run_baseline(output_dir: Path = RESULTS_DIR):
    """Run baseline aperture, power, distance, and numerical-check scans."""
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = Aperture(config=CFG)
    cfg = grid.config
    distance = cfg["reference_distance_m"]
    phase_bits = cfg["phase_bits"]
    rows = []
    for architecture in ["144", "continuous"]:
        for side in cfg["tx_side_m"]:
            bits = phase_bits if architecture == "144" else None
            row = _baseline_row(grid, side, distance, architecture, bits)
            row["fresnel_capture_fraction"] = fresnel_capture(
                side, distance, config=cfg
            )
            row["wall_efficiency_at_5w"] = (
                cfg["target_dc_w"] / row["wall_for_5w_w"]
            )
            rows.append(row)
            print(
                f"scan {architecture} D={side:.3f}: "
                f"capture={row['capture_fraction']:.5f}, "
                f"RF5={row['rf_feed_for_5w_w']:.2f}W", flush=True,
            )
    power_rows = _power_scan(rows, cfg, include_architecture=True)

    distance_rows = []
    for scan_distance in DISTANCE_SCAN_M:
        for side in DISTANCE_TX_SIDES_M:
            row = _baseline_row(grid, side, scan_distance, "144", phase_bits)
            distance_rows.append(row)
            print(
                f"distance {scan_distance}m D={side}: "
                f"RF5={row['rf_feed_for_5w_w']:.2f}W", flush=True,
            )

    best = min(
        (row for row in rows if row["architecture"] == "144"),
        key=lambda row: row["rf_feed_for_5w_w"],
    )
    checks = []
    check_cases = (
        (0.03, distance, "144"),
        (0.20, distance, "144"),
        (0.60, distance, "144"),
        (0.10, 5.0, "144"),
    )
    for side, check_distance, architecture in check_cases:
        source_rows = rows if check_distance == distance else distance_rows
        base = next(
            row for row in source_rows
            if row["tx_side_m"] == side
            and row["distance_m"] == check_distance
            and row["architecture"] == architecture
        )
        for label, n, dx_fraction in CHECK_GRIDS:
            test = Aperture(n, dx_fraction, config=cfg)
            result = _baseline_row(
                test, side, check_distance, architecture, phase_bits
            )
            checks.append({
                "type": label,
                "tx_side_m": side,
                "distance_m": check_distance,
                "capture_base": base["capture_fraction"],
                "capture_checked": result["capture_fraction"],
                "relative_change": (
                    result["capture_fraction"] / base["capture_fraction"] - 1
                ),
                "outside_e_ratio_change": (
                    result["plane_outside_20cm_peak_e_rms_at_5w_v_m"]
                    / base["plane_outside_20cm_peak_e_rms_at_5w_v_m"] - 1
                ),
                "plane_power": result["plane_power_w_per_radiated_w"],
            })
            print(
                f"check {label} D={side} R={check_distance}: "
                f"change={checks[-1]['relative_change']:.3%}", flush=True,
            )
            del test

    controls = []
    for side in CONTROL_TX_SIDES_M:
        for architecture, bits in [("144", None), ("unfocused", None)]:
            row = _baseline_row(grid, side, distance, architecture, bits)
            controls.append({
                "architecture": architecture,
                "phase_bits": bits,
                "tx_side_m": side,
                "capture_fraction": row["capture_fraction"],
                "rf_feed_for_5w_w": row["rf_feed_for_5w_w"],
            })
    max_conservation = max(
        abs(row["plane_power_w_per_radiated_w"] - 1)
        for row in rows + distance_rows
    )
    assert max_conservation < 1e-10
    output = {
        "config": cfg,
        "model": (
            "Forward propagating vector angular spectrum, transversely "
            "projected equivalent aperture, no room/body scattering"
        ),
        "scan": rows,
        "distance_sensitivity": distance_rows,
        "controls": controls,
        "numerical_checks": checks,
        "best_sampled_144": best,
        "max_plane_power_error": max_conservation,
        "elapsed_s": time.time() - start,
    }
    _write_json(output_dir / "summary.json", output)
    _write_json(output_dir / "config.json", cfg)
    _write_csv(output_dir / "aperture_scan.csv", rows)
    _write_csv(output_dir / "power_scan.csv", power_rows)
    _write_csv(output_dir / "distance_sensitivity.csv", distance_rows)
    _write_csv(output_dir / "numerical_checks.csv", checks)
    print(json.dumps({
        "best": best, "seconds": time.time() - start,
        "power_error": max_conservation,
    }), flush=True)
    return output


def run_optimized(output_dir: Path = RESULTS_DIR):
    """Optimize using the baseline's saved configuration and write results."""
    start = time.time()
    output_dir = Path(output_dir)
    base = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    grid = Aperture(config=base["config"])
    cfg = grid.config
    distance = cfg["reference_distance_m"]
    check = PhaseOptimizer(grid, 0.30, distance)
    count = check.channel_count
    phi = check.initial + np.random.default_rng(
        GRADIENT_CHECK_PHASE_SEED
    ).normal(0, 0.03, count)
    _, grad = check.evaluate(phi, True)
    direction = np.random.default_rng(
        GRADIENT_CHECK_DIRECTION_SEED
    ).normal(size=count)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    finite = (
        check.evaluate(phi + epsilon * direction)
        - check.evaluate(phi - epsilon * direction)
    ) / (2 * epsilon)
    analytic = float(np.dot(grad, direction))
    assert abs(finite - analytic) < 1e-7
    print(
        f"adjoint check analytic={analytic:.10g}, finite={finite:.10g}",
        flush=True,
    )

    rows, histories, phases = [], {}, {}
    for side in OPTIMIZED_TX_SIDES_M:
        optimizer = PhaseOptimizer(grid, side, distance)
        phi, history = optimizer.optimize()
        quantized = quantize_phase(phi, cfg["phase_bits"])
        spectrum = optimizer.spectrum(quantized)
        result = grid.summarize(grid.plane(spectrum, distance))
        old = next(
            row for row in base["scan"]
            if row["architecture"] == "144" and row["tx_side_m"] == side
        )
        row = {
            "architecture": "144_local_optimized",
            "tx_side_m": side,
            "distance_m": distance,
            **result,
            "continuous_phase_capture": optimizer.evaluate(phi),
            "initial_geometric_capture": old["capture_fraction"],
            "improvement_over_geometric_percent": (
                result["capture_fraction"] / old["capture_fraction"] - 1
            ) * 100,
            "iterations": len(history),
            "wall_efficiency_at_5w": (
                cfg["target_dc_w"] / result["wall_for_5w_w"]
            ),
            "optimization_status": optimizer.last_status,
        }
        rows.append(row)
        histories[str(side)] = history
        phases[str(side)] = quantized.tolist()
        print(
            f"optimized D={side:.3f} kappa={row['capture_fraction']:.5f} "
            f"RF5={row['rf_feed_for_5w_w']:.2f}W "
            f"improvement={row['improvement_over_geometric_percent']:.2f}% "
            f"iters={len(history)}", flush=True,
        )
    best = min(rows, key=lambda row: row["rf_feed_for_5w_w"])

    phase_checks = []
    for side in sorted(set([FIELD_REFERENCE_SIDE_M, best["tx_side_m"], 0.60])):
        row = next(row for row in rows if row["tx_side_m"] == side)
        for label, n, dx_fraction in CHECK_GRIDS:
            test = Aperture(n, dx_fraction, config=cfg)
            optimizer = PhaseOptimizer(test, side, distance)
            spectrum = optimizer.spectrum(np.array(phases[str(side)]))
            result = test.summarize(test.plane(spectrum, distance))
            phase_checks.append({
                "type": label,
                "tx_side_m": side,
                "capture": result["capture_fraction"],
                "relative_change": (
                    result["capture_fraction"] / row["capture_fraction"] - 1
                ),
                "outside_e_relative_change": (
                    result["plane_outside_20cm_peak_e_rms_at_5w_v_m"]
                    / row["plane_outside_20cm_peak_e_rms_at_5w_v_m"] - 1
                ),
                "plane_power": result["plane_power_w_per_radiated_w"],
            })
            del test, optimizer

    for side in sorted(set([FIELD_REFERENCE_SIDE_M, best["tx_side_m"]])):
        row = next(row for row in rows if row["tx_side_m"] == side)
        optimizer = PhaseOptimizer(grid, side, distance)
        spectrum = optimizer.spectrum(np.array(phases[str(side)]))
        _save_fields(output_dir, grid, spectrum, row, "optimized_fields")

    power_rows = _power_scan(rows, cfg)
    result = {
        "config": cfg,
        "optimized_scan": rows,
        "best_sampled": best,
        "phase_checks": phase_checks,
        "adjoint_check": {
            "analytic": analytic,
            "finite_difference": finite,
            "absolute_error": abs(analytic - finite),
        },
        "histories": histories,
        # Kept for existing consumers; config records the actual channel count.
        "phase_radians_12_by_12_flat": phases,
        "elapsed_s": time.time() - start,
    }
    _write_json(output_dir / "optimized_summary.json", result)
    _write_csv(output_dir / "optimized_aperture_scan.csv", rows)
    _write_csv(output_dir / "optimized_power_scan.csv", power_rows)
    print(json.dumps({
        "best": best, "elapsed": time.time() - start,
    }), flush=True)
    return result

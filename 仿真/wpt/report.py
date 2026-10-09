"""Build the native LaTeX report from saved numerical results.

The template preserves the edited report, figures, equations, and study scope.
Only computed result tables and the original numerical summary values are
substituted. No Markdown file or conversion step is required. Importing this
module performs no filesystem writes.
"""

from pathlib import Path
import json
import re
from typing import Any, Iterable, Sequence

from .paths import LATEX_REPORT, RESULTS_DIR

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
PLACEHOLDER = re.compile(r"@@([a-z][a-z_0-9]*)@@")
COMPARISON_SIDES = (0.20, 0.30, 0.40, 0.60)
DISTANCE_SIDES = (0.10, 0.20, 0.30, 0.40, 0.60)


def _read_json(directory: Path, name: str) -> Any:
    """Read one saved result without running the simulation."""
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _escape_cell(value: object) -> str:
    """Escape plain result cells; equations stay in the native template."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "κ": r"\ensuremath{\kappa}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render numerical rows using the report's verified LaTeX table layout."""
    count = len(headers)
    weights = {
        2: (0.51, 0.49),
        3: (0.23, 0.30, 0.47),
        4: (0.18, 0.26, 0.29, 0.27),
        5: (0.16, 0.23, 0.23, 0.21, 0.17),
        6: (0.13, 0.15, 0.21, 0.18, 0.17, 0.16),
    }[count]
    usable_mm = 166 - 4 * (count - 1)
    columns = "@{}" + "".join(
        r">{\raggedright\arraybackslash}p{" + f"{usable_mm * weight:.2f}mm" + "}"
        for weight in weights
    ) + "@{}"
    header = " & ".join(
        r"\textbf{" + _escape_cell(cell) + "}" for cell in headers
    ) + r" \\"
    lines = [
        r"\begingroup\small",
        r"\setlength{\tabcolsep}{2mm}\renewcommand{\arraystretch}{1.22}",
        r"\begin{longtable}{" + columns + "}",
        r"\toprule", header, r"\midrule\endfirsthead",
        r"\toprule", header, r"\midrule\endhead",
        r"\midrule\endfoot", r"\bottomrule\endlastfoot",
    ]
    for row in rows:
        if len(row) != count:
            raise ValueError("A report table row has the wrong column count.")
        lines.append(" & ".join(_escape_cell(cell) for cell in row) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


def _result_tables(
    baseline: dict,
    optimized: dict,
    receiver: dict,
    public_points: list[dict],
) -> dict[str, str]:
    """Build the nine tables whose values come from numerical output."""
    rows = optimized["optimized_scan"]
    best = optimized["best_sampled"]
    by_side = {row["tx_side_m"]: row for row in rows}
    base_by = {
        (row["architecture"], row["tx_side_m"]): row
        for row in baseline["scan"]
    }
    distances = {
        (row["tx_side_m"], row["distance_m"]): row
        for row in baseline["distance_sensitivity"]
    }

    tables = {}
    tables["aperture_table"] = _table(
        [
            "发射口径边长", "收能比例 κ", "5 W 所需射频馈入",
            "对应辐射功率", "墙插输入", "墙插效率",
        ],
        [
            [
                f"{row['tx_side_m'] * 100:g} cm",
                f"{row['capture_fraction'] * 100:.2f}%",
                f"{row['rf_feed_for_5w_w']:.2f} W",
                f"{row['radiated_for_5w_w']:.2f} W",
                f"{row['wall_for_5w_w']:.2f} W",
                f"{row['wall_efficiency_at_5w'] * 100:.3f}%",
            ]
            for row in rows
        ],
    )
    tables["architecture_table"] = _table(
        [
            "口径", "几何调相 RF / 5 W", "局部优化后 RF / 5 W",
            "收能相对增幅", "连续细调相 RF / 5 W",
        ],
        [
            [
                f"{side * 100:g} cm",
                f"{base_by['144', side]['rf_feed_for_5w_w']:.2f} W",
                f"{by_side[side]['rf_feed_for_5w_w']:.2f} W",
                f"{by_side[side]['improvement_over_geometric_percent']:.3f}%",
                f"{base_by['continuous', side]['rf_feed_for_5w_w']:.2f} W",
            ]
            for side in COMPARISON_SIDES
        ],
    )

    power_rows = []
    for power in (10, 20, 30, 50, 75, 100):
        load = power * best["rf_feed_to_load_efficiency"]
        config = baseline["config"]
        wall = power / (config["psu_efficiency"] * config["pa_efficiency"]) + config["aux_wall_w"]
        power_rows.append([
            f"{power} W", f"{load:.3f} W", f"{wall:.2f} W",
            f"{100 * load / wall:.3f}%",
        ])
    tables["power_table"] = _table(
        ["射频馈入", "负载功率", "墙插输入", "墙插效率"], power_rows
    )
    tables["position_error_table"] = _table(
        ["横向位置误差", "固定波束下负载功率"],
        [
            [f"{row['lateral_error_m'] * 100:g} cm", f"{row['load_w']:.3f} W"]
            for row in receiver["position_errors_x"]
        ],
    )
    tables["field_peaks_table"] = _table(
        [
            "口径", "手机面峰值 E", "手机面峰值 H",
            "所选外部区域峰值 E", "所选外部区域峰值 H",
        ],
        [
            [
                f"{side * 100:g} cm",
                f"{by_side[side]['rx_peak_e_rms_at_5w_v_m']:.0f} V/m",
                f"{by_side[side]['rx_peak_h_rms_at_5w_a_m']:.3f} A/m",
                (
                    f"{by_side[side]['plane_outside_20cm_peak_e_rms_at_5w_v_m']:.0f}"
                    " V/m"
                ),
                (
                    f"{by_side[side]['plane_outside_20cm_peak_h_rms_at_5w_a_m']:.3f}"
                    " A/m"
                ),
            ]
            for side in (0.20, 0.30, 0.40)
        ],
    )
    tables["exposure_table"] = _table(
        ["口径", "受电面 E/H 筛查允许的 RF 馈入", "相应负载上限"],
        [
            [
                f"{side * 100:g} cm",
                (
                    f"{by_side[side]['rx_plane_eh_constrained_rf_feed_w'] * 1000:.3f}"
                    " mW"
                ),
                (
                    f"{by_side[side]['rx_plane_eh_constrained_load_w'] * 1000:.3f}"
                    " mW"
                ),
            ]
            for side in DISTANCE_SIDES
        ],
    )
    tables["public_points_table"] = _table(
        ["位置", "E 有效值", "H 有效值", "两种场量中较大的超限比"],
        [
            [
                row["label"], f"{row['e_rms_v_m']:.1f} V/m",
                f"{row['h_rms_a_m']:.3f} A/m", f"{row['limit_ratio']:.2f}",
            ]
            for row in public_points
            if row["tx_side_m"] == best["tx_side_m"]
        ],
    )
    tables["distance_table"] = _table(
        ["口径", "1 m 时 RF / 5 W", "3 m 时 RF / 5 W", "5 m 时 RF / 5 W"],
        [
            [
                f"{side * 100:g} cm",
                f"{distances[side, 1.0]['rf_feed_for_5w_w']:.2f} W",
                f"{base_by['144', side]['rf_feed_for_5w_w']:.2f} W",
                f"{distances[side, 5.0]['rf_feed_for_5w_w']:.2f} W",
            ]
            for side in DISTANCE_SIDES
        ],
    )
    tables["numerical_checks_table"] = _table(
        ["口径", "核查方式", "手机收能比例变化", "外部区域 E 峰值变化"],
        [
            [
                f"{row['tx_side_m'] * 100:g} cm",
                "采样加密" if row["type"] == "resolution" else "窗口扩大",
                f"{row['relative_change'] * 100:+.3f}%",
                f"{row['outside_e_relative_change'] * 100:+.2f}%",
            ]
            for row in optimized["phase_checks"]
        ],
    )
    return tables


def _summary_values(baseline: dict, optimized: dict) -> dict[str, str]:
    """Format summary values with the existing report's precision."""
    best = optimized["best_sampled"]
    config = baseline["config"]
    checks = optimized["phase_checks"]
    max_capture_change = max(abs(row["relative_change"]) for row in checks)
    max_field_change = max(
        abs(row["outside_e_relative_change"]) for row in checks
    )
    envelope_load = (
        config["target_dc_w"] * best["envelope_capture_fraction"] / best["capture_fraction"]
    )
    return {
        "aperture_count": str(len(optimized["optimized_scan"])),
        "best_side_cm": f"{best['tx_side_m'] * 100:.0f}",
        "best_feed_1": f"{best['rf_feed_for_5w_w']:.1f}",
        "best_feed_2": f"{best['rf_feed_for_5w_w']:.2f}",
        "best_wall_0": f"{best['wall_for_5w_w']:.0f}",
        "best_wall_efficiency_2": f"{best['wall_efficiency_at_5w'] * 100:.2f}",
        "load_at_50w_2": f"{50 * best['rf_feed_to_load_efficiency']:.2f}",
        "best_capture_percent_2": f"{best['capture_fraction'] * 100:.2f}",
        "envelope_capture_percent_2": (
            f"{best['envelope_capture_fraction'] * 100:.2f}"
        ),
        "envelope_load_2": f"{envelope_load:.2f}",
        "best_e_ratio_1": f"{best['rx_peak_e_rms_at_5w_v_m'] / config['public_e_rms_limit_v_m']:.1f}",
        "best_h_ratio_1": f"{best['rx_peak_h_rms_at_5w_a_m'] / config['public_h_rms_limit_a_m']:.1f}",
        "max_capture_change_percent_2": f"{max_capture_change * 100:.2f}",
        "max_outside_e_change_percent_1": f"{max_field_change * 100:.1f}",
        "plane_power_error": f"{baseline['max_plane_power_error']:.2g}",
        "adjoint_error": f"{optimized['adjoint_check']['absolute_error']:.2g}",
    }


def render_report(data_dir: Path = RESULTS_DIR) -> str:
    """Return native LaTeX from existing results without writing any files."""
    data_dir = Path(data_dir)
    from .analysis import load_study

    study = load_study(data_dir)
    baseline, optimized = study.baseline, study.optimized
    receiver = _read_json(data_dir, "receiver_sensitivity.json")
    public_points = _read_json(data_dir, "public_point_screening.json")
    values = _result_tables(baseline, optimized, receiver, public_points)
    values.update(_summary_values(baseline, optimized))
    template = (TEMPLATE_DIR / "report.tex").read_text(encoding="utf-8")
    missing = set(PLACEHOLDER.findall(template)) - values.keys()
    if missing:
        raise ValueError(f"Missing report template values: {sorted(missing)}")
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def main(
    output_path: Path = LATEX_REPORT,
    data_dir: Path = RESULTS_DIR,
) -> Path:
    """Write the report to the caller-selected path without simulating."""
    result = render_report(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    print(f"LaTeX report written: {output_path}")
    return output_path

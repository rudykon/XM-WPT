"""Render four report figures from saved results without running the solver."""

from pathlib import Path
import warnings

import numpy as np

from .analysis import load_fields, load_study, screening_ratio
from .paths import ASSETS_DIR, RESULTS_DIR


def _font_family(font_manager):
    candidates = (
        "Microsoft YaHei", "Microsoft YaHei UI", "Noto Sans CJK SC",
        "Source Han Sans SC", "Source Han Sans CN", "SimHei",
        "WenQuanYi Micro Hei", "Arial Unicode MS",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            return candidate
    warnings.warn(
        "No supported Chinese font was found. Falling back to DejaVu Sans; "
        "install Microsoft YaHei, Noto Sans CJK SC, or Source Han Sans SC "
        "to render the report's Chinese labels.",
        RuntimeWarning, stacklevel=2,
    )
    return "DejaVu Sans"


def _finish(plt, figure, directory, name):
    path = directory / f"{name}.png"
    try:
        figure.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return path


def _power_plot(plt, log_norm, study):
    rows = study.optimized["optimized_scan"]
    config = study.config
    x = np.array([row["tx_side_m"] for row in rows]) * 100
    power = np.geomspace(1, 300, 181)
    eta = np.array([row["rf_feed_to_load_efficiency"] for row in rows])
    load = power[:, None] * eta[None, :]
    area = config["phone_width_m"] * config["phone_height_m"] * 1e4
    channels = config["tx_channels_per_axis"] ** 2
    target = config["target_dc_w"]
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    art = axis.pcolormesh(
        x, power, load, norm=log_norm(0.02, 30), cmap="viridis",
        shading="nearest", rasterized=True,
    )
    axis.plot(
        x, [row["rf_feed_for_5w_w"] for row in rows], color="white", lw=2.3,
        label=f"{target:g} W 负载所需馈入",
    )
    axis.set(
        xlim=(3, 60), ylim=(1, 300), yscale="log",
        xlabel="方形发射口径边长 / cm", ylabel="发射端总射频馈入 / W",
        title=f"固定 {area:g} cm² 手机接收面：{channels:g} 通道局部优化后功率扫描",
    )
    figure.colorbar(art, ax=axis, label="负载直流功率 / W")
    axis.legend(loc="lower right")
    return figure


def _efficiency_plot(plt, study):
    config = study.config
    rows = study.optimized["optimized_scan"]
    x = np.array([row["tx_side_m"] for row in rows]) * 100
    target = config["target_dc_w"]
    channels = config["tx_channels_per_axis"] ** 2
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    styles = (
        ("144", f"{channels:g} 通道：几何调相", "#8A969B", "--"),
        ("continuous", "更细调相的理想口径", "#087E8B", "-"),
    )
    for architecture, label, color, line_style in styles:
        selected = [row for row in study.baseline["scan"]
                    if row["architecture"] == architecture]
        axes[0].semilogy(
            [row["tx_side_m"] * 100 for row in selected],
            [row["rf_feed_for_5w_w"] for row in selected],
            color=color, ls=line_style, label=label,
        )
    axes[0].semilogy(
        x, [row["rf_feed_for_5w_w"] for row in rows], "o-",
        color="#C85A36", ms=4, label=f"{channels:g} 通道：局部优化后",
    )
    complete_capture_eta = (
        config["tx_radiation_efficiency"] * config["rx_collection_efficiency"]
        * config["rectifier_efficiency"] * config["pmic_efficiency"]
    )
    axes[0].axhline(
        target / complete_capture_eta, color="#555555", ls=":",
        label="全部截获时的馈入功率下界",
    )
    axes[0].set(
        xlabel="发射口径边长 / cm", ylabel=f"实现 {target:g} W 的射频馈入 / W",
        title="尺寸收益受调相分区限制",
    )
    axes[0].legend(fontsize=11)
    axes[1].plot(
        x, [100 * row["rf_feed_to_load_efficiency"] for row in rows], "o-",
        color="#087E8B", ms=4, label="射频馈入到负载",
    )
    axes[1].plot(
        x, [100 * row["wall_efficiency_at_5w"] for row in rows], "s-",
        color="#C85A36", ms=4, label=f"墙插到负载（目标 {target:g} W）",
    )
    axes[1].set(
        xlabel="发射口径边长 / cm", ylabel="效率 / %",
        title=f"固定各级效率；辅助耗电假设 {config['aux_wall_w']:g} W",
    )
    axes[1].legend(fontsize=11)
    for axis in axes:
        axis.grid(alpha=0.2)
    return figure


def _path_field_plot(plt, log_norm, study, fields):
    config = study.config
    figure, axes_grid = plt.subplots(
        1, len(fields), figsize=(10, 4.8), layout="constrained",
        sharey=True, squeeze=False,
    )
    axes = axes_grid[0]
    for axis, field in zip(axes, fields):
        data = field.arrays
        ratio = screening_ratio(field, config)
        art = axis.pcolormesh(
            data["z"], data["slice_x"] * 100, ratio.T,
            norm=log_norm(0.1, 100), cmap="magma", shading="nearest",
            rasterized=True,
        )
        if np.min(ratio) < 1 < np.max(ratio):
            axis.contour(
                data["z"], data["slice_x"] * 100, ratio.T, levels=[1],
                colors=["cyan"], linewidths=0.8,
            )
        distance = config["reference_distance_m"]
        half_width_cm = config["phone_width_m"] * 50
        axis.plot(
            [distance, distance], [-half_width_cm, half_width_cm],
            color="white", lw=4,
        )
        axis.set(
            xlabel="距发射面 z / m", ylim=(-65, 65),
            title=(f"口径 {field.side_m * 100:g} cm，"
                   f"负载 {config['target_dc_w']:g} W"),
        )
    axes[0].set_ylabel("横向位置 x / cm（y=0 切面）")
    figure.colorbar(art, ax=axes, label="场量 / 对应限值（E、H 取较大比值）")
    figure.suptitle(
        "公众场量筛查：青线为比值 1；白线表示手机接收面位置", fontsize=12,
    )
    return figure


def _receiver_field_plot(plt, log_norm, rectangle, study, fields):
    config = study.config
    figure, axes_grid = plt.subplots(
        1, len(fields), figsize=(10, 4.5), layout="constrained",
        sharey=True, squeeze=False,
    )
    axes = axes_grid[0]
    rectangles = (
        ("phone_width_m", "phone_height_m", "cyan", 1.5, "-", "固定"),
        ("phone_envelope_width_m", "phone_envelope_height_m",
         "white", 1, "--", "乐观包络"),
    )
    for axis, field in zip(axes, fields):
        data = field.arrays
        art = axis.pcolormesh(
            data["x"] * 100, data["x"] * 100,
            np.maximum(data["target_sz"] * field.radiated_w, 1e-6),
            norm=log_norm(1, 15000), cmap="inferno", shading="nearest",
            rasterized=True,
        )
        for spec in rectangles:
            width_key, height_key, color, line_width, line_style, name = spec
            width, height = config[width_key] * 100, config[height_key] * 100
            label = (f"固定 {width * height:g} cm² 接收面" if name == "固定"
                     else f"{width * height:g} cm² 乐观包络")
            axis.add_patch(rectangle(
                (-width / 2, -height / 2), width, height, fill=False,
                color=color, lw=line_width, ls=line_style, label=label,
            ))
        axis.set(
            xlim=(-20, 20), ylim=(-20, 20), aspect="equal", xlabel="x / cm",
            title=(f"口径 {field.side_m * 100:g} cm，手机平面 "
                   f"z={config['reference_distance_m']:g} m"),
        )
        axis.legend(fontsize=11, loc="upper right")
    axes[0].set_ylabel("y / cm")
    figure.colorbar(art, ax=axes, label="朝手机方向的功率流密度 / W/m²")
    figure.suptitle(
        f"达到 {config['target_dc_w']:g} W 时的焦平面分布；积分使用原始有符号通量",
        fontsize=12,
    )
    return figure


def main(
    data_dir: Path = RESULTS_DIR,
    assets_dir: Path = ASSETS_DIR,
):
    """Write four PNG figures using the existing LaTeX publication style.

    This function reads saved results and does not write analysis JSON files.
    """
    study = load_study(data_dir)
    fields = load_fields(study)
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(assets_dir)
    directory.mkdir(exist_ok=True, parents=True)
    style = {
        "font.family": _font_family(font_manager), "axes.unicode_minus": False,
        "font.size": 14, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    }
    paths = []
    with plt.rc_context(style):
        plots = (
            ("01_发射口径与功率", lambda: _power_plot(plt, LogNorm, study)),
            ("02_口径与效率", lambda: _efficiency_plot(plt, study)),
            ("03_传播路径场强",
             lambda: _path_field_plot(plt, LogNorm, study, fields)),
            ("04_手机接收面", lambda: _receiver_field_plot(
                plt, LogNorm, Rectangle, study, fields)),
        )
        for name, build in plots:
            paths.append(_finish(plt, build(), directory, name))
    return paths

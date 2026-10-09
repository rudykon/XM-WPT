"""Compile the existing LaTeX source and package only the maintained project."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from .paths import (
    CACHE_DIR, DOCUMENTS_DIR, LATEX_REPORT,
    PDF_REPORT, PROJECT_DIR, REPORT_ARCHIVE, RESULTS_DIR, SIMULATION_DIR,
)


def compile_pdf() -> Path:
    """Compile twice in an ASCII-named directory, then copy a verified PDF."""
    compiler = shutil.which("xelatex")
    if compiler is None:
        raise RuntimeError("未找到 XeLaTeX。请先安装含 ctex 和 Fandol 字体的 TeX Live。")
    if not LATEX_REPORT.is_file():
        raise FileNotFoundError("未找到 LaTeX 报告，请先运行 python run.py report。")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="latex-build-", dir=DOCUMENTS_DIR) as temporary:
        build_dir = Path(temporary)
        command = [
            compiler, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
            "-jobname=focused_report", f"-output-directory={build_dir.name}", LATEX_REPORT.name,
        ]
        for _ in range(2):
            result = subprocess.run(
                command, cwd=DOCUMENTS_DIR, capture_output=True,
                text=True, encoding="utf-8", errors="replace", check=False,
            )
            (CACHE_DIR / "latex_compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"LaTeX 编译失败，详见 {CACHE_DIR / 'latex_compile.log'}")
        output = build_dir / "focused_report.pdf"
        if not output.is_file():
            raise RuntimeError("编译器未生成 PDF，请检查编译日志。")
        try:
            shutil.copyfile(output, PDF_REPORT)
        except PermissionError as error:
            preserved = CACHE_DIR / "focused_report.pdf"
            shutil.copyfile(output, preserved)
            raise RuntimeError(f"PDF 正被其他程序占用，请关闭后重试；新文件已保存在 {preserved}") from error
    print(PDF_REPORT)
    return PDF_REPORT


def package() -> Path:
    """Validate current report links and produce an allow-listed ZIP archive."""
    body = LATEX_REPORT.read_text(encoding="utf-8")
    images = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    if body.count(r"\section{") != 6:
        raise ValueError("LaTeX 报告应包含六章。")
    if len(images) != 4 or any(not (DOCUMENTS_DIR / name).is_file() for name in images):
        raise ValueError("报告应包含四幅可读取的图。")
    if body.count(r"\caption[") != 4:
        raise ValueError("四幅图应各有图注。")
    if body.count(r"\[") != body.count(r"\]"):
        raise ValueError("LaTeX 报告中的公式标记未配对。")
    from .analysis import load_study

    study = load_study()
    best = study.optimized["best_sampled"]
    if abs(best["rf_feed_for_5w_w"] * best["rf_feed_to_load_efficiency"] - study.config["target_dc_w"]) > 1e-10:
        raise ValueError("保存结果的目标负载功率不一致。")
    figure_paths = [DOCUMENTS_DIR / name for name in images]
    if PDF_REPORT.stat().st_mtime < max(path.stat().st_mtime for path in [LATEX_REPORT, *figure_paths]):
        raise ValueError("PDF 早于 LaTeX 或图片，请运行 python run.py pdf 后再打包。")
    inputs = [RESULTS_DIR / name for name in [
        "summary.json", "optimized_summary.json", "public_point_screening.json", "receiver_sensitivity.json",
    ]]
    if LATEX_REPORT.stat().st_mtime < max(path.stat().st_mtime for path in inputs):
        raise ValueError("报告早于数值结果，请先运行 python run.py report 和 pdf。")

    files = [LATEX_REPORT, PDF_REPORT, DOCUMENTS_DIR / "LaTeX编译说明.txt"]
    files += [SIMULATION_DIR / name for name in ["run.py", "README.md", "requirements.txt", ".gitignore"]]
    files += figure_paths
    for directory in [SIMULATION_DIR / "wpt", SIMULATION_DIR / "tests", SIMULATION_DIR / "docs", RESULTS_DIR]:
        files.extend(
            path for path in sorted(directory.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("打包缺少文件：" + ", ".join(missing))
    with ZipFile(REPORT_ARCHIVE, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT_DIR).as_posix())
    with ZipFile(REPORT_ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("压缩包完整性校验失败。")
    print(f"已打包 {len(files)} 个文件：{REPORT_ARCHIVE}")
    return REPORT_ARCHIVE

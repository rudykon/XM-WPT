"""One place for project paths; resolving paths has no filesystem side effects."""

from pathlib import Path

SIMULATION_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SIMULATION_DIR
RESULTS_DIR = SIMULATION_DIR / "results" / "focused_aperture"
CACHE_DIR = SIMULATION_DIR / ".cache"
DOCUMENTS_DIR = PROJECT_DIR / "文档"
LATEX_REPORT = DOCUMENTS_DIR / "小米隔空充电_技术方案.tex"
PDF_REPORT = DOCUMENTS_DIR / "小米隔空充电_技术方案_最新版.pdf"
ASSETS_DIR = LATEX_REPORT.with_suffix(".assets")
REPORT_ARCHIVE = LATEX_REPORT.with_name("小米隔空充电_仿真与LaTeX完整包.zip")

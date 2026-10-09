"""Explicit commands for calculation, inspection, and document generation."""

import argparse
import os
import sys

from .paths import CACHE_DIR, SIMULATION_DIR


def _plot() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
    from .plots import main

    main()


def _report() -> None:
    from .report import main

    main()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="固定手机接收面积的定向聚焦仿真")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("simulate", help="运行几何调相与连续口径基准扫描")
    commands.add_parser("optimize", help="运行固定通道数的相位局部优化")
    commands.add_parser("analyze", help="从保存的场数据计算曝露与定位误差")
    commands.add_parser("plot", help="只重绘 LaTeX 图片，不改动仿真数据")
    commands.add_parser("report", help="从保存的结果生成 LaTeX 报告")
    commands.add_parser("pdf", help="用本机 XeLaTeX 编译最新 LaTeX 报告")
    commands.add_parser("package", help="检查并打包代码、结果和报告")
    commands.add_parser("test", help="运行功率守恒、梯度与保存结果的回归检查")
    full = commands.add_parser("all", help="依次运行数值计算、后处理、绘图和报告")
    full.add_argument("--pdf", action="store_true", help="最后编译 PDF（需要 XeLaTeX）")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "simulate":
            from .simulation import run_baseline

            run_baseline()
        elif args.command == "optimize":
            from .simulation import run_optimized

            run_optimized()
        elif args.command == "analyze":
            from .analysis import run

            run()
        elif args.command == "plot":
            _plot()
        elif args.command == "report":
            _report()
        elif args.command == "pdf":
            from .publishing import compile_pdf

            compile_pdf()
        elif args.command == "package":
            from .publishing import package

            package()
        elif args.command == "test":
            import unittest

            suite = unittest.defaultTestLoader.discover(str(SIMULATION_DIR / "tests"))
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        elif args.command == "all":
            from .analysis import run
            from .simulation import run_baseline, run_optimized

            run_baseline()
            run_optimized()
            run()
            _plot()
            _report()
            if args.pdf:
                from .publishing import compile_pdf

                compile_pdf()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"无法完成 {args.command}：{error}", file=sys.stderr)
        return 1
    return 0

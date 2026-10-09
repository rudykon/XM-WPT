"""Single entry point; optional local dependencies are loaded only here."""

from pathlib import Path
import sys


def main() -> int:
    # A virtual environment uses its own installed requirements. The existing
    # bundled dependencies are a fallback for this computer's base interpreter.
    local_dependencies = Path(__file__).resolve().parent / ".deps"
    if sys.prefix == sys.base_prefix and local_dependencies.is_dir():
        sys.path.insert(0, str(local_dependencies))
    sys.dont_write_bytecode = True

    from wpt.cli import main as run_command

    return run_command()


if __name__ == "__main__":
    raise SystemExit(main())

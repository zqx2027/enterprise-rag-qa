from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "compileall",
            "-q",
            "app",
            "scripts",
            "streamlit_app.py",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    questions_path = PROJECT_ROOT / "data" / "eval" / "questions.json"
    json.loads(questions_path.read_text(encoding="utf-8"))
    print("Python compilation and evaluation data checks passed.")


if __name__ == "__main__":
    main()

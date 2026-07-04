import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
PARENT_DIR = ROOT.parent


def build_streamlit_command(port: int = 8501) -> List[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "frontend" / "main.py"),
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the enterprise retail support app")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    command = build_streamlit_command(port=args.port)
    print("Starting enterprise retail support app...")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PARENT_DIR), check=True)


if __name__ == "__main__":
    main()

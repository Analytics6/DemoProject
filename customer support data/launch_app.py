"""
Launch the full Enterprise Retail Support application:
  - FastAPI backend (SQLite + LangChain + OpenAI) on port 8000
  - React frontend on port 3000

Usage:
  python launch_app.py
  python launch_app.py --backend-only
  python launch_app.py --install
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REACT_DIR = ROOT / "frontend" / "react-app"


def run_command(command: list, cwd: Path, env: dict | None = None) -> None:
    print(f"[launch] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env or os.environ.copy(), check=True)


def install_dependencies() -> None:
    print("[launch] Installing Python dependencies…")
    run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], ROOT)

    npm = shutil.which("npm")
    if not npm:
        print("[launch] npm not found — install Node.js to run the React frontend.")
        return

    print("[launch] Installing React dependencies…")
    run_command([npm, "install"], REACT_DIR)


def ensure_env_file() -> None:
    env_path = ROOT / ".env"
    example = ROOT / ".env.example"
    if not env_path.exists() and example.exists():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        print("[launch] Created .env from .env.example — add your OPENAI_API_KEY.")


def start_backend(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    load_dotenv = __import__("dotenv")
    load_dotenv.load_dotenv(ROOT / ".env")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--reload",
    ]
    print(f"[launch] Starting FastAPI backend on http://127.0.0.1:{port}")
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def start_frontend() -> subprocess.Popen | None:
    npm = shutil.which("npm")
    if not npm:
        print("[launch] Skipping React — npm not installed.")
        return None
    env = os.environ.copy()
    env.setdefault("BROWSER", "none")
    cmd = [npm, "start"]
    print("[launch] Starting React frontend on http://127.0.0.1:3000")
    return subprocess.Popen(cmd, cwd=str(REACT_DIR), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Enterprise Retail Support app")
    parser.add_argument("--install", action="store_true", help="Install Python and npm dependencies")
    parser.add_argument("--backend-only", action="store_true", help="Start only the FastAPI backend")
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    args = parser.parse_args()

    if args.install:
        install_dependencies()

    ensure_env_file()

    from frontend.app_backend import initialize_database

    initialize_database(db_path=str(ROOT / "databases" / "support_app.sqlite"))
    print("[launch] SQLite database initialized.")

    backend = start_backend(args.port)
    frontend = None if args.backend_only else start_frontend()

    print("\n" + "=" * 60)
    print("Enterprise Retail Support is running")
    print(f"  API:      http://127.0.0.1:{args.port}")
    print(f"  API docs: http://127.0.0.1:{args.port}/docs")
    if frontend:
        print("  React UI: http://127.0.0.1:3000")
    print("  Login:    admin / Admin@123!")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                raise RuntimeError("Backend process exited unexpectedly.")
            if frontend and frontend.poll() is not None:
                raise RuntimeError("Frontend process exited unexpectedly.")
    except KeyboardInterrupt:
        print("\n[launch] Shutting down…")
    finally:
        for proc in [frontend, backend]:
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()

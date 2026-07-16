#!/usr/bin/env python3
"""Run InsideNeuralNets in a dedicated fullscreen Chromium booth window."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3450
BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "microsoft-edge",
    "microsoft-edge-stable",
)


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without replacing exported values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name:
            os.environ.setdefault(name, value.strip().strip('"').strip("'"))


def _find_browser(explicit_browser: str | None = None) -> str:
    """Return a Chromium-family browser executable or raise a clear error."""
    requested = explicit_browser or os.environ.get("BOOTH_BROWSER")
    if requested:
        resolved = shutil.which(requested)
        if resolved:
            return resolved
        requested_path = Path(requested).expanduser()
        if requested_path.is_file():
            return str(requested_path.resolve())
        raise RuntimeError(f"Configured booth browser was not found: {requested}")

    for candidate in BROWSER_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "No supported Chromium browser was found. Install Chromium, Chrome, or Edge, "
        "or set BOOTH_BROWSER to its executable path."
    )


def _browser_command(browser: str, url: str, profile_dir: Path, *, windowed: bool) -> list[str]:
    """Build an isolated Chromium command for booth or rehearsal mode."""
    command = [
        browser,
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--disable-session-crashed-bubble",
        "--disable-infobars",
    ]
    command.append("--app=" + url if windowed else "--kiosk")
    if not windowed:
        command.append(url)
    return command


def _server_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::", "*"} else host
    return f"http://{browser_host}:{port}"


def _wait_until_ready(server: subprocess.Popen[bytes], health_url: str, timeout: float = 30.0) -> None:
    """Wait for FastAPI readiness while also detecting an early server exit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exit_code = server.poll()
        if exit_code is not None:
            raise RuntimeError(f"The booth server exited before it became ready (code {exit_code}).")
        try:
            with urlopen(health_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"The booth server did not become ready within {timeout:.0f} seconds.")


def _stop_process(process: subprocess.Popen[bytes] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {name}…")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def main() -> int:
    _load_env_file(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run InsideNeuralNets in a fullscreen booth window.")
    parser.add_argument("--host", default=os.environ.get("FRONTEND_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FRONTEND_PORT", DEFAULT_PORT)))
    parser.add_argument("--browser", help="Chromium, Chrome, or Edge executable name or path.")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Open an app-style window instead of fullscreen kiosk mode for rehearsal.",
    )
    args = parser.parse_args()

    server_url = _server_url(args.host, args.port)
    browser = _find_browser(args.browser)
    profile_dir = PROJECT_ROOT / ".booth-browser-profile"
    profile_dir.mkdir(exist_ok=True)

    stop_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "stop_dev.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--project-root",
        str(PROJECT_ROOT),
    ]
    subprocess.run(stop_command, cwd=PROJECT_ROOT, check=True)

    server_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        "warning",
        "--no-access-log",
    ]
    server: subprocess.Popen[bytes] | None = None
    browser_process: subprocess.Popen[bytes] | None = None
    try:
        print(f"Starting booth server at {server_url}")
        server = subprocess.Popen(server_command, cwd=PROJECT_ROOT)
        _wait_until_ready(server, server_url + "/api/health")
        print("Booth server is ready. Opening the dedicated browser window.")
        browser_process = subprocess.Popen(
            _browser_command(browser, server_url, profile_dir, windowed=args.windowed),
            cwd=PROJECT_ROOT,
        )
        return browser_process.wait()
    except KeyboardInterrupt:
        print("Booth mode interrupted.")
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Could not start booth mode: {exc}", file=sys.stderr)
        return 1
    finally:
        _stop_process(browser_process, "booth browser")
        _stop_process(server, "booth server")


if __name__ == "__main__":
    raise SystemExit(main())

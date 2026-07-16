from pathlib import Path

import pytest

from scripts import run_booth


def test_run_dev_quiets_uvicorn_info_and_access_logs() -> None:
    script = Path("scripts/run_dev.ps1").read_text(encoding="utf-8")

    assert "--log-level warning" in script
    assert "--no-access-log" in script


def test_run_dev_uses_project_python_module_entrypoint() -> None:
    script = Path("scripts/run_dev.ps1").read_text(encoding="utf-8")

    assert "& $PythonBin -m uvicorn app:app" in script
    assert ".venv/bin/uvicorn" not in script


def test_booth_browser_command_uses_isolated_fullscreen_profile(tmp_path: Path) -> None:
    command = run_booth._browser_command(
        "/usr/bin/chromium",
        "http://127.0.0.1:3450",
        tmp_path,
        windowed=False,
    )

    assert command[0] == "/usr/bin/chromium"
    assert f"--user-data-dir={tmp_path}" in command
    assert "--kiosk" in command
    assert command[-1] == "http://127.0.0.1:3450"


def test_booth_windowed_command_uses_app_window(tmp_path: Path) -> None:
    command = run_booth._browser_command(
        "chromium",
        "http://127.0.0.1:3450",
        tmp_path,
        windowed=True,
    )

    assert "--kiosk" not in command
    assert "--app=http://127.0.0.1:3450" in command


def test_booth_browser_lookup_has_clear_missing_browser_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOOTH_BROWSER", raising=False)
    monkeypatch.setattr(run_booth.shutil, "which", lambda _candidate: None)

    with pytest.raises(RuntimeError, match="No supported Chromium browser"):
        run_booth._find_browser()

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


def test_powershell_launchers_support_windows_and_unix_virtual_environments() -> None:
    for script_name in ("run_dev.ps1", "run_booth.ps1", "stop_dev.ps1"):
        script = Path("scripts", script_name).read_text(encoding="utf-8")

        assert ".venv/Scripts/python.exe" in script
        assert ".venv/bin/python" in script


def test_powershell_setup_uses_the_platform_virtual_environment_python() -> None:
    script = Path("scripts/setup.ps1").read_text(encoding="utf-8")

    assert "Scripts/python.exe" in script
    assert "bin/python" in script
    assert "Get-Command python3, python, py" in script


def test_run_booth_powershell_supports_fullscreen_and_windowed_modes() -> None:
    script = Path("scripts/run_booth.ps1").read_text(encoding="utf-8")

    assert 'scripts/run_booth.py' in script
    assert "[switch]$Windowed" in script
    assert '$BoothArguments += "--windowed"' in script
    assert "& $PythonBin @BoothArguments" in script


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

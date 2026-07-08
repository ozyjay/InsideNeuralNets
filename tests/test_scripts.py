from pathlib import Path


def test_run_dev_quiets_uvicorn_info_and_access_logs() -> None:
    script = Path("scripts/run_dev.ps1").read_text(encoding="utf-8")

    assert "--log-level warning" in script
    assert "--no-access-log" in script

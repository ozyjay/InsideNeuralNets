#!/usr/bin/env python3
"""Stop previous local Uvicorn demo processes for this project.

This is intentionally scoped: it only terminates processes that are listening on
this demo port and look like this project's `uvicorn app:app` server.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import time
from pathlib import Path

LISTEN_STATE = "0A"


def _normalise_host(host: str) -> set[str]:
    if host in {"0.0.0.0", "::", "*"}:
        return {"0.0.0.0", "127.0.0.1", "::", "::1"}
    return {host}


def _hex_to_ipv4(value: str) -> str:
    raw = bytes.fromhex(value)
    return socket.inet_ntop(socket.AF_INET, raw[::-1])


def _hex_to_ipv6(value: str) -> str:
    raw = bytes.fromhex(value)
    # /proc/net/tcp6 stores groups little-endian.
    reordered = b"".join(raw[i : i + 4][::-1] for i in range(0, 16, 4))
    return socket.inet_ntop(socket.AF_INET6, reordered)


def _listening_socket_inodes(port: int, host: str) -> set[str]:
    wanted_hosts = _normalise_host(host)
    inodes: set[str] = set()
    for proc_file, parser in ((Path("/proc/net/tcp"), _hex_to_ipv4), (Path("/proc/net/tcp6"), _hex_to_ipv6)):
        if not proc_file.exists():
            continue
        for line in proc_file.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_address, state, inode = parts[1], parts[3], parts[9]
            address_hex, port_hex = local_address.split(":")
            if state != LISTEN_STATE or int(port_hex, 16) != port:
                continue
            try:
                address = parser(address_hex)
            except OSError:
                continue
            if address in wanted_hosts or host in {"0.0.0.0", "::", "*"}:
                inodes.add(inode)
    return inodes


def _pids_for_inodes(inodes: set[str]) -> set[int]:
    pids: set[int] = set()
    if not inodes:
        return pids
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        fd_dir = proc_dir / "fd"
        if not fd_dir.exists():
            continue
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.add(int(proc_dir.name))
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return pids


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _cwd(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except OSError:
        return None


def _is_this_demo_server(pid: int, project_root: Path) -> bool:
    cmdline = _cmdline(pid)
    cwd = _cwd(pid)
    if "uvicorn" not in cmdline or "app:app" not in cmdline:
        return False
    if cwd == project_root:
        return True
    return str(project_root) in cmdline


def stop_previous_server(host: str, port: int, project_root: Path, *, dry_run: bool = False) -> list[int]:
    inodes = _listening_socket_inodes(port, host)
    candidate_pids = _pids_for_inodes(inodes)
    pids = sorted(pid for pid in candidate_pids if _is_this_demo_server(pid, project_root))

    for pid in pids:
        print(f"Stopping previous demo server process {pid} on {host}:{port}")
        if not dry_run:
            os.kill(pid, signal.SIGTERM)

    if not dry_run:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if all(not Path(f"/proc/{pid}").exists() for pid in pids):
                break
            time.sleep(0.1)
        for pid in pids:
            if Path(f"/proc/{pid}").exists():
                print(f"Force-stopping unresponsive demo server process {pid}")
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    return pids


def main() -> int:
    parser = argparse.ArgumentParser(description="Stop previous local AlexNet demo Uvicorn process.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3450)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    if not Path("/proc/net/tcp").exists():
        print("No /proc network table available; skipping previous-process cleanup.")
        return 0

    pids = stop_previous_server(args.host, args.port, project_root, dry_run=args.dry_run)
    if not pids:
        print(f"No previous demo server found on {args.host}:{args.port}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

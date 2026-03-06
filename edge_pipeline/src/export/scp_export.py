# =============================================================================
# scp_export.py — Push trajectory files to a remote host via SCP
# =============================================================================
# Purpose:  Transfer generated trajectory files (CSV/JSON) from the Pi to a
#           desktop/server for MATLAB ingestion using paramiko/SCP.
#
# Usage:
#   python -m src.export.scp_export --file trajectory.csv --host 192.168.1.100
# =============================================================================
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def scp_push(
    local_path: str | Path,
    remote_host: str,
    remote_path: str = "~/trajectories/",
    username: str = "pi",
    password: str | None = None,
    key_file: str | None = None,
    port: int = 22,
) -> None:
    """Push a local file to a remote host via SCP.

    Parameters
    ----------
    local_path : str or Path
        Local file to transfer.
    remote_host : str
        Remote hostname or IP.
    remote_path : str
        Destination path on remote host.
    username : str
        SSH username.
    password : str | None
        SSH password (if not using key-based auth).
    key_file : str | None
        Path to SSH private key file.
    port : int
        SSH port (default 22).
    """
    try:
        import paramiko
    except ImportError:
        logger.error("paramiko not installed. Install with: pip install paramiko")
        raise

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Connect
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": remote_host,
        "port": port,
        "username": username,
    }
    if key_file:
        connect_kwargs["key_filename"] = key_file
    elif password:
        connect_kwargs["password"] = password

    try:
        ssh.connect(**connect_kwargs)
        sftp = ssh.open_sftp()

        remote_file = f"{remote_path.rstrip('/')}/{local_path.name}"
        logger.info("SCP: %s → %s:%s", local_path, remote_host, remote_file)

        # Ensure remote directory exists
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            sftp.mkdir(remote_path)

        sftp.put(str(local_path), remote_file)
        logger.info("SCP transfer complete")

    finally:
        ssh.close()


# ── CLI entrypoint ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Push trajectory to remote host via SCP")
    parser.add_argument("--file", required=True, help="Local trajectory file")
    parser.add_argument("--host", required=True, help="Remote hostname or IP")
    parser.add_argument("--remote-path", default="~/trajectories/")
    parser.add_argument("--user", default="pi")
    parser.add_argument("--password", default=None)
    parser.add_argument("--key", default=None, help="SSH private key file")
    parser.add_argument("--port", type=int, default=22)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    scp_push(
        local_path=args.file,
        remote_host=args.host,
        remote_path=args.remote_path,
        username=args.user,
        password=args.password,
        key_file=args.key,
        port=args.port,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
capture_environment.py — Record a full environment snapshot for reproducibility.

Usage:
    python3 validation/capture_environment.py validation/environment.json

Can also be imported:
    from validation.capture_environment import capture_environment
    env = capture_environment()
"""

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], *, timeout: int = 10) -> str | None:
    """Run a subprocess and return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _run_lines(cmd: list[str], *, timeout: int = 30) -> list[str] | None:
    """Run a subprocess and return stdout as a list of lines, or None."""
    raw = _run(cmd, timeout=timeout)
    if raw is None:
        return None
    return [line for line in raw.splitlines() if line]


def _first_line(cmd: list[str]) -> str | None:
    """Return the first non-empty line of a command's stdout, or None."""
    raw = _run(cmd)
    if raw is None:
        return None
    for line in raw.splitlines():
        if line.strip():
            return line.strip()
    return None


# ---------------------------------------------------------------------------
# Individual capture functions
# ---------------------------------------------------------------------------

def _capture_os() -> dict:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "platform_string": platform.platform(),
        "libc": list(platform.libc_ver()),
    }


def _capture_python() -> dict:
    return {
        "version": sys.version,
        "version_info": list(sys.version_info[:5]),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "implementation": platform.python_implementation(),
    }


def _capture_pip_freeze() -> list[str] | None:
    return _run_lines([sys.executable, "-m", "pip", "freeze"], timeout=60)


def _capture_objdump() -> dict:
    line = _first_line(["objdump", "--version"])
    return {
        "available": line is not None,
        "version_string": line,
    }


def _capture_tool_version(tool: str) -> dict:
    line = _first_line([tool, "--version"])
    return {
        "available": line is not None,
        "version_string": line,
    }


def _capture_cpu() -> dict:
    info: dict = {
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }

    # Linux: pull model name and cache from /proc/cpuinfo
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text(errors="replace")
            for line in text.splitlines():
                if "model name" in line and "model_name" not in info:
                    info["model_name"] = line.split(":", 1)[1].strip()
                if "cache size" in line and "cache_size" not in info:
                    info["cache_size"] = line.split(":", 1)[1].strip()
        except OSError:
            pass

    # Physical core count via lscpu (not always available)
    lscpu = _run(["lscpu"])
    if lscpu:
        for line in lscpu.splitlines():
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "Core(s) per socket":
                info["cores_per_socket"] = val
            elif key == "Socket(s)":
                info["sockets"] = val
            elif key == "CPU max MHz":
                info["cpu_max_mhz"] = val

    return info


def _capture_git(project_root: Path) -> dict:
    """Capture git state of the project repository."""

    def git(*args: str) -> str | None:
        return _run(["git", "-C", str(project_root), *args])

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    tag = git("describe", "--tags", "--exact-match", "HEAD")
    dirty = git("status", "--porcelain")

    return {
        "commit": commit,
        "branch": branch,
        "tag": tag,
        "dirty": bool(dirty),  # True if there are uncommitted changes
        "dirty_files": dirty.splitlines() if dirty else [],
    }


def _capture_random_seeds() -> dict:
    """
    Record seeding configuration relevant to the pipeline.

    The pipeline uses numpy/random; we record whether PYTHONHASHSEED is
    pinned and note that no explicit seed is set by default in binary_dna.py
    (results are therefore deterministic only via fixed corpus + fixed code).
    """
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not_set"),
        "numpy_random_seed": "not_set_by_pipeline",
        "python_random_seed": "not_set_by_pipeline",
        "note": (
            "The pipeline does not set an explicit RNG seed. "
            "Set PYTHONHASHSEED=0 for fully reproducible dict ordering on Python 3.7+."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def capture_environment(project_root: Path | None = None) -> dict:
    """
    Collect a full environment snapshot and return it as a plain dict.

    Parameters
    ----------
    project_root:
        Root of the git repository.  Defaults to the parent of this file's
        directory (i.e., /home/aaslyan/OpCode-Stats when called from
        validation/capture_environment.py).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    env: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": _capture_os(),
        "python": _capture_python(),
        "pip_freeze": _capture_pip_freeze(),
        "tools": {
            "objdump": _capture_objdump(),
            "gcc": _capture_tool_version("gcc"),
            "clang": _capture_tool_version("clang"),
            "make": _capture_tool_version("make"),
            "strip": _capture_tool_version("strip"),
        },
        "cpu": _capture_cpu(),
        "git": _capture_git(project_root),
        "random_seeds": _capture_random_seeds(),
    }

    return env


def save_environment(output_path: Path, project_root: Path | None = None) -> None:
    """Capture environment and write it as indented JSON to *output_path*."""
    env = capture_environment(project_root=project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(env, fh, indent=2, default=str)
    print(f"[capture_environment] Written to {output_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python3 validation/capture_environment.py <output.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    out = Path(sys.argv[1])
    project_root_arg = Path(sys.argv[2]) if len(sys.argv) >= 3 else None
    save_environment(out, project_root=project_root_arg)

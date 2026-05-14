from __future__ import annotations

"""
Resolve paths to external dependencies (gaussian-splatting, MASt3R repos).

Priority order:
  1. Environment variables: ROOMSPLAT_GS_DIR, ROOMSPLAT_MAST3R_DIR
  2. Config file: ~/.roomsplat/config.json
  3. Built-in defaults (common install locations)

Users set these once; the CLI reads them transparently.
"""

import json
import os
from pathlib import Path

_CONFIG_FILE = Path.home() / ".roomsplat" / "config.json"

_DEFAULTS = {
    "gs_dir": Path.home() / "gaussian-splatting",
    "mast3r_dir": Path.home() / "mast3r",
}


def _load_file() -> dict:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text())
        except Exception:
            return {}
    return {}


def get_gs_dir() -> Path:
    if v := os.environ.get("ROOMSPLAT_GS_DIR"):
        return Path(v)
    if v := _load_file().get("gs_dir"):
        return Path(v)
    return _DEFAULTS["gs_dir"]


def get_mast3r_dir() -> Path:
    if v := os.environ.get("ROOMSPLAT_MAST3R_DIR"):
        return Path(v)
    if v := _load_file().get("mast3r_dir"):
        return Path(v)
    return _DEFAULTS["mast3r_dir"]


def save(gs_dir: Path | None = None, mast3r_dir: Path | None = None) -> None:
    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_file()
    if gs_dir is not None:
        data["gs_dir"] = str(gs_dir)
    if mast3r_dir is not None:
        data["mast3r_dir"] = str(mast3r_dir)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2))


def validate() -> list[str]:
    """Return list of missing/broken dependency messages."""
    issues = []
    gs = get_gs_dir()
    if not (gs / "train.py").exists():
        issues.append(
            f"gaussian-splatting not found at {gs}\n"
            "  Clone it: git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting\n"
            "  Then set: roomsplat config --gs-dir /path/to/gaussian-splatting"
        )
    mast3r = get_mast3r_dir()
    if not (mast3r / "mast3r" / "model.py").exists():
        issues.append(
            f"MASt3R not found at {mast3r} (optional, needed for --pose-method mast3r)\n"
            "  Clone it: git clone --recursive https://github.com/naver/mast3r\n"
            "  Then set: roomsplat config --mast3r-dir /path/to/mast3r"
        )
    return issues

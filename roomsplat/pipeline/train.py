from __future__ import annotations

"""
3DGS training: COLMAP sparse + images → point_cloud.ply

Wraps gaussian-splatting/train.py as a subprocess so that the rasterizer
(diff_gaussian_rasterization) stays in its own Python process — critical when
multiple environments have conflicting rasterizer installs (lesson from the
Neural Rendering benchmarking work: depth-diff and standard rasterizers clash).

The source directory passed to train.py must have:
  <source>/
  ├── images/     ← the extracted JPEG frames
  └── sparse/0/   ← cameras.bin, images.bin, points3D.bin

This is exactly the layout produced by the extract + poses steps.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

console = Console()


def train(
    source_path: Path,
    model_path: Path,
    gs_dir: Path,
    iterations: int = 30_000,
    sh_degree: int = 3,
    extra_args: list[str] | None = None,
    train_python: str | None = None,
) -> Path:
    """
    Run gaussian-splatting/train.py and return the model output directory.

    Args:
        source_path:  Directory with images/ and sparse/0/ subdirs
        model_path:   Where to write the trained model
        gs_dir:       Root of the cloned gaussian-splatting repo
        iterations:   Training iterations (30k ≈ 35 min; 7k ≈ 5 min preview)
        sh_degree:    Spherical harmonics degree 0-3 (higher = better view-dep colour)
        extra_args:   Extra flags forwarded to train.py (e.g. --densify_until_iter)

    Returns:
        model_path (for chaining)
    """
    train_script = gs_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"gaussian-splatting/train.py not found at {train_script}.\n"
            "Set ROOMSPLAT_GS_DIR or run: roomsplat config --gs-dir /path/to/gaussian-splatting"
        )

    model_path.mkdir(parents=True, exist_ok=True)

    python_exe = train_python or _find_python_with_rasterizer() or sys.executable

    cmd = [
        python_exe, str(train_script),
        "--source_path",    str(source_path.resolve()),
        "--model_path",     str(model_path.resolve()),
        "--iterations",     str(iterations),
        "--sh_degree",      str(sh_degree),
        "--save_iterations", str(iterations),
        "--checkpoint_iterations", str(iterations // 2), str(iterations),
        "--disable_viewer",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = _build_env(gs_dir)

    console.print(
        f"  [dim]Training 3DGS: {iterations:,} iterations "
        f"(~{iterations / 30_000 * 35:.0f} min on RTX 3080)[/]"
    )

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(gs_dir))
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"3DGS training failed (exit code {result.returncode}).\n"
            "Check the output above for CUDA / memory errors."
        )

    console.print(f"  [green]Training complete in {elapsed / 60:.1f} min[/]")

    config = {
        "source_path": str(source_path),
        "iterations": iterations,
        "sh_degree": sh_degree,
        "train_time_min": round(elapsed / 60, 1),
    }
    (model_path / "roomsplat_train_config.json").write_text(json.dumps(config, indent=2))

    return model_path


def find_ply(model_path: Path, iteration: int) -> Path:
    """Return path to the trained point_cloud.ply for a given iteration."""
    return model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"


def count_gaussians(ply_path: Path) -> int:
    try:
        from plyfile import PlyData
        data = PlyData.read(str(ply_path))
        return len(data["vertex"])
    except Exception:
        return -1


# ── Environment helpers ─────────────────────────────────────────────────────────

def _find_python_with_rasterizer() -> str | None:
    """
    Try common Python locations that are likely to have diff_gaussian_rasterization
    installed (conda envs active during Neural Rendering work).
    Returns the first one that works, or None.
    """
    import shutil
    candidates = [
        # Active conda env (if different from current)
        shutil.which("python"),
        # Common conda env names used in the Neural Rendering project
        str(Path.home() / "miniconda3/envs/gaussian_splatting/bin/python"),
        str(Path.home() / "miniconda3/envs/3dgs/bin/python"),
        str(Path.home() / "anaconda3/envs/gaussian_splatting/bin/python"),
    ]
    check = (
        "import sys; "
        "from diff_gaussian_rasterization import GaussianRasterizer; "
        "print(sys.executable)"
    )
    for py in candidates:
        if not py or not Path(py).exists():
            continue
        try:
            result = subprocess.run(
                [py, "-c", check],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                found = result.stdout.strip()
                console.print(f"  [dim]Auto-detected rasterizer Python: {found}[/]")
                return found
        except Exception:
            continue
    return None


def _build_env(gs_dir: Path) -> dict:
    """Build subprocess environment with CUDA_HOME and PYTHONPATH set."""
    env = dict(os.environ)

    # Try common CUDA home locations if not set
    if "CUDA_HOME" not in env:
        for candidate in [
            Path.home() / "cuda-home",
            Path("/usr/local/cuda"),
            Path("/usr/local/cuda-12"),
            Path("/usr/local/cuda-11"),
        ]:
            if (candidate / "bin" / "nvcc").exists():
                env["CUDA_HOME"] = str(candidate)
                break

    cuda_home = env.get("CUDA_HOME", "")
    if cuda_home:
        env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")

    # Use GCC-12 if available (required for some CUDA rasterizer builds)
    if Path("/usr/bin/gcc-12").exists():
        env.setdefault("CC", "/usr/bin/gcc-12")
        env.setdefault("CXX", "/usr/bin/g++-12")

    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{gs_dir}:{pythonpath}" if pythonpath else str(gs_dir)
    # Mitigate CUDA memory fragmentation during long densification runs.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    return env

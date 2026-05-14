"""
Convert 3DGS point_cloud.ply → .splat binary for WebGL viewing.

The .splat format (antimatter15/splat, MIT licence) stores each Gaussian as
32 bytes in the following layout:
    bytes  0-11  position xyz          3 × float32 (12 bytes)
    bytes 12-23  scale xyz             3 × float32 (12 bytes)
    bytes 24-27  colour RGBA           4 × uint8   ( 4 bytes)
    bytes 28-31  rotation quaternion   4 × uint8   ( 4 bytes)

The .ply from gaussian-splatting contains (per vertex):
  x, y, z                    — world-space position (float)
  f_dc_0/1/2                 — DC spherical harmonic coefficients (base colour)
  f_rest_0 … f_rest_N        — higher-order SH (view-dependent, ignored for base colour)
  opacity                    — logit-transformed opacity (float)
  scale_0/1/2                — log-scale per axis (float)
  rot_0/1/2/3                — unnormalised quaternion (float)

Conversions:
  colour  = sigmoid(f_dc * C0 + 0.5) clamped to [0,255]   (C0 = 0.2820948)
  alpha   = sigmoid(opacity) * 255
  scale   = exp(scale_i)
  rot     = normalise(rot) → pack each component as uint8: round((r+1)/2 * 255)

Gaussians are sorted by opacity (most opaque first) so the viewer can cull
transparent ones at the tail — reduces overdraw significantly.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

# SH DC coefficient for degree-0 term (converts SH to linear colour)
_SH_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _prune_gaussians(
    xyz: np.ndarray,
    scales: np.ndarray,
    alpha: np.ndarray,
    *,
    alpha_min: float = 0.05,
    nn_mult: float = 10.0,
) -> np.ndarray:
    """Return boolean keep-mask. Removes floaters and low-opacity noise.

    NOTE: scale ratio is NOT filtered here — disk/pancake Gaussians representing
    flat surfaces legitimately need ratios of 20:1 or higher.  The scale ratio
    cap (30:1 → fixed 10:1 in the export path) is applied AFTER pruning.
    """
    from scipy.spatial import cKDTree

    N = len(xyz)
    keep = np.ones(N, dtype=bool)

    # 1. Opacity threshold — nearly-transparent Gaussians contribute noise
    keep &= alpha >= alpha_min
    console.print(f"  [dim]  after opacity filter (>={alpha_min}): {keep.sum():,} ({100*keep.mean():.0f}%)[/]")

    # 2. Spatial isolation — true floaters have no nearby neighbours
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=2)  # k=2: [self=0, nearest neighbour]
    nn = dists[:, 1]
    threshold = np.median(nn[keep]) * nn_mult
    keep &= nn <= threshold
    console.print(f"  [dim]  after spatial pruning (NN ≤ {nn_mult:.0f}× median): {keep.sum():,} ({100*keep.mean():.0f}%)[/]")

    console.print(
        f"  [dim]Pruned {(~keep).sum():,} / {N:,} Gaussians "
        f"({100*(~keep).mean():.1f}% removed) → {keep.sum():,} remain[/]"
    )
    return keep


def ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    """
    Convert a gaussian-splatting point_cloud.ply to the .splat binary format.

    Args:
        ply_path:    Path to point_cloud.ply from gaussian-splatting training
        splat_path:  Output path for the .splat file

    Returns:
        Number of Gaussians written.
    """
    from plyfile import PlyData

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    console.print(f"  [dim]Loading PLY: {ply_path.name}[/]")
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    n = len(v)
    console.print(f"  [dim]{n:,} Gaussians (before pruning)[/]")

    # ── Position ────────────────────────────────────────────────────────────────
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)  # (N,3)

    # ── Scale: exp(log_scale), floor to avoid ratio-clamping bug ────────────
    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    ).astype(np.float32)  # (N,3)
    scales = np.maximum(scales, 1e-6)   # floor before ratio comparisons

    # ── Opacity ──────────────────────────────────────────────────────────────
    alpha = _sigmoid(np.array(v["opacity"])).astype(np.float32)   # (N,)

    # ── Optional pruning ───────────────────────────────────────────────────────
    # Keep all splats by default for best fidelity relative to CUDA rendering.
    # Pruning can speed up the web viewer, but aggressive thresholds may remove
    # thin structures and cause "messy"/hollow looking views.
    use_pruning = os.getenv("ROOMSPLAT_PRUNE", "0") == "1"
    if use_pruning:
        console.print("  [dim]Pruning floaters… (ROOMSPLAT_PRUNE=1)[/]")
        keep = _prune_gaussians(xyz, scales, alpha)
        xyz    = xyz[keep]
        scales = scales[keep]
        alpha  = alpha[keep]
    else:
        keep = slice(None)
        console.print("  [dim]Skipping pruning (set ROOMSPLAT_PRUNE=1 to enable)[/]")

    # ── Cap extreme needle Gaussians (ratio > 300:1) ─────────────────────────
    # Applied unconditionally — flat-surface Gaussians legitimately need up to
    # ~100:1; only the degenerate tail (~1–2%) is affected.
    s_min = scales.min(axis=1, keepdims=True)
    scales = np.minimum(scales, s_min * 300.0)

    # ── Colour: clamp(f_dc * C0 + 0.5, 0, 1) → [0,255] uint8 ──────────────
    # Matches gaussian-splatting renderer: clamp_min(sh2rgb + 0.5, 0).
    # Do NOT use sigmoid here — it compresses dynamic range and gives wrong colours.
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)[keep]  # (N,3)
    rgb_linear = np.clip(_SH_C0 * f_dc + 0.5, 0.0, 1.0)              # (N,3) in [0,1]
    rgb_uint8 = np.round(rgb_linear * 255).astype(np.uint8)           # (N,3)

    alpha_uint8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)       # (N,)

    rgba_uint8 = np.concatenate(
        [rgb_uint8, alpha_uint8[:, None]], axis=1
    )  # (N,4)

    # ── Rotation: normalise quaternion → pack as uint8 ──────────────────────
    # gaussian-splatting stores (w, x, y, z); .splat expects (x, y, z, w) — reorder
    rot_raw = np.stack(
        [v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], axis=1  # xyzw
    ).astype(np.float64)[keep]
    norms = np.linalg.norm(rot_raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    rot_norm = (rot_raw / norms).astype(np.float32)  # (N,4) in [-1,1]
    # Pack to uint8: (r + 1) / 2 * 255
    rot_uint8 = np.clip((rot_norm + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)  # (N,4)

    # ── Sort by opacity descending (opaque first) ───────────────────────────
    order = np.argsort(-alpha)
    xyz       = xyz[order]
    scales    = scales[order]
    rgba_uint8 = rgba_uint8[order]
    rot_uint8  = rot_uint8[order]

    # ── Pack to 32-byte records and write ────────────────────────────────────
    splat_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a structured array for zero-copy packing
    # Layout: [xyz float32×3] [scale float32×3] [rgba uint8×4] [rot uint8×4]
    n = len(xyz)  # pruned count
    buf = np.zeros(n, dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
        ("qx", "u1"), ("qy", "u1"), ("qz", "u1"), ("qw", "u1"),
    ])
    buf["x"], buf["y"], buf["z"]   = xyz[:, 0],    xyz[:, 1],    xyz[:, 2]
    buf["sx"], buf["sy"], buf["sz"] = scales[:, 0], scales[:, 1], scales[:, 2]
    buf["r"], buf["g"], buf["b"], buf["a"] = (
        rgba_uint8[:, 0], rgba_uint8[:, 1], rgba_uint8[:, 2], rgba_uint8[:, 3]
    )
    buf["qx"], buf["qy"], buf["qz"], buf["qw"] = (
        rot_uint8[:, 0], rot_uint8[:, 1], rot_uint8[:, 2], rot_uint8[:, 3]
    )

    splat_path.write_bytes(buf.tobytes())

    size_mb = splat_path.stat().st_size / 1_048_576

    # ── SH3 sidecar (.splat.sh) ──────────────────────────────────────────────
    # 48 float16 per Gaussian, same opacity-sorted order as .splat.
    # Layout: 4 groups × 4 texels × 4 values = 16 floats per channel × 3 channels
    #   Channel R (values 0–15):  [f_dc_0, f_rest_0..14]
    #   Channel G (values 16–31): [f_dc_1, f_rest_15..29]
    #   Channel B (values 32–47): [f_dc_2, f_rest_30..44]
    sh_fields = [f"f_rest_{i}" for i in range(45)]
    if all(f in v.data.dtype.names for f in sh_fields):
        sh = np.empty((n, 48), dtype=np.float32)
        sh[:, 0]    = np.array(v["f_dc_0"],    dtype=np.float32)[keep][order]
        sh[:, 1:16] = np.stack([np.array(v[f"f_rest_{i}"], dtype=np.float32)
                                 for i in range(15)], axis=1)[keep][order]
        sh[:, 16]    = np.array(v["f_dc_1"],    dtype=np.float32)[keep][order]
        sh[:, 17:32] = np.stack([np.array(v[f"f_rest_{i}"], dtype=np.float32)
                                  for i in range(15, 30)], axis=1)[keep][order]
        sh[:, 32]    = np.array(v["f_dc_2"],    dtype=np.float32)[keep][order]
        sh[:, 33:48] = np.stack([np.array(v[f"f_rest_{i}"], dtype=np.float32)
                                  for i in range(30, 45)], axis=1)[keep][order]

        sh_path = splat_path.parent / (splat_path.name + ".sh")
        sh_path.write_bytes(sh.astype(np.float16).tobytes())
        sh_mb = sh_path.stat().st_size / 1_048_576
        console.print(f"  [dim]Exported SH3 → {sh_path.name} ({sh_mb:.1f} MB)[/]")

    console.print(f"  [green]Exported {n:,} Gaussians → {splat_path.name} ({size_mb:.1f} MB)[/]")
    return n


def write_cameras_json(sparse_dir: Path, viewer_dir: Path) -> None:
    """
    Read COLMAP images.bin and write cameras.json for the web viewer.

    cameras.json format:
      { "initial": <int>, "cameras": [{"pos": [x,y,z], "forward": [fx,fy,fz], "name": "..."}, ...] }

    The viewer uses this to start at a real training-camera pose instead of a
    heuristic guess that often lands outside the reconstructed volume.
    """
    images_bin = sparse_dir / "images.bin"
    if not images_bin.exists():
        return

    cameras = _read_colmap_images(images_bin)
    if not cameras:
        return

    cameras.sort(key=lambda c: c["name"])

    (viewer_dir / "cameras.json").write_text(
        json.dumps({"initial": len(cameras) // 2, "cameras": cameras}, indent=None)
    )
    console.print(f"  [dim]Exported {len(cameras)} camera poses → cameras.json[/]")


def _read_colmap_images(path: Path) -> list:
    """Parse COLMAP images.bin and return list of {pos, forward, name} dicts."""
    from scipy.spatial.transform import Rotation

    cameras = []
    with open(path, "rb") as f:
        (n_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_images):
            (_image_id,) = struct.unpack("<I", f.read(4))
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            (_camera_id,) = struct.unpack("<I", f.read(4))

            # Read null-terminated name
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")

            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(n_pts * 24)  # skip 2D points (x, y float64 + point3D_id int64)

            # World-to-camera rotation from quaternion (scipy uses xyzw order)
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])

            pos     = (-R.T @ t).tolist()          # camera centre in world space
            forward = R[2].tolist()                # camera looks along +Z_cam in world
            # Physical up in world = -(camera Y-axis in world); R[1] = image-down direction
            up      = (-R[1]).tolist()

            cameras.append({"pos": pos, "forward": forward, "up": up, "name": name})

    return cameras

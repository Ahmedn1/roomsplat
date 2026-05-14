from __future__ import annotations

"""
Camera pose estimation: frames → COLMAP sparse/0/ directory.

Two strategies:
  1. COLMAP sequential matcher (default) — best for ordered video frames with
     texture. Matches each frame to its N nearest temporal neighbours (fast,
     O(N·overlap) instead of O(N²)).

  2. MASt3R (--pose-method mast3r) — deep-learning pose estimator that works
     even with featureless rooms. Runs pairwise on a sub-sampled set of frames
     then writes COLMAP-compatible output so the rest of the pipeline is unchanged.
     Use this when COLMAP fails (registration < 80% of frames).

Output: <output_dir>/sparse/0/{cameras,images,points3D}.bin
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from rich.console import Console

console = Console()

# ── COLMAP ─────────────────────────────────────────────────────────────────────

def run_colmap(
    frames_dir: Path,
    output_dir: Path,
    camera_model: str = "SIMPLE_PINHOLE",
    sequential_overlap: int = 10,
    min_num_inliers: int = 15,
    undistort: bool = False,
    matcher: str = "sequential",
) -> dict:
    """
    Run COLMAP sequential SfM on a directory of video frames.

    Uses the sequential matcher (match each frame to ±overlap neighbours)
    rather than exhaustive matching — correct for ordered video frames and
    O(N·overlap) instead of O(N²).

    Args:
        frames_dir:          Directory containing frame_XXXXX.jpg files
        output_dir:          Where to write sparse/0/ (workspace)
        camera_model:        SIMPLE_PINHOLE (required by gaussian-splatting).
                             Phone lens distortion is minor at typical room distances.
        sequential_overlap:  Number of adjacent frames to match (default 10)
        min_num_inliers:     Minimum matches to accept a pair

    Returns:
        dict with registration stats
    """
    try:
        import pycolmap
    except ImportError:
        raise ImportError(
            "pycolmap is required for COLMAP pose estimation.\n"
            "Install it: pip install pycolmap\n"
            "Or use MASt3R: roomsplat run ... --pose-method mast3r"
        )

    images = sorted(frames_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")

    workspace = output_dir / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    sparse_dir = workspace / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    db_path = workspace / "database.db"

    n_images = len(images)
    console.print(f"  [dim]COLMAP: {n_images} frames, sequential overlap={sequential_overlap}[/]")

    # Feature extraction — SIMPLE_RADIAL: fx, fy(=fx), cx, cy, k (one radial term)
    # Phone cameras have noticeable barrel distortion; one radial term usually enough.
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model

    extract_opts = pycolmap.FeatureExtractionOptions()
    extract_opts.sift.max_num_features = 8192
    extract_opts.use_gpu = True

    pycolmap.extract_features(
        database_path=db_path,
        image_path=frames_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,  # one shared camera model (same phone)
        reader_options=reader_opts,
        extraction_options=extract_opts,
    )

    # Sequential matching — natural for video (frames i and i±overlap share content)
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.use_gpu = True

    verify_opts = pycolmap.TwoViewGeometryOptions()
    verify_opts.min_num_inliers = min_num_inliers

    seq_opts = pycolmap.SequentialPairingOptions()
    seq_opts.overlap = sequential_overlap
    seq_opts.loop_detection = False  # disable loop detection (needs vocab tree)

    if matcher == "sequential":
        pycolmap.match_sequential(
            database_path=db_path,
            matching_options=match_opts,
            pairing_options=seq_opts,
            verification_options=verify_opts,
        )
    elif matcher == "exhaustive":
        pycolmap.match_exhaustive(
            database_path=db_path,
            matching_options=match_opts,
            verification_options=verify_opts,
        )
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    # Incremental mapping
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_num_matches = 10

    maps = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=frames_dir,
        output_path=workspace / "sparse",
        options=mapper_opts,
    )

    if not maps:
        raise RuntimeError(
            "COLMAP reconstruction failed — no cameras registered.\n"
            "Possible causes:\n"
            "  • Room has featureless walls/carpet — try --pose-method mast3r\n"
            "  • Video is too dark or blurry\n"
            "  • Try increasing --sequential-overlap (default 10)"
        )

    best = max(maps.values(), key=lambda r: len(r.images))
    registered = len(best.images)
    pct = registered / n_images * 100

    if pct < 80:
        console.print(
            f"  [yellow]Warning: only {registered}/{n_images} frames registered ({pct:.0f}%).[/]\n"
            "  Consider --pose-method mast3r for better coverage."
        )

    best.write(workspace / "sparse" / "0")

    # Move sparse/0 to the standard output location
    import shutil
    out_sparse = output_dir / "sparse" / "0"
    out_sparse.mkdir(parents=True, exist_ok=True)
    for f in (workspace / "sparse" / "0").iterdir():
        shutil.copy2(f, out_sparse / f.name)

    if undistort:
        undistorted_dir = output_dir / "undistorted"

        console.print(
            f"  [dim]Running pycolmap undistorter → {undistorted_dir}[/]"
        )

        pycolmap.undistort_images(
            output_path=undistorted_dir,
            input_path=workspace / "sparse" / "0",
            image_path=frames_dir,
        )

        # Replace training inputs with undistorted data
        train_images = output_dir / "images"
        train_sparse = output_dir / "sparse"

        if train_images.exists():
            shutil.rmtree(train_images)

        if train_sparse.exists():
            shutil.rmtree(train_sparse)

        shutil.copytree(undistorted_dir / "images", train_images)

        train_sparse_0 = output_dir / "sparse" / "0"
        train_sparse_0.mkdir(parents=True, exist_ok=True)

        src_sparse = undistorted_dir / "sparse"
        src_sparse_0 = undistorted_dir / "sparse" / "0"

        if src_sparse_0.exists():
            src = src_sparse_0
        else:
            src = src_sparse

        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, train_sparse_0 / f.name)

    return {
        "method": "colmap",
        "n_frames": n_images,
        "registered": registered,
        "pct_registered": pct,
        "n_points3D": len(best.points3D),
    }


# ── MASt3R ─────────────────────────────────────────────────────────────────────

def run_mast3r(
    frames_dir: Path,
    output_dir: Path,
    mast3r_dir: Path,
    image_size: int = 256,
    max_frames: int = 30,
    niter: int = 100,
    device: str = "cuda",
    align_device: str = "cuda",
) -> dict:
    """
    Estimate camera poses with MASt3R (deep-learning SfM).

    MASt3R works by running a Vision Transformer on all image pairs and jointly
    optimising globally consistent poses. It handles featureless indoor rooms
    where COLMAP's SIFT features are sparse.

    For video input we subsample to max_frames to keep pairwise inference
    tractable (max_frames*(max_frames-1)/2 pairs; 80 frames = 3,160 pairs ≈ 4 min).

    Args:
        frames_dir:    Directory of frame_XXXXX.jpg files
        output_dir:    Where to write sparse/0/
        mast3r_dir:    Root of the cloned MASt3R repository
        image_size:    Inference resolution (512 recommended)
        max_frames:    Maximum frames to use (subsample if more)
        device:        'cuda' or 'cpu'

    Returns:
        dict with registration stats
    """
    dust3r_dir = mast3r_dir / "dust3r"
    for p in (mast3r_dir, dust3r_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    ckpt_dir = mast3r_dir / "checkpoints"
    checkpoint = None
    for pattern in ("*.pth", "*.safetensors"):
        candidates = list(ckpt_dir.glob(pattern))
        if candidates:
            checkpoint = candidates[0]
            break
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            f"MASt3R checkpoint not found under {ckpt_dir}.\n"
            "Download it from https://github.com/naver/mast3r#checkpoints"
        )

    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from scipy.spatial.transform import Rotation

    all_frames = sorted(frames_dir.glob("*.jpg"))
    if not all_frames:
        raise FileNotFoundError(f"No .jpg frames in {frames_dir}")

    # Subsample evenly if we have more than max_frames
    if len(all_frames) > max_frames:
        indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
        frame_paths = [all_frames[i] for i in indices]
        console.print(f"  [dim]MASt3R: subsampled {len(all_frames)} → {len(frame_paths)} frames[/]")
    else:
        frame_paths = all_frames

    n = len(frame_paths)
    console.print(f"  [dim]MASt3R: {n} frames, {n*(n-1)//2} pairs, device={device}[/]")

    model = AsymmetricMASt3R.from_pretrained(str(checkpoint))
    model = model.to(device).eval()

    import torch
    images = load_images([str(p) for p in frame_paths], size=image_size, verbose=False)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=True)

    # Free inference model before alignment — reclaims ~2.6 GB VRAM so the
    # alignment optimizer fits on GPU when align_device="cuda".
    del model
    torch.cuda.empty_cache()

    scene = global_aligner(
        output,
        device=align_device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        verbose=True,
    )
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)

    console.print(f"  [dim]MASt3R global alignment loss: {loss:.4f}[/]")

    poses   = scene.get_im_poses()  # extract results before freeing scene
    focals  = scene.get_focals()
    pts3d   = scene.get_pts3d()
    conf    = scene.get_conf()
    imgs_np = scene.imgs

    # Build COLMAP-format dicts
    true_shape = images[0]["true_shape"]
    mast3r_h = int(true_shape[0, 0]) if true_shape.ndim == 2 else int(true_shape[0])
    mast3r_w = int(true_shape[0, 1]) if true_shape.ndim == 2 else int(true_shape[1])

    cameras_dict = {}
    images_dict  = {}

    for i, fp in enumerate(frame_paths):
        import cv2 as _cv2
        orig = _cv2.imread(str(fp))
        oh, ow = orig.shape[:2]
        f_scaled = float(focals[i].detach().cpu().item()) * ((ow / mast3r_w + oh / mast3r_h) / 2)

        cameras_dict[i + 1] = {
            "model_id": 1,  # PINHOLE
            "width": ow, "height": oh,
            "params": [f_scaled, f_scaled, ow / 2.0, oh / 2.0],
        }

        c2w = poses[i].detach().cpu().numpy()
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        q = Rotation.from_matrix(R).as_quat()  # xyzw

        images_dict[i + 1] = {
            "qw": float(q[3]), "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]),
            "tx": float(t[0]), "ty": float(t[1]), "tz": float(t[2]),
            "camera_id": i + 1,
            "name": fp.name,
        }

    # Sparse point cloud from high-confidence MASt3R predictions
    pts_xyz, pts_rgb = [], []
    for i in range(n):
        pts = pts3d[i].detach().cpu().numpy().reshape(-1, 3)
        cf  = conf[i].detach().cpu().numpy().reshape(-1)
        rgb = (imgs_np[i] * 255).astype(np.uint8).reshape(-1, 3)
        mask = cf > np.percentile(cf, 80)
        pts_xyz.append(pts[mask])
        pts_rgb.append(rgb[mask])

    all_pts = np.concatenate(pts_xyz)
    all_rgb = np.concatenate(pts_rgb)
    if len(all_pts) > 50_000:
        idx = np.random.choice(len(all_pts), 50_000, replace=False)
        all_pts, all_rgb = all_pts[idx], all_rgb[idx]

    points3d_dict = {
        i + 1: {
            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "r": int(c[0]), "g": int(c[1]), "b": int(c[2]),
            "error": 1.0, "track": [],
        }
        for i, (p, c) in enumerate(zip(all_pts, all_rgb))
    }

    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(cameras_dict, sparse_dir / "cameras.bin")
    _write_images_bin(images_dict, sparse_dir / "images.bin")
    _write_points3d_bin(points3d_dict, sparse_dir / "points3D.bin")

    meta = {
        "method": "mast3r",
        "n_frames": len(all_frames),
        "frames_used": n,
        "n_points3D": len(all_pts),
        "final_loss": float(loss),
    }
    (output_dir / "poses_meta.json").write_text(json.dumps(meta, indent=2))

    # Free alignment scene so 3DGS training starts with clean VRAM.
    del scene
    torch.cuda.empty_cache()

    console.print(f"  [dim]MASt3R: {n} poses written, {len(all_pts)} sparse points[/]")
    return meta


# ── Dispatch ───────────────────────────────────────────────────────────────────

_COLMAP_KEYS = {"camera_model", "sequential_overlap", "min_num_inliers", "undistort", "matcher"}
_MAST3R_KEYS  = {"image_size", "max_frames", "device", "align_device", "niter"}

def estimate_poses(
    frames_dir: Path,
    output_dir: Path,
    method: Literal["colmap", "mast3r"] = "colmap",
    mast3r_dir: Path | None = None,
    **kwargs,
) -> dict:
    if method == "colmap":
        colmap_kwargs = {k: v for k, v in kwargs.items() if k in _COLMAP_KEYS}
        return run_colmap(frames_dir, output_dir, **colmap_kwargs)
    elif method == "mast3r":
        if mast3r_dir is None:
            from roomsplat.config import get_mast3r_dir
            mast3r_dir = get_mast3r_dir()
        mast3r_kwargs = {k: v for k, v in kwargs.items() if k in _MAST3R_KEYS}
        return run_mast3r(frames_dir, output_dir, mast3r_dir=mast3r_dir, **mast3r_kwargs)
    else:
        raise ValueError(f"Unknown pose method: {method!r}. Use 'colmap' or 'mast3r'.")


# ── COLMAP binary writers (self-contained, no external deps) ───────────────────

def _write_cameras_bin(cameras: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(cameras)))
        for cam_id, cam in cameras.items():
            params = cam["params"]
            f.write(struct.pack("I", cam_id))
            f.write(struct.pack("i", cam["model_id"]))
            f.write(struct.pack("Q", cam["width"]))
            f.write(struct.pack("Q", cam["height"]))
            f.write(struct.pack(f"{len(params)}d", *params))


def _write_images_bin(images: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(images)))
        for im_id, im in images.items():
            f.write(struct.pack("I", im_id))
            f.write(struct.pack("4d", im["qw"], im["qx"], im["qy"], im["qz"]))
            f.write(struct.pack("3d", im["tx"], im["ty"], im["tz"]))
            f.write(struct.pack("I", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("Q", 0))  # 0 2D points


def _write_points3d_bin(points: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(points)))
        for pt_id, pt in points.items():
            f.write(struct.pack("Q", pt_id))
            f.write(struct.pack("3d", pt["x"], pt["y"], pt["z"]))
            f.write(struct.pack("3B", pt["r"], pt["g"], pt["b"]))
            f.write(struct.pack("d", pt["error"]))
            f.write(struct.pack("Q", len(pt["track"])))
            for image_id, pt2d_idx in pt["track"]:
                f.write(struct.pack("II", image_id, pt2d_idx))

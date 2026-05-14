"""
Video → JPEG frames.

Strategy for a phone room walkthrough:
- Skip first/last 5% of video (entry/exit artifacts)
- Extract at target_fps (default 2.5 fps → ~150 frames for a 60-sec video)
- Resize so the longest edge is max_dim (default 1600px) — matches 3DGS training defaults
- Skip near-duplicate frames using frame difference threshold (handles when user pauses)
"""

from pathlib import Path

import cv2
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 2.5,
    max_dim: int = 1600,
    jpeg_quality: int = 95,
    skip_similar_threshold: float = 0.02,
) -> int:
    """
    Extract frames from a phone walkthrough video.

    Args:
        video_path:             Path to the input video (MP4, MOV, etc.)
        output_dir:             Directory where frame_XXXX.jpg files are written
        target_fps:             Target extraction frame rate (default 2.5 fps)
        max_dim:                Resize so longest edge ≤ max_dim (aspect preserved)
        jpeg_quality:           JPEG compression quality 0-100
        skip_similar_threshold: Skip a frame if mean absolute diff from previous
                                is below this fraction of 255. Handles pauses.
                                Set to 0 to disable.

    Returns:
        Number of frames written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps if src_fps > 0 else 0

    if src_fps <= 0:
        raise ValueError(f"Could not read FPS from video: {video_path}")

    # Skip first/last 5% — entry and exit frames are usually blurry/partial
    start_frame = int(n_frames * 0.05)
    end_frame = int(n_frames * 0.95)
    usable_frames = end_frame - start_frame

    stride = max(1, round(src_fps / target_fps))
    candidate_indices = list(range(start_frame, end_frame, stride))

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"  Extracting frames (stride={stride}, ~{len(candidate_indices)} candidates)",
            total=len(candidate_indices),
        )

        written = 0
        prev_gray = None

        for idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                progress.advance(task)
                continue

            # Skip near-duplicate frames (user paused, shaky video staying still)
            if skip_similar_threshold > 0 and prev_gray is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
                if diff < skip_similar_threshold:
                    progress.advance(task)
                    continue
                prev_gray = gray
            elif prev_gray is None:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize: longest edge → max_dim
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                # Round to nearest even pixel — avoids 1-pixel width mismatches
                # between OpenCV output and COLMAP's stored camera dimensions
                new_w = int(round(w * scale / 2)) * 2
                new_h = int(round(h * scale / 2)) * 2
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = output_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            written += 1
            progress.advance(task)

    cap.release()

    if written < 20:
        raise RuntimeError(
            f"Only {written} frames extracted — too few for a good reconstruction.\n"
            "Record a longer video (30-60 sec) and walk slowly around the room.\n"
            "Try lowering --fps or increasing --skip-similar-threshold."
        )

    return written


def video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":      cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["n_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info

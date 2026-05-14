"""
roomsplat CLI

Commands:
  run   <video.mp4>   Full pipeline: video → frames → poses → 3DGS → .splat → viewer
  view  <output_dir>  Open an existing scene in the browser viewer
  config              Show / update paths to external deps (gaussian-splatting, MASt3R)
"""

import shlex
import shutil
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="roomsplat",
    help="Real estate room tours from a phone video via 3D Gaussian Splatting.",
    add_completion=False,
)
console = Console()


# ── run ────────────────────────────────────────────────────────────────────────

@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="Path to the phone walkthrough video (MP4, MOV, etc.)")],
    name: Annotated[str, typer.Option("--name", "-n", help="Room name shown in the viewer")] = "",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory (default: <video_stem>_splat/)")] = None,
    fps: Annotated[float, typer.Option("--fps", help="Frame extraction rate (default 2.5 → ~150 frames/min)")] = 2.5,
    max_dim: Annotated[int, typer.Option("--max-dim", help="Resize longest edge to this (default 1600px)")] = 1600,
    pose_method: Annotated[str, typer.Option("--pose-method", help="Pose estimator: 'colmap' (default) or 'mast3r'")] = "colmap",
    sequential_overlap: Annotated[int, typer.Option("--sequential-overlap", help="COLMAP: frames to match on each side (default 10)")] = 10,
    camera_model: Annotated[str, typer.Option("--camera-model", help="COLMAP camera model: SIMPLE_RADIAL, SIMPLE_PINHOLE, PINHOLE, OPENCV")] = "SIMPLE_RADIAL",
    undistort: Annotated[bool, typer.Option("--undistort", help="Run COLMAP image_undistorter after radial reconstruction")] = False,
    iterations: Annotated[int, typer.Option("--iterations", help="3DGS training iterations (30000 ≈ 35 min; 7000 ≈ 5 min preview)")] = 30_000,
    sh_degree: Annotated[int, typer.Option("--sh-degree", help="Spherical harmonics degree 0-3 (higher = better colour)")] = 3,
    eval_mode: Annotated[bool, typer.Option("--eval", help="Hold out every 8th frame as test set (enables PSNR evaluation on unseen views)")] = False,
    skip_extract: Annotated[bool, typer.Option("--skip-extract", help="Skip frame extraction (reuse existing frames/)")] = False,
    skip_poses: Annotated[bool, typer.Option("--skip-poses", help="Skip pose estimation (reuse existing sparse/)")] = False,
    skip_train: Annotated[bool, typer.Option("--skip-train", help="Skip 3DGS training (reuse existing model/)")] = False,
    no_view: Annotated[bool, typer.Option("--no-view", help="Don't open the viewer after export")] = False,
    train_python: Annotated[Optional[str], typer.Option("--train-python", help="Python executable to use for 3DGS training (must have diff_gaussian_rasterization installed)")] = None,
    low_vram: Annotated[bool, typer.Option("--low-vram", help="Use memory-safer 3DGS densification settings (recommended for 16GB GPUs)")] = False,
    antialiasing: Annotated[bool, typer.Option("--antialiasing", help="Enable Mip-Splatting-style EWA antialiasing in 3DGS (often cleaner PLY / less shimmer)")] = False,
    matcher: Annotated[str, typer.Option("--matcher", help="Select either 'sequential' or 'exhaustive' matcher")] = "sequential",
    train_forward: Annotated[
        Optional[str],
        typer.Option(
            "--train-forward",
            help='Extra args passed to gaussian-splatting train.py (shell-quoted), e.g. \'--lambda_dssim 0.25 --densify_grad_threshold 0.00015\'',
        ),
    ] = None,
    mast3r_max_frames: Annotated[int, typer.Option("--mast3r-max-frames", help="Max frames for MASt3R (subsampled if more; 30 ≈ 8 GB RAM, 60 ≈ 28 GB RAM)")] = 30,
) -> None:
    """Run the full pipeline: video → navigable 3DGS room tour."""
    if not video.exists():
        console.print(f"[red]Video not found: {video}[/]")
        raise typer.Exit(1)

    # Resolve output directory
    out_dir = output or video.parent / f"{video.stem}_splat"
    out_dir.mkdir(parents=True, exist_ok=True)
    room_name = name or video.stem.replace("_", " ").title()

    _print_header(room_name, video, out_dir, iterations)

    # Validate external deps
    from roomsplat.config import validate, get_gs_dir, get_mast3r_dir
    issues = validate()
    # MASt3R is optional — only fail if selected
    gs_issues = [i for i in issues if "gaussian-splatting" in i]
    if gs_issues and not skip_train:
        for issue in gs_issues:
            console.print(f"[red]{issue}[/]")
        raise typer.Exit(1)

    mast3r_issues = [i for i in issues if "MASt3R" in i]
    if mast3r_issues and pose_method == "mast3r":
        for issue in mast3r_issues:
            console.print(f"[red]{issue}[/]")
        raise typer.Exit(1)

    # ── Step 1: Frame extraction ───────────────────────────────────────────────
    frames_dir = out_dir / "frames"
    if not skip_extract:
        console.rule("[bold]Step 1/4  Frame Extraction")
        from roomsplat.pipeline.extract import extract_frames, video_info
        info = video_info(video)
        console.print(
            f"  Video: {info['width']}×{info['height']} @ {info['fps']:.1f} fps, "
            f"{info['duration_sec']:.0f} sec"
        )
        n_frames = extract_frames(
            video_path=video,
            output_dir=frames_dir,
            target_fps=fps,
            max_dim=max_dim,
        )
        console.print(f"  [green]✓ {n_frames} frames → {frames_dir}[/]")
    else:
        n_frames = len(list(frames_dir.glob("*.jpg")))
        console.print(f"[dim]Skipping extraction — {n_frames} existing frames[/]")

    # ── Step 2: Copy frames into images/ for gaussian-splatting ───────────────
    images_dir = out_dir / "images"
    if not skip_poses and not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(frames_dir.glob("*.jpg")):
            shutil.copy2(f, images_dir / f.name)

    # ── Step 3: Pose estimation ────────────────────────────────────────────────
    sparse_dir = out_dir / "sparse" / "0"
    if not skip_poses:
        console.rule("[bold]Step 2/4  Camera Pose Estimation")
        console.print(f"  Method: [bold]{pose_method}[/]")
        from roomsplat.pipeline.poses import estimate_poses
        meta = estimate_poses(
            frames_dir=images_dir,
            output_dir=out_dir,
            method=pose_method,
            sequential_overlap=sequential_overlap,
            camera_model=camera_model,
            undistort=undistort,
            mast3r_dir=get_mast3r_dir() if pose_method == "mast3r" else None,
            max_frames=mast3r_max_frames,
            matcher=matcher,
        )
        _print_pose_summary(meta)
    else:
        console.print(f"[dim]Skipping pose estimation — using {sparse_dir}[/]")

    if not sparse_dir.exists():
        console.print(f"[red]Sparse directory missing: {sparse_dir}[/]")
        raise typer.Exit(1)

    # ── Step 4: 3DGS Training ──────────────────────────────────────────────────
    model_dir = out_dir / "model"
    if not skip_train:
        console.rule("[bold]Step 3/4  3DGS Training")
        from roomsplat.pipeline.train import train
        train_extra_args = []
        if eval_mode:
            train_extra_args.append("--eval")
        if low_vram:
            # Prevent runaway Gaussian growth during densification on smaller GPUs.
            train_extra_args.extend([
                "--densify_until_iter", "8000",
                "--densification_interval", "200",
            ])
            console.print(
                "  [dim]Low-VRAM mode: densify_until_iter=8000, "
                "densification_interval=200[/]"
            )
        if antialiasing:
            train_extra_args.append("--antialiasing")
            console.print("  [dim]Antialiasing enabled (--antialiasing)[/]")
        if train_forward:
            try:
                parsed = shlex.split(train_forward)
            except ValueError as e:
                console.print(f"[red]Invalid --train-forward (shlex): {e}[/]")
                raise typer.Exit(1)
            train_extra_args.extend(parsed)
            console.print(f"  [dim]train-forward: {' '.join(parsed)}[/]")
        train(
            source_path=out_dir,
            model_path=model_dir,
            gs_dir=get_gs_dir(),
            iterations=iterations,
            sh_degree=sh_degree,
            train_python=train_python,
            extra_args=train_extra_args or None,
        )
    else:
        console.print(f"[dim]Skipping training — using {model_dir}[/]")

    # ── Step 5: Export to .splat ───────────────────────────────────────────────
    console.rule("[bold]Step 4/4  Export to .splat")
    from roomsplat.pipeline.train import find_ply, count_gaussians
    ply_path = find_ply(model_dir, iterations)
    if not ply_path.exists():
        console.print(f"[red]Trained PLY not found: {ply_path}[/]")
        raise typer.Exit(1)

    viewer_dir = out_dir / "viewer"
    splat_path = viewer_dir / "scene.splat"
    _install_viewer(viewer_dir, room_name, ply_path)

    from roomsplat.pipeline.export import ply_to_splat, write_cameras_json
    ply_to_splat(ply_path, splat_path)
    write_cameras_json(sparse_dir, viewer_dir)

    # ── Done ───────────────────────────────────────────────────────────────────
    n_gaussians = count_gaussians(ply_path)
    console.print(
        Panel(
            f"[bold green]Done![/]  {n_gaussians:,} Gaussians\n\n"
            f"Output: [cyan]{out_dir}[/]\n"
            f"Viewer: [cyan]{viewer_dir / 'index.html'}[/]\n\n"
            f"To reopen: [bold]roomsplat view {out_dir}[/]",
            title=f"[bold]{room_name}",
        )
    )

    if not no_view:
        _open_viewer(viewer_dir)


# ── view ───────────────────────────────────────────────────────────────────────

@app.command()
def view(
    output_dir: Annotated[Path, typer.Argument(help="roomsplat output directory containing viewer/")],
    port: Annotated[int, typer.Option("--port", help="HTTP server port (default 8080)")] = 8080,
) -> None:
    """Open an existing room scene in the browser."""
    viewer_dir = output_dir / "viewer"
    if not (viewer_dir / "index.html").exists():
        console.print(f"[red]Viewer not found in {output_dir}[/]")
        raise typer.Exit(1)
    _open_viewer(viewer_dir, port=port)


# ── config ─────────────────────────────────────────────────────────────────────

@app.command()
def config(
    gs_dir: Annotated[Optional[Path], typer.Option("--gs-dir", help="Path to gaussian-splatting repo")] = None,
    mast3r_dir: Annotated[Optional[Path], typer.Option("--mast3r-dir", help="Path to MASt3R repo")] = None,
    show: Annotated[bool, typer.Option("--show", help="Show current config")] = False,
) -> None:
    """Show or update paths to external dependencies."""
    from roomsplat.config import save, get_gs_dir, get_mast3r_dir, validate

    if gs_dir is not None or mast3r_dir is not None:
        save(gs_dir=gs_dir, mast3r_dir=mast3r_dir)
        console.print("[green]Config saved.[/]")

    t = Table(title="roomsplat config")
    t.add_column("Setting", style="bold")
    t.add_column("Value")
    t.add_column("Status")
    gs = get_gs_dir()
    mast3r = get_mast3r_dir()
    t.add_row("gs_dir", str(gs), "[green]OK[/]" if (gs / "train.py").exists() else "[red]NOT FOUND[/]")
    t.add_row("mast3r_dir", str(mast3r), "[green]OK[/]" if (mast3r / "mast3r" / "model.py").exists() else "[yellow]NOT FOUND (optional)[/]")
    console.print(t)

    issues = validate()
    if issues:
        console.print("\n[yellow]Issues:[/]")
        for issue in issues:
            console.print(f"  • {issue}")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _print_header(name: str, video: Path, out_dir: Path, iterations: int) -> None:
    console.print(
        Panel(
            f"Room: [bold]{name}[/]\n"
            f"Video:  {video}\n"
            f"Output: {out_dir}\n"
            f"Iterations: {iterations:,}  (~{iterations / 30_000 * 35:.0f} min)",
            title="[bold cyan]roomsplat[/]",
        )
    )


def _print_pose_summary(meta: dict) -> None:
    if meta.get("method") == "colmap":
        pct = meta.get("pct_registered", 0)
        colour = "green" if pct >= 80 else "yellow"
        console.print(
            f"  [{colour}]Registered {meta['registered']}/{meta['n_frames']} frames "
            f"({pct:.0f}%), {meta['n_points3D']:,} sparse points[/]"
        )
    else:
        console.print(
            f"  [green]MASt3R: {meta['frames_used']} frames, "
            f"{meta['n_points3D']:,} sparse points[/]"
        )


def _install_viewer(viewer_dir: Path, room_name: str, ply_path: Path | None = None) -> None:
    """Copy static viewer assets into the output viewer/ directory."""
    viewer_dir.mkdir(parents=True, exist_ok=True)

    # Copy static assets from the package
    static_src = Path(__file__).parent / "viewer" / "static"
    for asset in static_src.iterdir():
        if asset.suffix in (".html", ".js", ".css"):
            shutil.copy2(asset, viewer_dir / asset.name)

    # Patch the room name into index.html
    index = viewer_dir / "index.html"
    if index.exists():
        html = index.read_text()
        html = html.replace("{{ROOM_NAME}}", room_name)
        index.write_text(html)

    # Symlink PLY so the browser can fetch it as scene.ply
    if ply_path is not None and ply_path.exists():
        ply_link = viewer_dir / "scene.ply"
        if ply_link.exists() or ply_link.is_symlink():
            ply_link.unlink()
        ply_link.symlink_to(ply_path.resolve())


def _open_viewer(viewer_dir: Path, port: int = 8080) -> None:
    from roomsplat.viewer.serve import serve
    serve(viewer_dir, port=port)

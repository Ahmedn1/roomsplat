# roomsplat

**Real estate room tours from a short phone video using 3D Gaussian Splatting.**

Record a short walkthrough of any room with your phone, run one command, and get a
navigable 3D scene you can explore in any browser — fully local, no cloud, no subscription.

```bash
roomsplat run living_room.mp4 --name "Living Room"
```

[![Live Demo](https://img.shields.io/badge/▶_Live_Demo-GitHub_Pages-blue)](https://ahmedn1.github.io/roomsplat)

<p align="center">
  <img src="docs/Screenshot from 2026-05-13 22-17-08.png" width="49%" alt="Office scene view 1"/>
  <img src="docs/Screenshot from 2026-05-13 22-18-52.png" width="49%" alt="Office scene view 2"/>
</p>

---

## How it works

```
Phone video  ──►  Frame extraction  ──►  Camera poses  ──►  3DGS training  ──►  Browser viewer
                  (OpenCV, ~150 frames)   (COLMAP SfM)       (gaussian-         (WASD + mouse,
                                                               splatting,         mkkellogg lib)
                                                               ~35–100 min)
```

1. **Frame extraction** — Samples frames at X fps, resizes to a target longest edge,
   skips near-duplicates (handles user pauses mid-walk).
2. **Pose estimation** — COLMAP sequential or exhaustive SfM recovers camera poses for every
   frame. MASt3R is available as a deep-learning fallback for texture-poor rooms.
3. **3DGS training** — The official [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
   trainer builds a scene of 300k–3M Gaussians (30k iter ≈ 90 min; 50k iter ≈ 120 min on RTX 3080).
4. **Export** — The trained `.ply` is converted to the compact `.splat` binary (≈10–60 MB) and a
   `cameras.json` is written so the viewer starts at a real training-camera position.
5. **Viewer** — A self-contained HTML page powered by
   [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D) renders the
   scene with SH2 view-dependent colour and full WASD + mouse navigation.

---

## Prerequisites

| Requirement | Version |
|---|---|
| GPU | NVIDIA ≥ 8 GB VRAM (tested: RTX 3080 16 GB) |
| CUDA | 11.8 or 12.x |
| Python | 3.10+ |

### External repos (one-time setup)

**gaussian-splatting** (required):
```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting ~/gaussian-splatting
cd ~/gaussian-splatting
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

**MASt3R** (optional — fallback for featureless rooms):
```bash
git clone --recursive https://github.com/naver/mast3r ~/mast3r
cd ~/mast3r && pip install -r requirements.txt
mkdir -p checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
     -O checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

**pycolmap** (required for COLMAP pose estimation):
```bash
pip install pycolmap
```

---

## Installation

```bash
git clone https://github.com/Ahmedn1/roomsplat
cd roomsplat
pip install -e .
```

Configure paths to external repos if not installed under `~/`:
```bash
roomsplat config --gs-dir /path/to/gaussian-splatting
roomsplat config --mast3r-dir /path/to/mast3r   # optional
roomsplat config --show
```

---

## Quick start

```bash
# Full pipeline (35–100 min depending on iterations)
roomsplat run living_room.mp4 --name "Living Room"

# Fast preview (~5 min, lower quality)
roomsplat run living_room.mp4 --iterations 7000

# Reopen an existing scene
roomsplat view living_room_splat/
```

---

## Options

```
roomsplat run <video> [OPTIONS]

Input / output
  --name TEXT              Room name shown in the viewer  [default: video filename]
  --output PATH            Output directory               [default: <stem>_splat/]
  --no-view                Don't open viewer after export

Frame extraction
  --fps FLOAT              Frame extraction rate          [default: 2.5]
  --max-dim INT            Longest edge resize target     [default: 1600]
  --skip-extract           Reuse existing frames/

Pose estimation
  --pose-method TEXT       'colmap' or 'mast3r'           [default: colmap]
  --camera-model TEXT      SIMPLE_RADIAL | SIMPLE_PINHOLE | PINHOLE
                                                          [default: SIMPLE_RADIAL]
  --matcher TEXT           'sequential' or 'exhaustive'   [default: sequential]
  --sequential-overlap INT COLMAP neighbour window        [default: 10]
  --undistort              Run COLMAP undistorter (requires SIMPLE_RADIAL)
  --skip-poses             Reuse existing sparse/

Training
  --iterations INT         Training iterations            [default: 30000]
  --sh-degree INT          Spherical harmonics degree 0–3 [default: 3]
  --train-forward TEXT     Extra args to gaussian-splatting train.py
                           e.g. '--densify_until_iter 30000 -r 2'
  --low-vram               Conservative densification for ≤16 GB GPUs
  --antialiasing           Mip-Splatting EWA antialiasing
  --skip-train             Reuse existing model/

Advanced
  --eval                   Hold out every 8th frame (enables PSNR reporting)
  --train-python PATH      Python with diff_gaussian_rasterization installed
  --mast3r-max-frames INT  MASt3R frame budget           [default: 30]
```

---

## Output structure

```
living_room_splat/
├── frames/                  # Extracted JPEG frames (~150 files)
├── images/                  # Frames used by gaussian-splatting
├── sparse/0/                # COLMAP camera poses (cameras/images/points3D.bin)
├── model/
│   └── point_cloud/iteration_30000/point_cloud.ply
└── viewer/
    ├── index.html           # Self-contained WebGL viewer
    ├── scene.ply -> ...     # Symlink to the full Gaussian data (for SH2 viewer)
    ├── scene.splat          # Compact binary for legacy .splat viewers
    ├── scene.splat.sh       # SH3 coefficient sidecar
    └── cameras.json         # Camera pose list (initial view selection)
```

---

## Capture tips

- **Walk slowly** — fast motion causes motion blur and bad feature matches
- **Cover all angles** — move around objects, not just along the walls
- **Overlap** — each position should share ~60% of the view with the previous frame
- **Light it well** — bright, even lighting; avoid very dark corners
- **Add texture to bare walls** — Big flat solid color surfaces should not dominate a frame without anchor geometric shapes. For instance, if you have a large empty wall, avoid havingthe camera directly at it. Every frame with that wall must have other objects with it. See the section below
- **Translational Movement**  — Do not rotate the camera/phone but try to do translational movement where the camera center changes between frames 

---

## The featureless wall problem

COLMAP uses SIFT feature matching: it finds distinctive corners and edges in each frame and
matches them across frames. This works brilliantly on textured surfaces (furniture, flooring,
posters) but completely fails on **plain painted walls** — a uniform beige wall has no features
to match, so COLMAP can't determine how two frames relate.

**Symptoms:** < 80% of frames registered, large gaps in the sparse point cloud, blurry or
missing wall regions in the final scene.

**Our workaround:** We taped several pages of newspaper across blank wall sections before
recording. Newspaper provides dense, high-contrast, non-repetitive texture that SIFT can match
reliably. After training the tape marks appear in the scene but are easy to overlook.

**Cleaner alternatives:**
- Hang temporary artwork, maps, or printed patterns on bare walls
- Use a laser texture projector (available cheaply for photography)
- Rooms with existing decoration (art, bookshelves, plants) reconstruct far better than
  empty or freshly painted rooms

**MASt3R fallback:** If the room is unavoidably bare, `--pose-method mast3r` uses a
vision-transformer that understands geometry from shading rather than texture — it handles
featureless rooms much better than COLMAP, at the cost of a slower pose step.

---

## Key learnings

These are lessons from 10+ training runs on a single office scene:

| Learning | Detail |
|---|---|
| **Low FPS helped** | `--fps 0.8` reduced a lot of duplicate frames and processing time without losing useful information |
| **Exhaustive > sequential matching** | `--matcher exhaustive` raised registration from 85% to 99% on a 420-frame walk; worth the extra time |
| **Max-dim sweet spot: 1600** | At 1920+ px the 16 GB GPU OOMs during training from Gaussian accumulation; 1600 px fits comfortably |
| **SH2 ≈ SH3 visually, but saves ~800 MB VRAM** | SH3 stores 48 floats/Gaussian vs 27 for SH2; imperceptible difference in a room scene |
| **Default densification > conservative** | `--densify_grad_threshold 0.0002` with interval 100 produced 2M Gaussians vs only 435k with tighter settings — and lower scale ratios |
| **High scale ratios = streak artifacts** | Training with `-r 2` (half-res loss) lets Gaussians stretch into needles; scale_ratio_p95 was 690:1 vs 98:1 for native-resolution training |
| **mkkellogg viewer > custom WebGL** | A custom GLSL SH3 shader produced persistent rainbow artifacts despite correct data; the mkkellogg library renders SH2 correctly out of the box |
| **`--undistort` needs consistent frame sizes** | If frames/ is reused from a prior run with a different `--max-dim`, COLMAP stores the old width and the undistorter crashes with a 1-pixel width mismatch; delete stale frames/ before changing max-dim |
| **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** | Set automatically by roomsplat; reduces memory fragmentation and often buys 200–400 MB of effective headroom |

---

## Comparison with commercial tools

| | roomsplat | Luma AI | Polycam | Matterport |
|---|---|---|---|---|
| Open source | ✅ | ❌ | ❌ | ❌ |
| Local / offline | ✅ | ❌ (cloud) | Partial | ❌ |
| Phone capture | ✅ | ✅ | ✅ | ❌ (special cam) |
| Representation | 3DGS | NeRF/3DGS | Mesh | Panorama |
| GPU required | Yes (local) | No (cloud) | Optional | No |
| Cost | Free | Subscription | Freemium | Enterprise |

---

## Troubleshooting

**COLMAP registers < 80% of frames**
→ Switch to exhaustive matching: `--matcher exhaustive`
→ Try `--pose-method mast3r` for texture-poor rooms
→ Add newspaper / printed texture to bare walls (see above)
→ Increase overlap: `--sequential-overlap 20`

**Training OOM (GPU out of memory)**
→ Lower resolution: `--max-dim 1600` (safe on 16 GB)
→ Lower SH degree: `--sh-degree 2` (saves ~800 MB at 2M Gaussians)
→ Add `-r 2` to `--train-forward` to train at half resolution
→ Note: conservative densification (`--densify_grad_threshold 0.0004`) avoids OOM but
  produces poor quality (high scale ratios, low Gaussian count)

**`--undistort` crashes with "width mismatch (1152 vs 1151)"**
→ Stale frames from an older run with a different `--max-dim`
→ Fix: `rm -rf <output>/frames <output>/images <output>/colmap_workspace <output>/sparse <output>/undistorted`
→ Re-run without `--skip-extract` so frames are regenerated at the current max-dim

**Viewer shows blank / no scene**
→ Open browser developer console (F12) and check for errors
→ Chrome/Edge have better WebGL2 support than Firefox for large scenes
→ If using the PLY viewer, ensure the `scene.ply` symlink is not broken:
  `ls -la viewer/scene.ply`

---

## Live demo (GitHub Pages)

The `docs/` directory contains a pre-built interactive demo of the office scene (11 MB .splat).
Enable GitHub Pages in your repo settings (main branch → `/docs`) and the demo will be live at:

```
https://ahmedn1.github.io/roomsplat
```

To serve it locally first:
```bash
cd docs && python3 -m http.server 8080
# then open http://localhost:8080
```

---

## License

MIT. See [LICENSE](LICENSE).

The viewer uses [`@mkkellogg/gaussian-splats-3d`](https://github.com/mkkellogg/GaussianSplats3D) (MIT).
gaussian-splatting is © Inria / GRAPHDECO — research use; see their
[license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).

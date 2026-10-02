# Uncertainty-Aware Camera Intrinsics Calibration

Certifiable, information-driven selection of calibration views. Instead of collecting
calibration images ad hoc, we score candidate views with the **marginal calibration Fisher
information** (camera poses eliminated via Schur complement) and pick the subset that best
constrains the intrinsics under a fixed view budget `K`.

- **Criteria:** E-optimal (maximise λ<sub>min</sub>), A-optimal (minimise total variance), D-optimal (maximise log-det)
- **Solvers:** greedy selection and Frank–Wolfe relaxations with duality-gap certificates
- **Backends:** OpenCV and Kalibr bundle adjustment
- **Data:** photorealistic Isaac Sim (ChArUco / AprilGrid, with distortion) and real cameras via a live guided-capture web demo

![Pipeline overview](optimal_camera_placement_new/paper/figs/Teaser.png)

## Live Demo

The live web demo opens a real camera. You first capture 5 seed views; it then estimates
intrinsics, computes the weakest calibration direction, and overlays a **ghost board**
showing where to hold the target next. Coverage, eigenvalue spectrum and residual heatmaps
update live.

[![Live guided calibration demo (time-lapse)](videos/RealWorldDataCollection_timelapse.gif)](videos/RealWorldDataCollection.mkv)

*Time-lapse of a full real-world data collection session. Click the GIF for the full-length recording,
[`videos/RealWorldDataCollection.mkv`](videos/RealWorldDataCollection.mkv) (42 min, 1080p, ~670 MB via Git LFS).*

Run it:

```bash
cd optimal_camera_placement_new
python3 isaac_sim_scripts/live_calibration_web_demo.py \
  --camera-index 0 --board-type charuco \
  --board-rows 11 --board-cols 15 --board-square-size 0.021 --board-marker-size 0.016 \
  --seed-count 5 --next-k 10 --output-dir live_demo_charuco
# then open http://127.0.0.1:8765
```

Recorded sessions are in `optimal_camera_placement_new/live_demo_charuco*/`.

## Experiment Snapshots

**Real-world view diversity:** frames chosen by each criterion (session 3, K = 20, ChArUco).
Information-driven criteria pick far more varied tilts and image-plane placements than random sampling.

![Real-world view diversity](optimal_camera_placement_new/paper/figs/fig_view_diversity.png)

**Noise sensitivity (Isaac Sim, K = 20):** λ<sub>min</sub> and held-out reprojection RMS across
all methods for AprilGrid and ChArUco targets.

![Noise sensitivity](optimal_camera_placement_new/paper/figs/fig1_noise_sensitivity.png)

**View-budget sweep:** held-out reprojection RMS vs. K for each noise level.

![K sweep held-out RMS](optimal_camera_placement_new/paper/figs/fig_ksweep_heldout.png)

**Selected views in Isaac Sim** for eight methods:

![Isaac Sim view diversity](optimal_camera_placement_new/paper/figs/fig_isaac_diversity_8methods.png)

### Key findings

- Information-driven selection consistently beats random sampling and coverage heuristics in both FIM quality and downstream calibration accuracy, in simulation and on real cameras.
- Frank–Wolfe certificates are tight in practice: rounded solutions are within 2.5 % of the true E-optimum, exact for D-optimality, and within 10.6 % for A-optimality on exhaustively solvable instances.
- **A-optimality is the most robust default.** It handles high corner noise and small budgets (K = 5–15), and it matches what joint bundle-adjustment backends such as Kalibr need.
- E-optimality works well at moderate noise with K ≥ 20. Greedy-E has the best focal-length error on AprilGrid at σ = 0.5 and 1.0 px.
- Training reprojection error is a poor proxy for calibration quality, so we report held-out RMS and grouped parameter errors (ε<sub>f</sub>, ε<sub>pp</sub>, ε<sub>d</sub>, ε<sub>R</sub>, ε<sub>t</sub>).

Full results are in the paper draft (`optimal_camera_placement_new/paper/icra2026_draft.tex`) and in
`optimal_camera_placement_new/results/`.

## Repository Layout

```
.
├── README.md
├── videos/                         # live demo recording + GIF preview (Git LFS)
└── optimal_camera_placement_new/
    ├── OASIS/                      # FIM, Schur marginalisation, E/A/D criteria, greedy + Frank–Wolfe, analysis/plots
    ├── DataGenerator/              # synthetic candidate banks and measurement generation
    ├── Experiments/                # benchmarks, sweeps, certificate validation, Kalibr and real-data studies
    ├── isaac_sim_scripts/          # Isaac Sim capture, detection, Kalibr export, interactive loop, live web demo
    ├── live_demo_charuco*/         # recorded real-world capture sessions
    ├── paper/                      # paper source and figures
    └── results/                    # benchmark summaries, tables and plots
```

Large binaries (`*.png`, `*.jpg`, `*.pdf`, `*.gif`, `*.mkv`) are stored with **Git LFS**. Run `git lfs install` before cloning, or `git lfs pull` afterwards.
Kalibr rosbags, raw frame dumps, `.npz` problem files and `.g2o` datasets are not versioned; regenerate them with the scripts below.

## Setup

```bash
git lfs install
git clone git@github.com:mahajanparth/Uncertainity_Aware_Calibration.git
cd Uncertainity_Aware_Calibration/optimal_camera_placement_new
pip install numpy scipy matplotlib opencv-contrib-python joblib   # gtsam optional (FIM backend)
```

**All commands below are run from `optimal_camera_placement_new/`.** Isaac Sim scripts use the Isaac Sim Python
environment (`/home/parth/anaconda3/envs/env_isaacsim/bin/python` in the examples).

## Benchmarks (paper results)

```bash
# All K=20 benchmarks (AprilGrid/ChArUco x noise x distortion)
python3 -m Experiments.run_all_k20 --skip-existing

# LaTeX tables with grouped calibration errors
python3 -m Experiments.generate_k20_tables --results-dir results/k20 --output-dir results/k20_analysis

# Benchmark one exported problem
python3 -m Experiments.main_benchmark --problem isaac_outputs/.../all_candidate_problem.npz --select-k 20

# K-budget sweep
python3 -m Experiments.main_k_sweep_calib_metrics \
  --problem results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz \
  --tag aprilgrid_zero --select-ks 5 10 15 20 30 40 50 --output-dir results/k_sweep_calib_nk

# Frank–Wolfe certificate validation against exhaustive search
python3 -m Experiments.run_certificate_validation --output-dir results/certificate_validation

# Kalibr bundle-adjustment backend
python3 -m Experiments.main_kalibr_isaacsim --session-dir isaac_outputs/<run>/ --output-dir results/kalibr_isaacsim --select-k 20

# Real-world ChArUco study on a recorded live-demo session
python3 -m Experiments.main_real_charuco \
  --captures-dir live_demo_charuco/captures --summary-json live_demo_charuco/live_session_summary.json \
  --select-k 20 --output-dir results/real
```

## Synthetic Experiments

Single synthetic problem:

```bash
python3 Experiments/main.py
```

Average across multiple synthetic problems:

```bash
python3 Experiments/main_expectation.py --num_runs 5 --num-candidate-poses 1000 --select_k 20 --radius-samples 3 --min-radius 0.6 --max-radius 1.2
```

Sweeps:

```bash
python3 Experiments/main_checkerboard_spacing.py --square-sizes 0.03 0.1 0.2 0.4 0.6 0.8 1.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
python3 Experiments/main_pixel_noise.py --pixel-noises 0.1 0.5 1.0 2.0 3.0 5.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
python3 Experiments/main_azimuth_span.py --azimuth-spans-deg 30 60 90 120 180 360 --num_runs 5 --num-candidate-poses 1000 --select_k 20
python3 Experiments/main_candidate_k_sweep.py --candidate-counts 10 25 50 100 200 400 700 1000 --select-ks 10 20 40 60 80 100 --num_runs 3
```

Fixed-camera weakness suite:

```bash
python3 Experiments/main_fixed_camera_weaknesses.py --modes limited_viewpoint_diversity narrow_azimuth_coverage little_depth_variation little_tilt_variation poor_image_plane_coverage small_target_footprint too_few_visible_points symmetric_geometry planar_pose_degeneracy high_pixel_noise
```

## Isaac Sim Synthetic Data Workflow

Generate synthetic calibration datasets in Isaac Sim and convert them into the `CalibrationProblem` format used by the observability analysis.

### 1. Capture a synthetic ChArUco dataset

```bash
/home/parth/anaconda3/envs/env_isaacsim/bin/python isaac_sim_scripts/isaacsim_free_camera_checkerboard_capture.py \
  --output-dir isaac_outputs/free_camera_run \
  --board-type charuco \
  --board-rows 11 \
  --board-cols 15 \
  --board-square-size 0.021 \
  --board-marker-size 0.016 \
  --num-images 80 \
  --headless
```

This writes RGB frames under `rgb/` and `dataset_metadata.json` with camera intrinsics, board metadata and per-frame camera poses.

### 2. Detect ChArUco corners

```bash
python3 isaac_sim_scripts/detect_checkerboard_uv.py \
  --input isaac_outputs/free_camera_run/rgb \
  --output-dir isaac_outputs/free_camera_run/detections \
  --board-type charuco \
  --rows 11 \
  --cols 15 \
  --square-size 0.021 \
  --marker-size 0.016
```

### 3. Export to a calibration problem

Ground-truth projection path:

```bash
python3 isaac_sim_scripts/export_isaacsim_calibration_problem.py \
  --metadata isaac_outputs/free_camera_run/dataset_metadata.json \
  --output isaac_outputs/free_camera_run/calibration_problem.npz
```

Detection-driven path:

```bash
python3 isaac_sim_scripts/export_isaacsim_calibration_problem.py \
  --metadata isaac_outputs/free_camera_run/dataset_metadata.json \
  --detections isaac_outputs/free_camera_run/detections/detections.json \
  --use-detections \
  --output isaac_outputs/free_camera_run/calibration_problem_detected.npz
```

### 4. Run observability-aware pose selection

```bash
python3 Experiments/main_isaacsim_dataset.py \
  --problem isaac_outputs/free_camera_run/calibration_problem.npz \
  --select-k 20
```

## Isaac Sim Batch Recommendation Loop

Mimics "capture an initial set, then ask for the next best `k` views":

```bash
/home/parth/anaconda3/envs/env_isaacsim/bin/python isaac_sim_scripts/isaacsim_interactive_calibration_loop.py \
  --output-dir isaac_outputs/interactive_run \
  --board-type charuco \
  --board-rows 11 \
  --board-cols 15 \
  --board-square-size 0.021 \
  --board-marker-size 0.016 \
  --initial-k 5 \
  --max-captures 20
```

The script samples a candidate bank of camera poses, captures seed images, estimates `intrinsics_init`, computes
the weakest calibration direction, and recommends the next batch from the remaining candidates. It repeats until
`--max-captures` or the stopping condition is reached.

- `--recommendation-batch-size 5` gives smaller repeated batches (default: one batch for all remaining captures).
- `--interactive` pauses for confirmation before each recommended batch.
- `--pixel-noise-sigma 0` gives noiseless measurements (the FIM still uses a tiny numerical noise floor).
- Any nonzero `--dist-*` coefficient saves raw renders under `captures/original/` and distorted renders under `captures/distorted/`.
- `--plot-points` visualises candidate/selected poses; `--stay-open` keeps Isaac Sim open after the run.

Outputs: captured images under `captures/`, `all_candidate_problem.npz`, `interactive_summary.json`, and optionally a ROS1 bag and Kalibr target YAML.

### AprilGrid / Kalibr

For AprilGrid, `--board-rows`/`--board-cols` are tag grid dimensions, `--board-marker-size` is the tag size and
`--board-square-size` is the tag pitch (spacing = `square_size / marker_size - 1`). For a Kalibr 6x6 AprilGrid
with tag size `0.088 m` and spacing `0.0264 m`, use pitch `0.1144` and board size `0.8`:

```bash
source devel/setup.bash
/home/parth/anaconda3/envs/env_isaacsim/bin/python isaac_sim_scripts/isaacsim_interactive_calibration_loop.py \
  --output-dir isaac_outputs/interactive_run_aprilgrid_kalibr \
  --board-type aprilgrid \
  --board-rows 6 \
  --board-cols 6 \
  --board-square-size 0.1144 \
  --board-marker-size 0.088 \
  --aprilgrid-board-size 0.8 \
  --radii 1.0 1.2 1.4 1.6 1.8 \
  --elevations-deg 25 35 45 55 65 \
  --initial-k 5 \
  --max-captures 20 \
  --rosbag-output isaac_outputs/interactive_run_aprilgrid_kalibr/cam_april.bag \
  --rosbag-topics /cam0/image_raw \
  --kalibr-target-output isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml

rosrun kalibr kalibr_calibrate_cameras \
  --bag isaac_outputs/interactive_run_aprilgrid_kalibr/cam_april.bag \
  --target isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml \
  --models pinhole-radtan \
  --topics /cam0/image_raw
```

Generate and test the AprilGrid texture outside Isaac:

```bash
python3 isaac_sim_scripts/generate_kalibr_aprilgrid_target.py \
  --output isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6.png \
  --yaml isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml \
  --rows 6 --cols 6 --tag-size 0.088 --tag-spacing 0.3 --outer-margin-pitches 0.5

python3 isaac_sim_scripts/detect_kalibr_aprilgrid.py \
  --image isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6.png \
  --output isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6_detected.png \
  --rows 6 --cols 6
```

If the Isaac Python environment cannot import `rosbag`, the run still saves images and `interactive_summary.json`. Export the bag afterwards from a sourced ROS shell:

```bash
source devel/setup.bash
python3 isaac_sim_scripts/export_summary_to_rosbag.py \
  --summary isaac_outputs/interactive_run_aprilgrid_kalibr/interactive_summary.json \
  --bag isaac_outputs/interactive_run_aprilgrid_kalibr/cam_april.bag \
  --topics /cam0/image_raw
```

To keep Isaac's camera poses but swap in externally extracted 2D–3D matches (e.g. from Kalibr):

```bash
python3 isaac_sim_scripts/build_calibration_problem_from_matches.py \
  --summary isaac_outputs/interactive_run_aprilgrid_kalibr/interactive_summary.json \
  --base-problem isaac_outputs/interactive_run_aprilgrid_kalibr/all_candidate_problem.npz \
  --matches isaac_outputs/interactive_run_aprilgrid_kalibr/kalibr_matches.json \
  --output isaac_outputs/interactive_run_aprilgrid_kalibr/kalibr_matched_problem.npz
```

The matches JSON has one record per image:

```json
{
  "images": [
    {
      "image_name": "seed_000_pose_0000.png",
      "detected": true,
      "corner_ids": [0, 1, 2],
      "uv": [[612.1, 301.2], [640.8, 301.6], [669.0, 302.0]]
    }
  ]
}
```

The loop captures a single camera stream. Passing two `--rosbag-topics` writes the same images to both topics,
which is only useful for plumbing tests, not for stereo calibration.

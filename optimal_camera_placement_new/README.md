# Uncertainty-Aware Camera Intrinsics Calibration

This repository is being shaped around an interactive calibration workflow:

1. the user first captures an initial set of `k` ChArUco-board images
2. the system estimates camera intrinsics from those images
3. the estimated intrinsics are used as the linearization / initialization point for uncertainty analysis
4. the system then identifies the weakest observed calibration direction
5. the user is asked to capture the next image that improves observability in that worst-case direction

The main idea is not just to calibrate a camera, but to guide data collection so each additional view reduces uncertainty where the current calibration is weakest.

## Intended Objective

After an initial intrinsic estimate is available, we evaluate candidate next views using the calibration information matrix and prefer views that improve the smallest eigenvalue:

```math
\max_{s \in \{0,1\}^N,\ \mathbf{1}^\top s = k} \lambda_{\min}(H_{cal}(s))
```

where:

- `H_cal` is the intrinsics-only information matrix after marginalizing nuisance pose variables
- `lambda_min(H_cal)` measures the worst-constrained calibration direction
- increasing this value makes the calibration more robust in the weakest direction

In practical terms, the next suggested board pose should be the one that most improves observability of the least certain intrinsic mode.

## Planned User Workflow

1. Capture `k` seed images of a ChArUco board with reasonable pose diversity.
2. Detect board corners and estimate initial intrinsics.
3. Build the calibration Fisher / information matrix around that estimate.
4. Analyze the eigendirections of the resulting uncertainty.
5. Recommend a new board pose or camera pose that excites the weakest direction.
6. Add the newly captured image and repeat until the uncertainty is acceptable.

## Current Repository Status

The current codebase already contains the core uncertainty-aware selection machinery for synthetic calibration problems:

- synthetic checkerboard / calibration-target generation
- candidate view generation
- information-matrix construction for intrinsics calibration
- greedy pose selection using the minimum-eigenvalue objective
- analysis and visualization of selected versus random view sets

The full interactive ChArUco capture loop is the target workflow, but it is not yet cleanly wired as a single end-to-end user-facing pipeline in this repository.

## Current Experiments

Single synthetic problem:

```bash
python3 Experiments/main.py
```

Average across multiple synthetic problems:

```bash
python3 Experiments/main_expectation.py --num_runs 5 --num-candidate-poses 1000 --select_k 20 --radius-samples 3 --min-radius 0.6 --max-radius 1.2
```

Checkerboard spacing sweep:

```bash
python3 Experiments/main_checkerboard_spacing.py --square-sizes 0.03 0.1 0.2 0.4 0.6 0.8 1.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Pixel noise sweep:

```bash
python3 Experiments/main_pixel_noise.py --pixel-noises 0.1 0.5 1.0 2.0 3.0 5.0 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Azimuth-span sweep:

```bash
python3 Experiments/main_azimuth_span.py --azimuth-spans-deg 30 60 90 120 180 360 --num_runs 5 --num-candidate-poses 1000 --select_k 20
```

Fixed-camera weakness suite:

```bash
python3 Experiments/main_fixed_camera_weaknesses.py --modes limited_viewpoint_diversity narrow_azimuth_coverage little_depth_variation little_tilt_variation poor_image_plane_coverage small_target_footprint too_few_visible_points symmetric_geometry planar_pose_degeneracy high_pixel_noise
```

Candidate-count and selected-`k` sweep:

```bash
python3 Experiments/main_candidate_k_sweep.py --candidate-counts 10 25 50 100 200 400 700 1000 --select-ks 10 20 40 60 80 100 --num_runs 3
```

## Isaac Sim Synthetic Data Workflow

The repository now also supports generating synthetic calibration datasets from Isaac Sim and converting them into the same `CalibrationProblem` format used by the existing observability analysis.

### 1. Capture a synthetic ChArUco dataset in Isaac Sim

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

This writes:

- RGB frames under `rgb/`
- `dataset_metadata.json` with camera intrinsics, board metadata, and per-frame camera poses

### 2. Detect ChArUco corners from the rendered images

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

This writes annotated images, per-image CSV files, and a `detections.json` file that includes ChArUco corner IDs.

### 3. Export the Isaac Sim dataset to a calibration problem

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

### 4. Run observability-aware pose selection on the exported problem

```bash
python3 Experiments/main_isaacsim_dataset.py \
  --problem isaac_outputs/free_camera_run/calibration_problem.npz \
  --select-k 20
```

This produces the same style of selection report, pose plots, and before/after uncertainty analysis used by the original synthetic experiments.

## Isaac Sim Batch Recommendation Loop

If you want the workflow to mimic "capture an initial set, then ask for the next best `k` views," use the Isaac Sim batch recommendation loop instead of the offline subset-selection script.

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

AprilGrid is also supported. For AprilGrid, `--board-rows` and `--board-cols` are the tag grid dimensions, `--board-marker-size` is the tag size, and `--board-square-size` is the tag pitch, so spacing is computed as `(square_size / marker_size) - 1`.

For a Kalibr-compatible 6x6 AprilGrid with tag size `0.088 m` and spacing `0.0264 m`, use tag pitch `0.1144` and board size `0.8`.

```bash
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
  --kalibr-target-output isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml
```

You can also generate and test the AprilGrid texture outside Isaac:

```bash
python3 isaac_sim_scripts/generate_kalibr_aprilgrid_target.py \
  --output isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6.png \
  --yaml isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml \
  --rows 6 \
  --cols 6 \
  --tag-size 0.088 \
  --tag-spacing 0.3 \
  --outer-margin-pitches 0.5

python3 isaac_sim_scripts/detect_kalibr_aprilgrid.py \
  --image isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6.png \
  --output isaac_sim_scripts/generated_assets/kalibr_aprilgrid_6x6_detected.png \
  --rows 6 \
  --cols 6
```

To export the selected images as a ROS1 bag for Kalibr, source ROS first and pass a bag path plus a target YAML path:

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
```

Then run single-camera Kalibr calibration:

```bash
rosrun kalibr kalibr_calibrate_cameras \
  --bag isaac_outputs/interactive_run_aprilgrid_kalibr/cam_april.bag \
  --target isaac_outputs/interactive_run_aprilgrid_kalibr/april_6x6.yaml \
  --models pinhole-radtan \
  --topics /cam0/image_raw
```

If the Isaac Python environment cannot import ROS's `rosbag` module, the run will still save images and `interactive_summary.json`. Export the bag afterwards from a sourced ROS shell:

```bash
source devel/setup.bash
python3 isaac_sim_scripts/export_summary_to_rosbag.py \
  --summary isaac_outputs/interactive_run_aprilgrid_kalibr/interactive_summary.json \
  --bag isaac_outputs/interactive_run_aprilgrid_kalibr/cam_april.bag \
  --topics /cam0/image_raw
```

If you want to keep Isaac's camera poses but replace the synthetic corner measurements with externally extracted matches, export those matches as JSON and build an OASIS problem from them:

```bash
python3 isaac_sim_scripts/build_calibration_problem_from_matches.py \
  --summary isaac_outputs/interactive_run_aprilgrid_kalibr/interactive_summary.json \
  --base-problem isaac_outputs/interactive_run_aprilgrid_kalibr/all_candidate_problem.npz \
  --matches isaac_outputs/interactive_run_aprilgrid_kalibr/kalibr_matches.json \
  --output isaac_outputs/interactive_run_aprilgrid_kalibr/kalibr_matched_problem.npz
```

The matches JSON is expected to contain one record per image with `image_name` or `image_path`, `corner_ids`, and `uv`, for example:

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

This preserves the Isaac-selected pose geometry while swapping in measured 2D-3D correspondences for the captured views, which is the cleanest handoff into the existing OASIS uncertainty analysis.

The current loop captures one camera stream. Passing two `--rosbag-topics` writes the same images to both topics, which is useful only for plumbing tests, not a valid stereo calibration dataset.

Pose markers are not shown by default while capturing. Add `--plot-points` only if you want the candidate and selected poses visualized in the scene.

Isaac Sim closes automatically when the run finishes. Add `--stay-open` if you want to inspect the final scene after capture.

By default, after the initial seed captures, the script recommends all remaining captures up to `--max-captures` as one optimized batch. With the example above, it captures 5 seed views, then computes the next 15 recommended views together.

Use `--pixel-noise-sigma 0` only for noiseless synthetic measurements. The FIM calculation still uses a tiny numerical noise floor to avoid division by zero.

When any `--dist-*` coefficient is nonzero, the loop saves both image versions: raw pinhole renders under `captures/original/` and distorted renders under `captures/distorted/`. The summary `image_path` points to the distorted image and `original_image_path` points to the matching raw render.

If you want smaller repeated batches, add `--recommendation-batch-size`. For example, this captures 5 seed views, then recommends/captures batches of 5 until 20 total captures:

```bash
/home/parth/anaconda3/envs/env_isaacsim/bin/python isaac_sim_scripts/isaacsim_interactive_calibration_loop.py \
  --output-dir isaac_outputs/interactive_run \
  --initial-k 5 \
  --max-captures 20 \
  --recommendation-batch-size 5
```

Optional interactive prompting:

```bash
/home/parth/anaconda3/envs/env_isaacsim/bin/python isaac_sim_scripts/isaacsim_interactive_calibration_loop.py \
  --output-dir isaac_outputs/interactive_run \
  --initial-k 5 \
  --max-captures 20 \
  --interactive
```

This script:

- samples a candidate bank of camera poses in Isaac Sim
- captures the initial seed images
- estimates `intrinsics_init` from those seed captures
- computes the current weakest calibration direction
- recommends the next best batch of poses from the remaining candidates
- optionally pauses for confirmation before capturing that recommended batch
- repeats until `max-captures` is reached or the stopping condition is met

Outputs include:

- captured seed and recommended images under `captures/`
- optional ROS1 bag and Kalibr AprilGrid target YAML
- `all_candidate_problem.npz` for offline analysis
- `interactive_summary.json` with seed-intrinsics, seed eigenvalues, and nested recommended-batch capture history

## Direction For Next Development

The next meaningful step is to connect the current observability analysis to a real-data front end that:

- accepts user-provided ChArUco images
- computes intrinsics from the initial batch
- scores candidate next views using the current uncertainty
- returns human-readable guidance for the next best board motion or camera motion

That would make the repository match the intended interactive calibration workflow described above.

# EIHMR: Collaborative Human-Camera Estimation for Global Human Mesh Recovery

EIHMR recovers world-coordinate 3D human motion and the camera trajectory from a monocular,

## Installation

Tested with Python 3.10, PyTorch 2.4.0 + CUDA 12.1 on Linux.

```bash
pip install -r requirements.txt
```

The following must be installed or downloaded separately (each under its own license):

- **DROID-SLAM** with its CUDA extensions (`droid_backends`, `lietorch`), built for your PyTorch/CUDA.
- **PyTorch3D** 0.7.8.
- **PromptHMR** checkout with its pretrained weights (`data/pretrain/phmr`, `data/pretrain/phmr_vid`,
  `data/pretrain/vitpose-h-coco_25.pth`), its bundled Metric3D (`pipeline/yvanyin_metric3d_main`)
  and its body models (`data/body_models/smplx`, `data/body_models/smpl`).
- **Asset root** with `pretrain/droid.pth`, `pretrain/sam_vit_h_4b8939.pth`,
  `pretrain/camcalib_sa_biased_l2.ckpt` and `smpl/`.
- **A YOLO person detector** checkpoint (e.g. `yolo11x.pt`) for video inference.
- **ZoeDepth**, for metric scale. For offline machines set `TORCH_HOME`, `ZOEDEPTH_REPO`,
  `MIDAS_REPO` and `ZOEDEPTH_CHECKPOINT` to local copies.

SMPL / SMPL-X models are read from `<prompthmr-root>/data/body_models` by default; override with
`--body-model-root` or the `EIHMR_BODY_MODELS` environment variable. The directory must contain
`smplx/SMPLX_NEUTRAL.npz` and `smpl/SMPL_NEUTRAL.pkl`.

The small GVHMR body-model, kinematics and rendering code EIHMR needs is vendored in `lib/gvhmr/`;
a GVHMR checkout is not required.

## Usage

### Video inference

```bash
python infer_video.py \
    --video input.mp4 --output-root runs/my_video \
    --asset-root ASSETS --droid-root DROID-SLAM --prompthmr-root PromptHMR \
    --detector-weights yolo11x.pt \
    [--intrinsics FX FY CX CY] [--gpu 0]
```

- Without `--intrinsics`, the focal length is set to the larger image dimension.
- A single person is tracked; `--track-id` selects a detector track. Missing boxes are
  interpolated up to `--max-track-gap` frames.
- Variable-rate videos must be resampled to a constant frame rate first; `--fps` only
  reinterprets timestamps, and `--frame-step` / `--max-frames` subsample and bound the clip.
- MO-SLAM uses uniform keyframes every `--keyframe-stride` frames (default 5) for every video; the
  rerun on these keyframes only corrects the first-pass camera through clamped per-keyframe deltas.

### EMDB2 evaluation

```bash
python tools/eval_emdb2.py \
    --dataset-root EMDB --asset-root ASSETS --droid-root DROID-SLAM --prompthmr-root PromptHMR \
    --detector-weights yolo11x.pt --output-root runs/emdb2 \
    --gpu-ids 0,1,2,3,4,5,6,7 [--slots-per-gpu 2] [--sequences SEQ1,SEQ2]
```

This runs the full pipeline end to end on the 25 EMDB2 sequences and writes
`per_sequence_results.csv` and `group_summary.csv`. The baseline, PromptHMR + Mask-DROID, is taken
from the same run (first-pass masked DROID camera and PromptHMR body, before SA-LHMR, frontend delta
and camera refine), so no external results are needed. The Short/Medium/Long split in
`configs/emdb2_sequences.csv` (ground-truth camera travel < 20 m, < 60 m, >= 60 m) is only used to
group the report. ATE/ATE-S are sequence means; body metrics are aggregated over valid frames.

### Exporting and evaluating trajectories

```bash
# numerical trajectories from completed camera.npz / smpl.npz results
python export_trajectories.py --results-root RESULTS --output-root EXPORTS \
    --body-model-root BODY_MODELS --fps 30 [--device cuda] \
    [--dataset-root EMDB --asset-root ASSETS --export-emdb-gt]

# standalone metrics (NumPy only)
python eval_trajectories.py --pred PREDICTION_BUNDLE --gt GT_BUNDLE \
    --protocol emdb_legacy --output metrics.json [--allow-partial]
```

## Outputs

Each video run writes `trajectories/camera.npz`, `trajectories/human.npz` and `manifest.json`:

- **camera:** original `frame_ids`, `timestamps` (s), `valid`, camera-to-world `c2w` [F, 4, 4]
  (OpenCV axes) and `K`.
- **human:** `joints22_cam/world`, `root_position_cam/world`, `root_orient_cam/world`,
  `body_pose_aa`, `betas`, offset-aware `transl_cam/world`, `skeleton_offset`; with
  `--export-vertices`, SMPL-topology meshes as well.
- `eval_joints24_world_raw` is unfiltered and `eval_joints24_world` uses the EMDB protocol's
  sigma-3 root filter.

All positions are in meters. The world frame is the final SLAM frame; it is not geographic and not
necessarily gravity-aligned.

## License

EIHMR's own code is released under the [MIT License](LICENSE). Parts of the code are derived from
**TRAM**, **GVHMR** and **PromptHMR**, and some files keep their original terms (full texts in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- TRAM-derived code (most of `lib/`): MIT.
- `lib/gvhmr/` (vendored from GVHMR): educational, research and non-profit use only.
- `lib/utils/rotation_conversions.py` (from PyTorch3D): BSD 3-Clause.
- Portions derived from PromptHMR: PromptHMR License, non-commercial scientific research only.

The pipeline also requires separately obtained components with their own licenses, notably
PromptHMR, the SMPL / SMPL-X body models (both non-commercial) and Ultralytics YOLO (AGPL-3.0).
**As a result, running EIHMR end to end is limited to non-commercial research use.**

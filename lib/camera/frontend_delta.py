"""Frontend-only DROID delta refinement.

Refines an existing metric camera trajectory using a *frontend-only* DROID-SLAM
pass over SMPL-rendered keyframes.

The trajectory produced by the frontend pass lives in DROID's own raw, unscaled
gauge, so it cannot be consumed directly.  Instead we only take the *relative*
SE(3) change each forced keyframe underwent, rescale the translation part into
metric units, clamp it, and interpolate it back onto the baseline trajectory.
Rotation and translation deltas are treated independently, matching the offline
sweep in ``scripts/run_frontend_delta_refine_emdb2_full.py``.

Warm-start poses and disparities must come from a previous full (frontend +
backend) warmstart SLAM pass over the *same* forced keyframes, which is what
makes running the frontend alone sufficient: the depth state is already
initialized, so no extra depth model is needed.
"""

import json
import logging
import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from lib.camera.human_mesh_camera_refine import (
    _interpolate_center_deltas_to_all_frames,
    _so3_exp,
)
from lib.utils.rotation_conversions import quaternion_to_matrix

logger = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _qxyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    q = torch.as_tensor(q_xyzw, dtype=torch.float32)
    return quaternion_to_matrix(q[:, [3, 0, 1, 2]]).detach().cpu().numpy()


def _se3_raw_to_c2w(raw_poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw DROID SE3 poses [n, 7] (w2c) into c2w rotation + translation."""
    from lietorch import SE3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw = torch.as_tensor(raw_poses, dtype=torch.float32, device=device)
    c2w = SE3(raw).inv().data.detach().cpu().numpy()
    return _qxyzw_to_rotmat(c2w[:, 3:]), c2w[:, :3]


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    out = []
    for Ri in R:
        cos = np.clip((np.trace(Ri) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cos))
        if angle < 1e-8:
            out.append(np.zeros(3, dtype=np.float32))
            continue
        axis = np.array([
            Ri[2, 1] - Ri[1, 2],
            Ri[0, 2] - Ri[2, 0],
            Ri[1, 0] - Ri[0, 1],
        ], dtype=np.float64) / (2.0 * np.sin(angle))
        out.append((axis * angle).astype(np.float32))
    return np.stack(out, axis=0)


def _clamp_vectors(x: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return x
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    scale = np.minimum(1.0, float(max_norm) / np.maximum(norm, 1e-12))
    return x * scale


def _estimate_raw_to_metric_scale(raw_t: np.ndarray, metric_t: np.ndarray) -> float:
    """Median per-step ratio between metric and raw translation magnitudes."""
    raw_step = np.linalg.norm(np.diff(raw_t, axis=0), axis=-1)
    metric_step = np.linalg.norm(np.diff(metric_t, axis=0), axis=-1)
    valid = (raw_step > 1e-6) & np.isfinite(raw_step) & np.isfinite(metric_step)
    ratios = metric_step[valid] / np.maximum(raw_step[valid], 1e-12)
    if len(ratios) == 0:
        return 1.0
    return float(np.median(ratios))


def _estimate_raw_to_world_rotation(raw_R: np.ndarray, world_R: np.ndarray) -> np.ndarray:
    """Fit the shared raw-to-world rotation from corresponding c2w orientations."""
    candidates = np.matmul(
        np.asarray(world_R, dtype=np.float64),
        np.swapaxes(np.asarray(raw_R, dtype=np.float64), -1, -2),
    )
    U, _, Vt = np.linalg.svd(candidates.mean(axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U @ Vt)
    return (U @ correction @ Vt).astype(np.float32)


def _saturation_ratio(vectors: np.ndarray, limit: float) -> float:
    """Fraction of deltas whose magnitude reached the clamp limit."""
    if limit <= 0 or len(vectors) == 0:
        return 0.0
    norms = np.linalg.norm(vectors, axis=-1)
    return float(np.mean(norms >= limit * (1.0 - 1e-4)))


def _make_initial_dicts(
    keyframe_info: Dict[str, Any]
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    tstamps = [int(t) for t in keyframe_info['keyframe_tstamps']]
    poses = keyframe_info['keyframe_poses_se3']
    disps = keyframe_info.get('keyframe_disps')
    initial_poses = {int(t): poses[i] for i, t in enumerate(tstamps)}
    initial_disps = (
        {int(t): disps[i] for i, t in enumerate(tstamps)} if disps is not None else None
    )
    return initial_poses, initial_disps


def _save_debug(path, debug):
    if path:
        from lib.pipeline.checkpoint import write_json
        write_json(path, debug)


def _guard_requested_delta(dT, dR_vec, translation_limit, rotation_limit_deg, optimize_rotation):
    if not np.isfinite(dT).all() or not np.isfinite(dR_vec).all():
        return {'reason': 'nonfinite_frontend_delta'}
    translation = float(np.linalg.norm(dT, axis=-1).max())
    rotation = float(np.degrees(np.linalg.norm(dR_vec, axis=-1)).max())
    rejected = translation > translation_limit or (optimize_rotation and rotation > rotation_limit_deg)
    return {'reason': 'requested_delta_exceeds_limits' if rejected else None,
            'max_guard_requested_translation_m': translation,
            'max_guard_requested_rotation_deg': rotation,
            'guard_translation_limit_m': translation_limit,
            'guard_rotation_limit_deg': rotation_limit_deg}


def refine_camera_with_frontend_delta(
    *,
    cam_R,
    cam_T,
    keyframe_info: Dict[str, Any],
    render_dir: str,
    original_image_dir: str,
    masks=None,
    intrinsics: Optional[Sequence[float]] = None,
    mapping_cam_R=None,
    mapping_cam_T=None,
    delta_coordinate_mode: str = 'baseline_world',
    use_rendered_for_keyframes: bool = True,
    keyframe_render_mode: str = 'adaptive',
    texture_threshold: float = 40.0,
    render_fusion_mode: str = 'replace',
    render_alpha: float = 1.0,
    max_translation_delta: float = 0.15,
    max_rotation_delta_deg: float = 3.0,
    fix_first_keyframe: bool = True,
    optimize_rotation: bool = True,
    min_common_keyframes: int = 4,
    interpolate_delta_to_all_frames: bool = True,
    droid_debug_filename: str = 'droid_frontend_delta.json',
    debug_path: Optional[str] = None,
    reject_on_excess_delta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Refine ``cam_R``/``cam_T`` (c2w) with a frontend-only pass on rendered keyframes.

    Never raises on SLAM failure: on any problem the baseline trajectory is
    returned unchanged together with a debug dict carrying ``enabled=False``
    and a ``reason``.

    Args:
        cam_R / cam_T: baseline metric c2w trajectory, [N, 3, 3] and [N, 3].
        keyframe_info: output of the Pass-1 warmstart SLAM, must contain
            ``keyframe_tstamps`` / ``keyframe_poses_se3`` and ideally
            ``keyframe_disps`` (used to warm-start the depth state).
        render_dir: directory of SMPL-rendered images, named by frame index.
        original_image_dir: directory of the untouched source images.
        masks: human masks used to suppress dynamic regions.
        intrinsics: ``[fx, fy, cx, cy]``.
        interpolate_delta_to_all_frames: if False, only keyframes are updated.
        droid_debug_filename: name of DROID's own frontend dump, written next to
            the rendered images under ``droid_debug/``.
        debug_path: where to write this function's delta summary.

    Returns:
        refined_R, refined_T, debug
    """
    base_R = torch.as_tensor(_to_numpy(cam_R)).float()
    base_T = torch.as_tensor(_to_numpy(cam_T)).float()
    num_frames = int(base_T.shape[0])
    if reject_on_excess_delta and (not math.isfinite(max_translation_delta) or max_translation_delta <= 0
            or not math.isfinite(max_rotation_delta_deg) or max_rotation_delta_deg <= 0):
        raise ValueError('Frontdelta safety limits must be finite and positive')

    debug: Dict[str, Any] = {
        'enabled': False,
        'use_rendered_for_keyframes': bool(use_rendered_for_keyframes),
        'keyframe_render_mode': keyframe_render_mode,
        'texture_threshold': float(texture_threshold),
        'render_fusion_mode': render_fusion_mode,
        'render_alpha': float(render_alpha),
        'max_translation_delta': float(max_translation_delta),
        'max_rotation_delta_deg': float(max_rotation_delta_deg),
        'num_frames': num_frames,
    }

    if keyframe_info is None or 'keyframe_poses_se3' not in keyframe_info:
        debug['reason'] = 'missing keyframe_info with keyframe_poses_se3'
        return base_R, base_T, debug

    if use_rendered_for_keyframes and not os.path.isdir(render_dir):
        debug['reason'] = f'render_dir does not exist: {render_dir}'
        return base_R, base_T, debug

    forced = [int(t) for t in keyframe_info['keyframe_tstamps']]
    if reject_on_excess_delta and len(forced) < min_common_keyframes:
        debug.update(reason='too_few_keyframes', accepted=False, fallback_to_baseline=True,
                     num_forced_keyframes=len(forced), reject_on_excess_delta=True)
        _save_debug(debug_path, debug)
        return base_R, base_T, debug
    initial_poses, initial_disps = _make_initial_dicts(keyframe_info)
    debug['num_forced_keyframes'] = len(forced)
    debug['has_initial_disps'] = initial_disps is not None
    if initial_disps is None:
        logger.warning(
            "[frontend_delta] keyframe_info has no keyframe_disps; the frontend will "
            "start from an uninitialized depth state, which usually degrades the delta."
        )

    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu()

    # Imported lazily: pulls in the DROID extension modules.
    from lib.camera.warmstart_droid import run_slam_warmstart

    droid = None
    try:
        droid, frontend_info = run_slam_warmstart(
            str(render_dir),
            forced_keyframes=forced,
            initial_poses_se3=initial_poses,
            initial_disps=initial_disps,
            masks=masks,
            calib=list(intrinsics) if intrinsics is not None else None,
            original_imagedir=str(original_image_dir),
            use_rendered_for_keyframes=use_rendered_for_keyframes,
            keyframe_render_mode=keyframe_render_mode,
            texture_threshold=texture_threshold,
            render_fusion_mode=render_fusion_mode,
            render_alpha=render_alpha,
            frontend_only=True,
            debug_filename=droid_debug_filename,
        )
    except Exception as exc:  # noqa: BLE001 - keep the pipeline alive
        logger.warning("[frontend_delta] frontend-only warmstart failed: %s", exc)
        debug['reason'] = f'frontend pass raised: {exc}'
        return base_R, base_T, debug
    finally:
        if droid is not None:
            del droid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if frontend_info is None:
        debug['reason'] = 'frontend-only warmstart returned no keyframe info'
        return base_R, base_T, debug

    new_tstamps = [int(t) for t in frontend_info['keyframe_tstamps']]
    new_by_t = {t: frontend_info['keyframe_poses_se3'][i] for i, t in enumerate(new_tstamps)}
    old_by_t = {t: initial_poses[t] for t in forced if t in initial_poses}
    common = sorted(t for t in new_by_t if t in old_by_t and 0 <= t < num_frames)

    debug['num_frontend_keyframes'] = len(new_tstamps)
    debug['num_common_keyframes'] = len(common)
    if len(common) < int(min_common_keyframes):
        debug['reason'] = (
            f'too few common keyframes: {len(common)} < {int(min_common_keyframes)}'
        )
        return base_R, base_T, debug

    old_raw_R, old_raw_t = _se3_raw_to_c2w(np.stack([old_by_t[t] for t in common], axis=0))
    new_raw_R, new_raw_t = _se3_raw_to_c2w(np.stack([new_by_t[t] for t in common], axis=0))
    if reject_on_excess_delta and not all(np.isfinite(v).all() for v in (old_raw_R, old_raw_t, new_raw_R, new_raw_t)):
        debug.update(enabled=False, accepted=False, reason='nonfinite_frontend_output',
                     fallback_to_baseline=True, reject_on_excess_delta=True)
        _save_debug(debug_path, debug)
        return base_R, base_T, debug
    finite = np.ones(len(common), dtype=bool)
    for v in (old_raw_R, old_raw_t, new_raw_R, new_raw_t):
        finite &= np.isfinite(v.reshape(len(common), -1)).all(axis=1)
    if not finite.all():  # a diverged frontend keyframe must not poison the whole correction
        debug['num_nonfinite_keyframes_dropped'] = int((~finite).sum())
        common = [t for t, keep in zip(common, finite) if keep]
        old_raw_R, old_raw_t, new_raw_R, new_raw_t = (v[finite] for v in (old_raw_R, old_raw_t, new_raw_R, new_raw_t))
        debug['num_common_keyframes'] = len(common)
        if len(common) < int(min_common_keyframes):
            debug.update(reason='too_few_finite_frontend_keyframes', fallback_to_baseline=True)
            _save_debug(debug_path, debug)
            return base_R, base_T, debug

    mapping_R_np = _to_numpy(mapping_cam_R if mapping_cam_R is not None else base_R).astype(np.float32)
    mapping_T_np = _to_numpy(mapping_cam_T if mapping_cam_T is not None else base_T).astype(np.float32)
    if mapping_R_np.shape != tuple(base_R.shape) or mapping_T_np.shape != tuple(base_T.shape):
        raise ValueError('Mapping camera must have the same frame count as the baseline')
    debug['separate_mapping_camera'] = mapping_cam_R is not None or mapping_cam_T is not None

    scale = _estimate_raw_to_metric_scale(old_raw_t, mapping_T_np[common])
    raw_dR = np.matmul(new_raw_R, np.swapaxes(old_raw_R, -1, -2))
    if delta_coordinate_mode == 'baseline_world':
        raw_to_world_R = _estimate_raw_to_world_rotation(old_raw_R, mapping_R_np[common])
        world_dR = np.matmul(np.matmul(raw_to_world_R, raw_dR), raw_to_world_R.T)
        dT = np.matmul(new_raw_t - old_raw_t, raw_to_world_R.T).astype(np.float32) * float(scale)
        debug['raw_to_world_rotation'] = raw_to_world_R.tolist()
    elif delta_coordinate_mode == 'legacy_scale_only':
        world_dR = raw_dR
        dT = (new_raw_t - old_raw_t).astype(np.float32) * float(scale)
    else:
        raise ValueError(f'Unknown delta coordinate mode: {delta_coordinate_mode}')
    if reject_on_excess_delta:
        from scipy.spatial.transform import Rotation
        dR_vec = Rotation.from_matrix(world_dR).as_rotvec().astype(np.float32)
    else:
        dR_vec = _rotmat_to_rotvec(world_dR)
    debug['delta_coordinate_frame'] = delta_coordinate_mode
    if reject_on_excess_delta:
        guard = _guard_requested_delta(dT, dR_vec, max_translation_delta, max_rotation_delta_deg, optimize_rotation)
        reason = guard.pop('reason')
        debug.update(guard, reject_on_excess_delta=True)
        if reason:
            debug.update(enabled=False, accepted=False, reason=reason, fallback_to_baseline=True)
            _save_debug(debug_path, debug)
            return base_R, base_T, debug

    if fix_first_keyframe:
        dR_vec[0] = 0.0
        dT[0] = 0.0

    max_rotation_delta_rad = math.radians(max_rotation_delta_deg)
    debug['raw_to_metric_scale'] = float(scale)
    debug['mean_requested_translation_delta'] = float(np.linalg.norm(dT, axis=-1).mean())
    debug['max_requested_translation_delta'] = float(np.linalg.norm(dT, axis=-1).max())
    debug['mean_requested_rotation_delta_deg'] = float(
        np.degrees(np.linalg.norm(dR_vec, axis=-1)).mean()
    )
    debug['max_requested_rotation_delta_deg'] = float(
        np.degrees(np.linalg.norm(dR_vec, axis=-1)).max()
    )
    # High saturation means the clamp is truncating what the frontend asked for,
    # so the applied correction is a direction rather than a magnitude.
    debug['translation_clamp_saturation'] = _saturation_ratio(dT, max_translation_delta)
    debug['rotation_clamp_saturation'] = _saturation_ratio(dR_vec, max_rotation_delta_rad)

    dT = _clamp_vectors(dT, max_translation_delta).astype(np.float32)
    dR_vec = _clamp_vectors(dR_vec, max_rotation_delta_rad).astype(np.float32)
    if not optimize_rotation:
        dR_vec = np.zeros_like(dR_vec)

    debug['mean_applied_translation_delta'] = float(np.linalg.norm(dT, axis=-1).mean())
    debug['max_applied_translation_delta'] = float(np.linalg.norm(dT, axis=-1).max())
    debug['mean_applied_rotation_delta_deg'] = float(
        np.degrees(np.linalg.norm(dR_vec, axis=-1)).mean()
    )
    debug['max_applied_rotation_delta_deg'] = float(
        np.degrees(np.linalg.norm(dR_vec, axis=-1)).max()
    )
    debug['common_keyframes'] = [int(t) for t in common]

    if interpolate_delta_to_all_frames:
        refined_R, refined_T = _interpolate_center_deltas_to_all_frames(
            base_R,
            base_T,
            common,
            torch.as_tensor(dR_vec).float(),
            torch.as_tensor(dT).float(),
            optimize_rotation=optimize_rotation,
        )
    else:
        refined_R = base_R.clone()
        refined_T = base_T.clone()
        idx = torch.as_tensor(common, dtype=torch.long)
        refined_T[idx] = base_T[idx] + torch.as_tensor(dT).float()
        if optimize_rotation:
            refined_R[idx] = torch.matmul(
                _so3_exp(torch.as_tensor(dR_vec).float()), base_R[idx]
            )

    if not (torch.isfinite(refined_R).all() and torch.isfinite(refined_T).all()):
        debug = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in debug.items()}
        debug.update(enabled=False, accepted=False, reason='nonfinite_interpolated_camera', fallback_to_baseline=True)
        _save_debug(debug_path, debug)
        return base_R, base_T, debug
    if reject_on_excess_delta:
        actual_dT = (refined_T - base_T).numpy()
        actual_dR = Rotation.from_matrix(torch.matmul(refined_R, base_R.transpose(-1, -2)).numpy()).as_rotvec()
        guard = _guard_requested_delta(actual_dT, actual_dR, max_translation_delta + 1e-6,
                                       max_rotation_delta_deg + 1e-4, optimize_rotation)
        if guard['reason']:
            debug.update(enabled=False, accepted=False, reason='applied_delta_exceeds_limits', fallback_to_baseline=True)
            _save_debug(debug_path, debug)
            return base_R, base_T, debug
        debug.update(accepted=True, fallback_to_baseline=False,
                     max_actual_translation_delta_m=guard['max_guard_requested_translation_m'],
                     max_actual_rotation_delta_deg=guard['max_guard_requested_rotation_deg'])
    debug['enabled'] = True
    debug['interpolated_to_all_frames'] = bool(interpolate_delta_to_all_frames)

    if debug_path:
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, 'w') as f:
                json.dump(debug, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[frontend_delta] failed to write debug json: %s", exc)

    return refined_R, refined_T, debug


def metric_scale_ratio(tstamps, scales, num_frames, median_window=15, max_ratio_deviation=0.2, strength=0.75):
    """Per-frame step-length correction from per-keyframe metric/DROID depth scales.

    The first pass converts DROID's gauge to metres with one median scale, so any local
    scale drift of DROID survives.  Each keyframe's own scale over that median is the
    local drift; it is median-filtered over keyframes, clipped, softened by ``strength``
    and interpolated to every frame.
    """
    from scipy.ndimage import median_filter

    tstamps = np.asarray(tstamps, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    valid = np.isfinite(scales) & (scales > 0)
    ratio = scales[valid] / np.median(scales[valid])
    if median_window > 1:
        ratio = median_filter(ratio, int(median_window), mode='nearest')
    ratio = np.clip(ratio, 1.0 - max_ratio_deviation, 1.0 + max_ratio_deviation) ** float(strength)
    return np.interp(np.arange(num_frames), tstamps[valid], ratio)


def rescale_camera_steps(cam_T, ratio):
    """Rescale every frame-to-frame translation step, keeping the first position fixed."""
    T = np.asarray(cam_T, dtype=np.float64)
    steps = np.diff(T, axis=0) * np.asarray(ratio, dtype=np.float64)[1:, None]
    return np.concatenate([T[:1], T[:1] + np.cumsum(steps, axis=0)])


def correct_local_metric_scale(cam_T, keyframe_info, *, enabled=True, median_window=15, max_ratio_deviation=0.2,
                               strength=0.75, min_keyframes=8):
    """Apply :func:`metric_scale_ratio` using the scales stored by the metric SLAM pass.

    Returns ``(T, debug)``; ``T`` is unchanged when disabled or when too few finite scales exist.
    """
    T = np.asarray(cam_T, dtype=np.float32)
    debug = {'enabled': False, 'median_window': int(median_window),
             'max_ratio_deviation': float(max_ratio_deviation), 'strength': float(strength)}
    scales = np.asarray((keyframe_info or {}).get('keyframe_metric_scales', []), dtype=np.float64)
    tstamps = np.asarray((keyframe_info or {}).get('keyframe_tstamps', []), dtype=np.float64)
    if not enabled:
        debug['reason'] = 'disabled'
        return T, debug
    if len(scales) != len(tstamps) or int((np.isfinite(scales) & (scales > 0)).sum()) < min_keyframes:
        debug['reason'] = 'too_few_finite_keyframe_scales'
        return T, debug
    ratio = metric_scale_ratio(tstamps, scales, len(T), median_window, max_ratio_deviation, strength)
    corrected = rescale_camera_steps(T, ratio).astype(np.float32)
    if not np.isfinite(corrected).all():
        debug['reason'] = 'nonfinite_rescaled_camera'
        return T, debug
    debug.update(enabled=True, ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                 max_position_change_m=float(np.linalg.norm(corrected - T, axis=-1).max()))
    return corrected, debug


def _umeyama(source, target):
    """Similarity (s, R, t) minimizing ||target - (s R source + t)||^2 for N x 3 point sets."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    x, y = source - mu_s, target - mu_t
    U, D, Vt = np.linalg.svd(y.T @ x / len(source))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    rotation = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / max(x.var(0).sum(), 1e-12))
    return scale, rotation, mu_t - scale * rotation @ mu_s


def pull_back_uniform_keyframes(*, cam_R, cam_T, forced_keyframes, original_image_dir, masks, intrinsics,
                                max_translation_delta, max_rotation_delta_deg, local_scale=None, debug_path=None):
    """Uniform-keyframe metric SLAM pass, pulled back onto the first-pass camera.

    DROID is rerun with keyframes forced on a uniform grid (original masked images, cold start).
    Its trajectory is not used directly: it is Sim3-aligned to the first pass and only the
    per-keyframe pose differences, clamped to the frontdelta limits and interpolated, are applied
    to the first-pass camera, so a diverged rerun cannot move the camera by more than the clamp.
    Also returns the grid keyframe state (poses/disparities), which warm-starts the rendered
    frontdelta on the same keyframes.  On failure the first-pass camera is returned with
    ``keyframe_info=None``.
    """
    from scipy.spatial.transform import Rotation

    base_R = torch.as_tensor(_to_numpy(cam_R)).float()
    base_T = torch.as_tensor(_to_numpy(cam_T)).float()
    debug = {'enabled': False, 'mode': 'uniform_keyframe_pullback', 'forced_keyframes': list(forced_keyframes),
             'alignment': 'sim3_to_first_pass', 'max_translation_delta': float(max_translation_delta),
             'max_rotation_delta_deg': float(max_rotation_delta_deg)}
    if len(forced_keyframes) < 8:
        debug['reason'] = 'too_few_uniform_keyframes'
        _save_debug(debug_path, debug)
        return base_R, base_T, None, debug
    try:
        from lib.camera.warmstart_droid import run_metric_slam_warmstart
        R, T, info = run_metric_slam_warmstart(
            str(original_image_dir), forced_keyframes=list(forced_keyframes), initial_poses_se3=None,
            initial_disps=None, masks=masks, calib=intrinsics, original_image_dir=str(original_image_dir),
            use_rendered_for_keyframes=False, keyframe_render_mode='all', return_keyframe_info=True)
        R, T = _to_numpy(R).astype(np.float64), _to_numpy(T).astype(np.float64)
        if R.shape != tuple(base_R.shape) or T.shape != tuple(base_T.shape):
            raise ValueError('Uniform-keyframe output timeline does not match the first pass')
        if not np.isfinite(R).all() or not np.isfinite(T).all():
            raise ValueError('Nonfinite uniform-keyframe camera')
        T, debug['local_metric_scale'] = correct_local_metric_scale(T, info, **(local_scale or {}))
        T = T.astype(np.float64)
        first_R, first_T = base_R.numpy().astype(np.float64), base_T.numpy().astype(np.float64)
        scale, rotation, shift = _umeyama(T, first_T)
        aligned_R, aligned_T = rotation @ R, scale * T @ rotation.T + shift
        keys = [int(t) for t in info['keyframe_tstamps']]
        dT = (aligned_T[keys] - first_T[keys]).astype(np.float32)
        dR = Rotation.from_matrix(aligned_R[keys] @ np.swapaxes(first_R[keys], -1, -2)).as_rotvec().astype(np.float32)
        dT[0] = dR[0] = 0.0
        debug.update(sim3_scale=scale,
                     mean_requested_translation_delta=float(np.linalg.norm(dT, axis=-1).mean()),
                     translation_clamp_saturation=_saturation_ratio(dT, max_translation_delta),
                     rotation_clamp_saturation=_saturation_ratio(dR, math.radians(max_rotation_delta_deg)))
        dT = _clamp_vectors(dT, max_translation_delta).astype(np.float32)
        dR = _clamp_vectors(dR, math.radians(max_rotation_delta_deg)).astype(np.float32)
        new_R, new_T = _interpolate_center_deltas_to_all_frames(base_R, base_T, keys, torch.as_tensor(dR),
                                                                torch.as_tensor(dT), optimize_rotation=True)
        if not (torch.isfinite(new_R).all() and torch.isfinite(new_T).all()):
            raise ValueError('Nonfinite pulled-back camera')
    except Exception as exc:  # noqa: BLE001 - keep the first-pass camera
        debug['reason'] = f'uniform_keyframe_pullback_failed: {exc}'
        _save_debug(debug_path, debug)
        return base_R, base_T, None, debug
    debug.update(enabled=True, num_keyframes=len(keys),
                 max_applied_change_m=float(torch.linalg.norm(new_T - base_T, dim=-1).max()))
    _save_debug(debug_path, debug)
    return new_R.float(), new_T.float(), info, debug

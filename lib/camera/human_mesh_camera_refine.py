"""Window-local human mesh consistency based camera refinement."""

import json
import math
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    K = _skew(omega / theta)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    return I + s * K + (1.0 - c) * (K @ K)


def _as_torch(x, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _to_list_int(x: Optional[Sequence[int]]) -> Optional[List[int]]:
    if x is None:
        return None
    return [int(v) for v in x]


def _robust_l2(diff: torch.Tensor, huber_delta: float) -> torch.Tensor:
    r = torch.linalg.norm(diff, dim=-1)
    if huber_delta <= 0:
        return r.pow(2)
    d = float(huber_delta)
    return torch.where(r <= d, 0.5 * r.pow(2), d * (r - 0.5 * d))


def _rotation_angle(R_delta: torch.Tensor) -> torch.Tensor:
    tr = R_delta.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos)


def _interpolate_center_deltas_to_all_frames(
    init_R: torch.Tensor,
    init_T: torch.Tensor,
    centers: Sequence[int],
    dR_eff: torch.Tensor,
    dT_eff: torch.Tensor,
    *,
    optimize_rotation: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linearly interpolate optimized center-frame SE(3) deltas to all frames.

    Translation deltas and rotation-vector deltas are interpolated independently,
    matching the offline sweep used for ATE/ATE-S experiments.
    """
    device = init_T.device
    dtype = init_T.dtype
    num_frames = int(init_T.shape[0])
    centers_np = np.asarray([int(c) for c in centers], dtype=np.float32)
    xs = np.arange(num_frames, dtype=np.float32)

    dT_np = dT_eff.detach().cpu().numpy().astype(np.float32)
    dR_np = (
        dR_eff.detach().cpu().numpy().astype(np.float32)
        if optimize_rotation
        else np.zeros_like(dT_np, dtype=np.float32)
    )
    dT_all_np = np.stack([np.interp(xs, centers_np, dT_np[:, k]) for k in range(3)], axis=-1).astype(np.float32)
    dR_all_np = np.stack([np.interp(xs, centers_np, dR_np[:, k]) for k in range(3)], axis=-1).astype(np.float32)

    dT_all = torch.as_tensor(dT_all_np, device=device, dtype=dtype)
    dR_all = torch.as_tensor(dR_all_np, device=device, dtype=dtype)
    R_delta_all = _so3_exp(dR_all) if optimize_rotation else torch.eye(3, device=device, dtype=dtype).expand(num_frames, 3, 3)
    return torch.matmul(R_delta_all, init_R), init_T + dT_all


def _pack_observations(
    window_observations: Sequence[Dict[str, Any]],
    *,
    debug_window_centers: Optional[Sequence[int]] = None,
    max_debug_windows: int = 0,
    min_shared_frames: int = 3,
    max_points_per_frame: int = 22,
    max_pair_factors: int = 0,
    device: str = "cuda",
) -> Tuple[List[int], Dict[int, Dict[str, torch.Tensor]], List[Tuple[int, int, int]], Dict[str, Any]]:
    """Select windows and build shared-frame pair factors.

    Returns:
        selected_centers: selected window centers.
        obs_by_center: center -> {frames, points}; points are center-view local.
        pair_factors: list of (frame_idx, center_a, center_b).
        info: debug summary.
    """
    valid = []
    for obs in window_observations:
        if not isinstance(obs, dict):
            continue
        if 'center' not in obs or 'frame_indices' not in obs:
            continue
        pts = obs.get('joints_center_cam')
        if pts is None:
            continue
        valid.append(obs)

    if debug_window_centers is not None:
        wanted = set(_to_list_int(debug_window_centers) or [])
        selected = [o for o in valid if int(o['center']) in wanted]
    else:
        limit = int(max_debug_windows) if max_debug_windows is not None else 0
        selected = valid if limit <= 0 else valid[:limit]

    selected_centers = [int(o['center']) for o in selected]
    obs_by_center: Dict[int, Dict[str, torch.Tensor]] = {}
    for obs in selected:
        c = int(obs['center'])
        frames_raw = obs['frame_indices']
        frames = torch.as_tensor(frames_raw, dtype=torch.long, device=device)
        pts = _as_torch(obs['joints_center_cam'], device)
        if pts.ndim != 3:
            continue
        if max_points_per_frame > 0 and pts.shape[1] > max_points_per_frame:
            idx = torch.linspace(0, pts.shape[1] - 1, max_points_per_frame, device=device).long()
            pts = pts[:, idx]
        n = min(frames.shape[0], pts.shape[0])
        if n <= 0:
            continue
        obs_by_center[c] = {
            'frames': frames[:n],
            'points': pts[:n],
        }

    frame_to_centers: Dict[int, List[int]] = {}
    for c, pack in obs_by_center.items():
        for f in pack['frames'].detach().cpu().tolist():
            frame_to_centers.setdefault(int(f), []).append(c)

    if max_pair_factors < 0:
        raise ValueError('max_pair_factors must be nonnegative')
    frame_centers = [(int(f), sorted(set(cs))) for f, cs in sorted(frame_to_centers.items())]
    total_pair_factors = sum(len(cs) * (len(cs) - 1) // 2 for _, cs in frame_centers)
    sample_indices = None
    if max_pair_factors > 0 and total_pair_factors > max_pair_factors:
        sample_indices = set(np.linspace(0, total_pair_factors - 1, max_pair_factors, dtype=np.int64).tolist())
    pair_factors: List[Tuple[int, int, int]] = []
    factor_index = 0
    for f, centers in frame_centers:
        for a, b in combinations(centers, 2):
            if sample_indices is None or factor_index in sample_indices:
                pair_factors.append((f, int(a), int(b)))
            factor_index += 1

    # If the first few windows overlap too little, keep the selected centers but
    # only report failure; the caller can change centers/stride without crashing.
    info = {
        'num_input_observations': len(window_observations),
        'num_valid_observations': len(valid),
        'selected_centers': selected_centers,
        'num_selected_observations': len(obs_by_center),
        'num_pair_factors': len(pair_factors),
        'num_pair_factors_before_sampling': total_pair_factors,
        'max_pair_factors': int(max_pair_factors),
        'min_shared_frames': int(min_shared_frames),
    }
    if len(pair_factors) < int(min_shared_frames):
        pair_factors = []
    return selected_centers, obs_by_center, pair_factors, info


def refine_camera_with_window_mesh_consistency(
    cam_R,
    cam_T,
    window_observations: Sequence[Dict[str, Any]],
    *,
    debug_window_centers: Optional[Sequence[int]] = None,
    max_debug_windows: int = 0,
    min_shared_frames: int = 3,
    max_points_per_frame: int = 22,
    max_pair_factors: int = 0,
    mesh_loss_weight: float = 1.0,
    pose_prior_weight: float = 0.5,
    smooth_weight: float = 0.01,
    relative_pose_weight: float = 0.0,
    relative_translation_sigma: float = 0.05,
    relative_rotation_sigma_deg: float = 1.0,
    accept_only_if_reduces_total: bool = False,
    huber_delta: float = 0.05,
    num_iters: int = 500,
    lr: float = 0.005,
    optimize_rotation: bool = True,
    fix_first_window: bool = True,
    max_translation_delta: float = 0.15,
    max_rotation_delta_deg: float = 3.0,
    accept_only_if_reduces_loss: bool = True,
    interpolate_delta_to_all_frames: bool = False,
    device: str = "cuda",
    debug_path: Optional[str] = None,
):
    """Refine selected c2w center poses using overlapping OAR window mesh observations.

    Each OAR observation stores points in its own window-center camera frame.
    A candidate center camera pose maps those points to world space.  For frames
    shared by two windows, their transformed points should coincide.
    """
    info: Dict[str, Any] = {'enabled': True}

    def _write_debug():
        if not debug_path:
            return
        try:
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception:
            pass

    if not window_observations:
        info.update({'enabled': False, 'reason': 'missing oar_window_mesh_observations'})
        _write_debug()
        return cam_R, cam_T, info

    cam_R_t = _as_torch(cam_R, device).clone()
    cam_T_t = _as_torch(cam_T, device).clone()
    init_R = cam_R_t.clone()
    init_T = cam_T_t.clone()

    selected_centers, obs_by_center, pair_factors, pack_info = _pack_observations(
        window_observations,
        debug_window_centers=debug_window_centers,
        max_debug_windows=max_debug_windows,
        min_shared_frames=min_shared_frames,
        max_points_per_frame=max_points_per_frame,
        max_pair_factors=max_pair_factors,
        device=device,
    )
    info.update(pack_info)
    centers = [c for c in selected_centers if c in obs_by_center and 0 <= c < cam_T_t.shape[0]]
    centers = sorted(set(centers))
    if len(centers) < 2 or not pair_factors:
        info.update({'enabled': False, 'reason': 'not enough overlapping selected window observations'})
        _write_debug()
        return cam_R, cam_T, info

    center_to_param = {c: i for i, c in enumerate(centers)}
    pair_factors = [(f, a, b) for (f, a, b) in pair_factors if a in center_to_param and b in center_to_param]
    if not pair_factors:
        info.update({'enabled': False, 'reason': 'no valid pair factors after camera range filtering'})
        _write_debug()
        return cam_R, cam_T, info

    frame_pos = {}
    for c, pack in obs_by_center.items():
        frame_pos[c] = {int(f): i for i, f in enumerate(pack['frames'].detach().cpu().tolist())}

    factor_a_idx = []
    factor_b_idx = []
    factor_pts_a = []
    factor_pts_b = []
    for f, a, b in pair_factors:
        if f not in frame_pos[a] or f not in frame_pos[b]:
            continue
        pa = obs_by_center[a]['points'][frame_pos[a][f]]
        pb = obs_by_center[b]['points'][frame_pos[b][f]]
        n = min(pa.shape[0], pb.shape[0])
        if n <= 0:
            continue
        factor_a_idx.append(center_to_param[a])
        factor_b_idx.append(center_to_param[b])
        factor_pts_a.append(pa[:n])
        factor_pts_b.append(pb[:n])
    if not factor_pts_a:
        info.update({'enabled': False, 'reason': 'no valid vectorized pair factors'})
        _write_debug()
        return cam_R, cam_T, info
    factor_a_idx_t = torch.as_tensor(factor_a_idx, dtype=torch.long, device=device)
    factor_b_idx_t = torch.as_tensor(factor_b_idx, dtype=torch.long, device=device)
    factor_pts_a_t = torch.stack(factor_pts_a, dim=0).to(device=device, dtype=torch.float32)
    factor_pts_b_t = torch.stack(factor_pts_b, dim=0).to(device=device, dtype=torch.float32)
    center_index_t = torch.as_tensor(centers, dtype=torch.long, device=device)
    init_R_centers = init_R[center_index_t]
    init_T_centers = init_T[center_index_t]
    if relative_pose_weight < 0 or relative_translation_sigma <= 0 or relative_rotation_sigma_deg <= 0:
        raise ValueError('Relative pose weight must be nonnegative and sigmas must be positive')
    init_relative_R = init_R_centers[:-1].transpose(-1, -2) @ init_R_centers[1:]
    init_relative_T = torch.einsum(
        'nij,nj->ni', init_R_centers[:-1].transpose(-1, -2),
        init_T_centers[1:] - init_T_centers[:-1],
    )
    relative_rotation_sigma = math.radians(float(relative_rotation_sigma_deg))

    K = len(centers)
    dT = torch.zeros(K, 3, device=device, requires_grad=True)
    dR = torch.zeros(K, 3, device=device, requires_grad=bool(optimize_rotation))
    params = [dT] + ([dR] if optimize_rotation else [])
    optimizer = torch.optim.Adam(params, lr=float(lr))
    max_rot = math.radians(float(max_rotation_delta_deg))

    def _effective_params():
        dT_eff = dT
        dR_eff = dR if optimize_rotation else torch.zeros_like(dT)
        if fix_first_window:
            dT_eff = dT_eff.clone()
            dR_eff = dR_eff.clone()
            dT_eff[0] = 0.0
            dR_eff[0] = 0.0
        return dR_eff, dT_eff

    def _loss_parts():
        dR_eff, dT_eff = _effective_params()
        if optimize_rotation:
            R_delta = _so3_exp(dR_eff)
        else:
            R_delta = torch.eye(3, device=device, dtype=init_R_centers.dtype).expand(K, 3, 3)
        R_eff_all = torch.matmul(R_delta, init_R_centers)
        T_eff_all = init_T_centers + dT_eff

        Ra = R_eff_all[factor_a_idx_t]
        Rb = R_eff_all[factor_b_idx_t]
        Ta = T_eff_all[factor_a_idx_t]
        Tb = T_eff_all[factor_b_idx_t]
        pa = torch.einsum('nij,npj->npi', Ra, factor_pts_a_t) + Ta[:, None, :]
        pb = torch.einsum('nij,npj->npi', Rb, factor_pts_b_t) + Tb[:, None, :]
        diff = pa - pb
        mesh_loss = _robust_l2(diff, float(huber_delta)).mean()
        mesh_dist = torch.linalg.norm(diff, dim=-1).mean()

        prior = dT_eff.pow(2).sum(dim=-1).mean()
        if optimize_rotation:
            prior = prior + dR_eff.pow(2).sum(dim=-1).mean()
        smooth = torch.tensor(0.0, device=device)
        if smooth_weight > 0 and K >= 3:
            smooth = (dT_eff[2:] + dT_eff[:-2] - 2.0 * dT_eff[1:-1]).pow(2).sum(dim=-1).mean()
        relative_loss = torch.tensor(0.0, device=device)
        if relative_pose_weight > 0:
            relative_R = R_eff_all[:-1].transpose(-1, -2) @ R_eff_all[1:]
            relative_T = torch.einsum(
                'nij,nj->ni', R_eff_all[:-1].transpose(-1, -2),
                T_eff_all[1:] - T_eff_all[:-1],
            )
            relative_loss = ((relative_T - init_relative_T) / float(relative_translation_sigma)).pow(2).sum(-1).mean()
            relative_loss = relative_loss + (
                (relative_R - init_relative_R).pow(2).sum(dim=(-1, -2)).mean()
                / (2.0 * relative_rotation_sigma ** 2)
            )
        total = (float(mesh_loss_weight) * mesh_loss + float(pose_prior_weight) * prior
                 + float(smooth_weight) * smooth + float(relative_pose_weight) * relative_loss)
        return total, mesh_loss, mesh_dist, prior, smooth, relative_loss

    with torch.no_grad():
        init_total, init_mesh_objective, init_mesh_dist, _, _, _ = _loss_parts()
        init_total_value = float(init_total.detach().cpu())
        init_mesh_loss_value = float(init_mesh_dist.detach().cpu())
        init_mesh_objective_value = float(init_mesh_objective.detach().cpu())

    iter_losses = []
    for it in range(int(num_iters)):
        optimizer.zero_grad()
        total, mesh_loss, mesh_dist, prior, smooth, relative_loss = _loss_parts()
        total.backward()
        optimizer.step()
        with torch.no_grad():
            if max_translation_delta > 0:
                norms = torch.linalg.norm(dT, dim=-1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(float(max_translation_delta) / norms, max=1.0)
                dT.mul_(scale)
            if optimize_rotation and max_rot > 0:
                norms = torch.linalg.norm(dR, dim=-1, keepdim=True).clamp(min=1e-12)
                scale = torch.clamp(max_rot / norms, max=1.0)
                dR.mul_(scale)
        if it == 0 or it == int(num_iters) - 1 or (it + 1) % max(1, int(num_iters) // 5) == 0:
            iter_losses.append({
                'iter': int(it),
                'total': float(total.detach().cpu()),
                'mesh_objective': float(mesh_loss.detach().cpu()),
                'mesh_dist_m': float(mesh_dist.detach().cpu()),
                'prior': float(prior.detach().cpu()),
                'smooth': float(smooth.detach().cpu()),
            })

    with torch.no_grad():
        final_total, final_mesh_objective, final_mesh_dist, final_prior, final_smooth, final_relative = _loss_parts()
        final_total_value = float(final_total.detach().cpu())
        final_mesh_loss_value = float(final_mesh_dist.detach().cpu())
        final_mesh_objective_value = float(final_mesh_objective.detach().cpu())
        dR_eff, dT_eff = _effective_params()

        center_refined_R = cam_R_t.clone()
        center_refined_T = cam_T_t.clone()
        for c in centers:
            pi = center_to_param[c]
            if optimize_rotation:
                center_refined_R[c] = _so3_exp(dR_eff[pi]) @ init_R[c]
            center_refined_T[c] = init_T[c] + dT_eff[pi]

        trans_delta = torch.linalg.norm(center_refined_T[centers] - init_T[centers], dim=-1)
        rot_delta = _rotation_angle(center_refined_R[centers] @ init_R[centers].transpose(-1, -2))

    accepted = True
    applied_interpolation = False
    reject_mesh = accept_only_if_reduces_loss and not (final_mesh_loss_value < init_mesh_loss_value)
    reject_total = accept_only_if_reduces_total and not (
        math.isfinite(final_total_value) and final_total_value < init_total_value
    )
    if reject_mesh or reject_total:
        refined_R = cam_R_t
        refined_T = cam_T_t
        accepted = False
    elif interpolate_delta_to_all_frames:
        with torch.no_grad():
            refined_R, refined_T = _interpolate_center_deltas_to_all_frames(
                init_R,
                init_T,
                centers,
                dR_eff,
                dT_eff,
                optimize_rotation=bool(optimize_rotation),
            )
        applied_interpolation = True
    else:
        refined_R = center_refined_R
        refined_T = center_refined_T

    info.update({
        'enabled': True,
        'selected_centers': centers,
        'num_pair_factors': len(pair_factors),
        'init_mesh_loss_m': init_mesh_loss_value,
        'final_mesh_loss_m': final_mesh_loss_value,
        'loss_delta_m': final_mesh_loss_value - init_mesh_loss_value,
        'init_mesh_objective': init_mesh_objective_value,
        'final_mesh_objective': final_mesh_objective_value,
        'accepted': bool(accepted),
        'accept_only_if_reduces_loss': bool(accept_only_if_reduces_loss),
        'interpolate_delta_to_all_frames': bool(interpolate_delta_to_all_frames),
        'applied_interpolation': bool(applied_interpolation),
        'mesh_loss_weight': float(mesh_loss_weight),
        'pose_prior_weight': float(pose_prior_weight),
        'smooth_weight': float(smooth_weight),
        'relative_pose_weight': float(relative_pose_weight),
        'relative_translation_sigma': float(relative_translation_sigma),
        'relative_rotation_sigma_deg': float(relative_rotation_sigma_deg),
        'accept_only_if_reduces_total': bool(accept_only_if_reduces_total),
        'init_total_objective': init_total_value,
        'final_total_objective': final_total_value,
        'final_relative_pose_loss': float(final_relative.detach().cpu()),
        'huber_delta': float(huber_delta),
        'num_iters': int(num_iters),
        'lr': float(lr),
        'optimize_rotation': bool(optimize_rotation),
        'fix_first_window': bool(fix_first_window),
        'max_translation_delta': float(max_translation_delta),
        'max_rotation_delta_deg': float(max_rotation_delta_deg),
        'translation_delta_by_center_m': {
            str(c): float(trans_delta[i].detach().cpu()) for i, c in enumerate(centers)
        },
        'rotation_delta_by_center_deg': {
            str(c): float(rot_delta[i].detach().cpu() * 180.0 / math.pi) for i, c in enumerate(centers)
        },
        'iter_losses': iter_losses,
        'final_prior': float(final_prior.detach().cpu()),
        'final_smooth': float(final_smooth.detach().cpu()),
    })
    if not accepted:
        info['reason'] = (
            'rejected because total objective did not decrease' if reject_total
            else 'rejected because mesh loss did not decrease'
        )
    _write_debug()

    if isinstance(cam_R, torch.Tensor):
        return refined_R.to(device=cam_R.device, dtype=cam_R.dtype), refined_T.to(device=cam_T.device, dtype=cam_T.dtype), info
    return refined_R.detach().cpu().numpy(), refined_T.detach().cpu().numpy(), info

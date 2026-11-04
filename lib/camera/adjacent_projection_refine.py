"""Adjacent SMPL projection based camera refinement.

This module implements a light-weight post-SLAM pose-only refinement based on
exactly the same local adjacent-SMPL projections used by
``AdjacentSMPLRenderer`` in ``render_mode='camera'``.

For every observed pair (center frame c, source SMPL frame s), the renderer
projects source-frame SMPL points into center view using the *reference* camera
trajectory that produced the adjacent render images:

    X_w = R_ref_s @ X_cam_s + T_ref_s
    u_obs = pi(K, R_ref_c^T (X_w - T_ref_c))

After rerunning SLAM, we refine the new camera pose at center frame c so that
it reprojects the same simulated local SMPL sequence to the same pixels:

    u_pred = pi(K, R_new_c^T (X_w - T_new_c))
    residual = u_pred - u_obs

The source points ``X_w`` and observations ``u_obs`` are generated from the old
camera trajectory, so the refinement intentionally pulls the new trajectory
towards the old one in the local SMPL-projection sense. This is useful before a
true sliding-removal/global SMPL trajectory is available.
"""

import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from .smpl_pose_refine import _project, _project_and_jacobian, _huber_weight, _so3_exp


def _as_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _umeyama_sim3(src: torch.Tensor, tgt: torch.Tensor):
    """Solve tgt ~= scale * R @ src + t for row-wise point sets."""
    src_mean = src.mean(dim=0)
    tgt_mean = tgt.mean(dim=0)
    src_c = src - src_mean[None, :]
    tgt_c = tgt - tgt_mean[None, :]
    var_src = (src_c ** 2).sum() / max(src.shape[0], 1)
    H = (tgt_c.T @ src_c) / max(src.shape[0], 1)
    U, S, Vh = torch.linalg.svd(H)
    D = torch.eye(3, device=src.device, dtype=src.dtype)
    if torch.det(U @ Vh) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vh
    scale = torch.trace(torch.diag(S) @ D) / var_src.clamp(min=1e-12)
    t = tgt_mean - scale * (R @ src_mean)
    return scale, R, t


def _build_observations(
    projection_info: Dict,
    ref_R: torch.Tensor,
    ref_T: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: str,
    sim3_ref_to_cam=None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Group adjacent projection factors by center frame.

    Returns a mapping:
        center_idx -> {"points_world": (M,3), "points2d": (M,2)}
    """
    source_indices = projection_info.get("source_indices")
    center_indices = projection_info.get("center_indices")
    source_points_cam = projection_info.get("source_points_cam")
    valid = projection_info.get("valid")

    if source_indices is None or center_indices is None or source_points_cam is None:
        raise ValueError("projection_info is missing required keys")

    source_indices = np.asarray(source_indices, dtype=np.int64)
    center_indices = np.asarray(center_indices, dtype=np.int64)
    source_points_cam = _as_torch(source_points_cam, device)  # (S, P, 3)
    if valid is not None:
        valid = torch.as_tensor(valid, dtype=torch.bool, device=device)
    else:
        valid = torch.ones(
            (len(source_indices), source_points_cam.shape[1]),
            dtype=torch.bool,
            device=device,
        )

    groups = {}
    for obs_i, (c, s) in enumerate(zip(center_indices.tolist(), source_indices.tolist())):
        pts_cam_s = source_points_cam[obs_i]  # (P, 3)
        pts_valid = valid[obs_i]
        if pts_valid.sum() < 1:
            continue

        R_s = ref_R[s]
        T_s = ref_T[s]
        R_c = ref_R[c]
        T_c = ref_T[c]

        # source camera -> reference world
        pts_w = torch.einsum("ij,pj->pi", R_s, pts_cam_s) + T_s[None, :]

        # reference world -> center camera -> obs 2D
        Xc = (pts_w - T_c[None, :]) @ R_c
        z = Xc[:, 2]
        front = z > 1e-3
        pts_valid = pts_valid & front
        if pts_valid.sum() < 1:
            continue

        Xc = Xc[pts_valid]
        pts_w = pts_w[pts_valid]
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        pts2d = torch.stack([u, v], dim=-1)

        # Optional Sim3 maps points from the reference-camera world into the
        # current warmstart-camera world. Observations remain generated in the
        # reference views; only 3D landmark coordinates are re-expressed.
        if sim3_ref_to_cam is not None:
            sim_s, sim_R, sim_t = sim3_ref_to_cam
            pts_w = sim_s * (pts_w @ sim_R.T) + sim_t[None, :]

        # Guard against huge out-of-image projections; keep a generous margin.
        finite = torch.isfinite(pts2d).all(dim=-1)
        if finite.sum() < 1:
            continue
        pts_w = pts_w[finite]
        pts2d = pts2d[finite]

        if c not in groups:
            groups[c] = {"points_world": [], "points2d": []}
        groups[c]["points_world"].append(pts_w)
        groups[c]["points2d"].append(pts2d)

    packed = {}
    for c, g in groups.items():
        packed[c] = {
            "points_world": torch.cat(g["points_world"], dim=0),
            "points2d": torch.cat(g["points2d"], dim=0),
        }
    return packed


def refine_camera_with_adjacent_smpl_projection(
    cam_R,
    cam_T,
    ref_R,
    ref_T,
    projection_info: Dict,
    intrinsics: Tuple[float, float, float, float],
    *,
    weight: float = 1.0,
    pose_prior_weight: float = 1.0,
    smooth_weight: float = 0.0,
    huber_delta: float = 8.0,
    num_iters: int = 5,
    damping: float = 1e-3,
    max_points_per_frame: int = 256,
    init_rmse_gate_px: float = 300.0,
    update_clip: float = 0.2,
    max_translation_delta: float = 0.15,
    accept_only_if_reduces_residual: bool = True,
    align_reference_to_source: bool = True,
    device: str = "cuda",
    debug_path: Optional[str] = None,
):
    """Refine current camera poses using adjacent-SMPL local projection factors.

    Only center frames present in ``projection_info`` are optimized. Non-center
    frames remain untouched.
    """
    info = {"enabled": True}

    def _write_debug():
        if debug_path:
            try:
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                with open(debug_path, "w") as f:
                    json.dump(info, f, indent=2)
            except Exception:
                pass

    if projection_info is None or not projection_info:
        info.update({"enabled": False, "reason": "missing projection_info"})
        _write_debug()
        return cam_R, cam_T, info
    if weight <= 0:
        info.update({"enabled": False, "reason": "weight <= 0"})
        _write_debug()
        return cam_R, cam_T, info

    cam_R_t = _as_torch(cam_R, device).clone()
    cam_T_t = _as_torch(cam_T, device).clone()
    init_R_t = cam_R_t.clone()
    init_T_t = cam_T_t.clone()
    ref_R_t = _as_torch(ref_R, device)
    ref_T_t = _as_torch(ref_T, device)
    fx, fy, cx, cy = [float(v) for v in intrinsics]

    sim3_ref_to_cam = None
    if align_reference_to_source:
        n = min(ref_T_t.shape[0], cam_T_t.shape[0])
        if n >= 3:
            sim_s, sim_R, sim_t = _umeyama_sim3(ref_T_t[:n], cam_T_t[:n])
            sim3_ref_to_cam = (sim_s, sim_R, sim_t)
            info["sim3_ref_to_source"] = {
                "scale": float(sim_s.detach().cpu()),
                "translation": sim_t.detach().cpu().tolist(),
            }

    obs_by_center = _build_observations(
        projection_info, ref_R_t, ref_T_t, fx, fy, cx, cy, device,
        sim3_ref_to_cam=sim3_ref_to_cam,
    )
    centers = sorted(obs_by_center.keys())
    info["num_center_frames"] = len(centers)
    info["num_observations"] = int(sum(v["points2d"].shape[0] for v in obs_by_center.values()))
    if not centers:
        info.update({"enabled": False, "reason": "no valid adjacent projection observations"})
        _write_debug()
        return cam_R, cam_T, info

    # Initial residual gate.
    with torch.no_grad():
        sq, cnt, negz = 0.0, 0, 0
        for c in centers:
            pts_w = obs_by_center[c]["points_world"]
            obs = obs_by_center[c]["points2d"]
            Xc = (pts_w - cam_T_t[c][None, :]) @ cam_R_t[c]
            z = Xc[:, 2]
            valid = z > 1e-3
            negz += int((~valid).sum().item())
            if valid.sum() < 1:
                continue
            Xc = Xc[valid]
            obs = obs[valid]
            u = fx * Xc[:, 0] / Xc[:, 2] + cx
            v = fy * Xc[:, 1] / Xc[:, 2] + cy
            r = torch.stack([u, v], dim=-1) - obs
            sq += float((r ** 2).sum().item())
            cnt += int(r.numel())
        init_rmse = (sq / max(cnt, 1)) ** 0.5
    info["init_rmse_pixels"] = init_rmse
    info["num_negative_depth_points"] = negz
    if init_rmse > init_rmse_gate_px:
        info.update({
            "enabled": False,
            "reason": f"init_rmse={init_rmse:.1f}px exceeds gate={init_rmse_gate_px}px",
        })
        _write_debug()
        return cam_R, cam_T, info

    iter_rmse = []
    total_rejected = 0
    total_clipped = 0
    max_total_translation_delta_seen = 0.0
    for _ in range(num_iters):
        sq, cnt = 0.0, 0
        for c in centers:
            pts_w = obs_by_center[c]["points_world"]
            obs = obs_by_center[c]["points2d"]
            if max_points_per_frame > 0 and pts_w.shape[0] > max_points_per_frame:
                # deterministic uniform subsample
                idx = torch.linspace(
                    0, pts_w.shape[0] - 1,
                    steps=max_points_per_frame,
                    device=device,
                ).long()
                pts_w = pts_w[idx]
                obs = obs[idx]

            proj, J3 = _project_and_jacobian(
                cam_R_t[c], cam_T_t[c], pts_w, fx, fy, cx, cy
            )
            r2d = proj - obs
            r = r2d.reshape(-1)
            sq += float((r ** 2).sum().item())
            cnt += int(r.numel())

            r2 = (r2d ** 2).sum(dim=-1)
            w = _huber_weight(r2, huber_delta).repeat_interleave(2) * float(weight)
            J = J3.reshape(-1, 6)
            JW = J * w[:, None]
            A = JW.T @ J
            b = JW.T @ r

            if pose_prior_weight > 0:
                # translation prior around the initial warmstart pose. Rotation
                # prior is approximated as small-angle zero perturbation.
                prior = torch.zeros(6, device=device)
                prior[3:] = cam_T_t[c] - init_T_t[c]
                A = A + float(pose_prior_weight) * torch.eye(6, device=device)
                b = b + float(pose_prior_weight) * prior

            if smooth_weight > 0 and 0 < c < cam_T_t.shape[0] - 1:
                smooth_r = 2 * cam_T_t[c] - cam_T_t[c - 1] - cam_T_t[c + 1]
                A[3:, 3:] = A[3:, 3:] + float(smooth_weight) * torch.eye(3, device=device)
                b[3:] = b[3:] + float(smooth_weight) * smooth_r

            A = A + float(damping) * torch.eye(6, device=device)
            try:
                delta = torch.linalg.solve(A, -b)
            except RuntimeError:
                continue
            if update_clip > 0:
                nrm = torch.linalg.norm(delta)
                if nrm > update_clip:
                    delta = delta * (update_clip / nrm)
            cand_R = _so3_exp(delta[:3]) @ cam_R_t[c]
            cand_T = cam_T_t[c] + delta[3:]

            # Total trust region around the original warmstart pose. Without
            # this, a center keyframe can accumulate update_clip over multiple
            # iterations (e.g. 5 * 0.2m) and produce single-frame camera spikes.
            if max_translation_delta > 0:
                total_dT = cand_T - init_T_t[c]
                total_norm = torch.linalg.norm(total_dT)
                max_total_translation_delta_seen = max(
                    max_total_translation_delta_seen,
                    float(total_norm.detach().cpu()),
                )
                if total_norm > max_translation_delta:
                    cand_T = init_T_t[c] + total_dT * (max_translation_delta / total_norm)
                    total_clipped += 1

            if accept_only_if_reduces_residual:
                cand_proj = _project(cand_R, cand_T, pts_w, fx, fy, cx, cy)
                old_rmse = torch.sqrt((r2d ** 2).mean())
                new_rmse = torch.sqrt(((cand_proj - obs) ** 2).mean())
                if not torch.isfinite(new_rmse) or new_rmse > old_rmse:
                    total_rejected += 1
                    continue

            cam_R_t[c] = cand_R
            cam_T_t[c] = cand_T

        iter_rmse.append((sq / max(cnt, 1)) ** 0.5)

    info["iter_rmse_pixels"] = iter_rmse
    info["weight"] = float(weight)
    info["pose_prior_weight"] = float(pose_prior_weight)
    info["smooth_weight"] = float(smooth_weight)
    info["huber_delta"] = float(huber_delta)
    info["num_iters"] = int(num_iters)
    info["total_rejected_updates"] = int(total_rejected)
    info["total_clipped_updates"] = int(total_clipped)
    info["max_total_translation_delta_seen"] = float(max_total_translation_delta_seen)
    info["max_translation_delta"] = float(max_translation_delta)
    info["accept_only_if_reduces_residual"] = bool(accept_only_if_reduces_residual)

    _write_debug()

    if isinstance(cam_R, torch.Tensor):
        out_R = cam_R_t.to(device=cam_R.device, dtype=cam_R.dtype)
        out_T = cam_T_t.to(device=cam_T.device, dtype=cam_T.dtype)
    else:
        out_R = cam_R_t.cpu()
        out_T = cam_T_t.cpu()
    return out_R, out_T, info

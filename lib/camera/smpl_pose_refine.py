"""
SMPL Joint Reprojection Pose-only Refinement
============================================

Refine warmstart SLAM keyframe camera poses using SMPL joint reprojection
residuals (Levenberg–Marquardt, per-keyframe pose-only).

Motivation
----------
In the `droid_warmstart` pipeline, keyframes are forced and frontend keyframe
removal is disabled. The original DROID-SLAM dense BA only sees pixel-level
correspondences from the (possibly rendered-replaced) keyframe images. When
the rendered SMPL mesh is used to replace the original image (e.g. on
low-texture frames), there is no human-aware loss term inside DROID's BA.

This module adds an explicit, optional refinement step *after* DROID terminate:

For every keyframe ``t`` with index in ``forced_keyframes``:

    1. Anchor the SMPL joint world-space positions using a *reference* set
       of camera extrinsics (typically the result of the first SLAM run, or
       the official TRAM cached cameras):

           J^w_{t,k} = R_ref_t @ J^c_{t,k} + T_ref_t

       where ``J^c_{t,k}`` are the SMPL 24 joints in camera frame computed
       from the (frozen) SMPL parameters in ``data.smpl_params``.

    2. Treat the SMPL 2D projections under the reference cameras as the
       observation:

           j^{2d,obs}_{t,k} = pi(K, J^c_{t,k})

    3. Optimize the warmstart camera extrinsics ``(R_t, T_t)`` (still c2w in
       the same convention as DROID's output) so that:

           pi(K, R_t^T (J^w_{t,k} - T_t)) == j^{2d,obs}_{t,k}

       The residual is zero when ``(R_t, T_t) == (R_ref_t, T_ref_t)``, so
       this term acts as a *human-region-weighted soft prior* that pulls the
       warmstart trajectory toward the reference whenever DROID's image-level
       cues become unreliable, without affecting non-keyframe poses or
       background-dominated keyframes that DROID handles correctly.

Notes
-----
- This is a **pose-only** refinement: SMPL params (rotmat / betas / trans)
  are frozen; only camera extrinsics for keyframes are optimized.
- Non-keyframe poses are untouched (they are produced by ``traj_filler``
  upstream of this module).
- All weights / iteration counts / Huber thresholds are exposed via the
  caller, so the pipeline-level YAML config can fully control behavior.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  SMPL joints in camera frame
# ---------------------------------------------------------------------------

def compute_camera_frame_joints(
    rotmat: torch.Tensor,      # (N, 24, 3, 3) camera-frame rotmat
    betas: torch.Tensor,       # (N, 10)
    trans: torch.Tensor,       # (N, 3) camera-frame translation
    keyframe_indices: Sequence[int],
    device: str = "cuda",
    use_default_smpl: bool = True,
) -> torch.Tensor:
    """Run SMPL forward and return camera-frame 24-joint positions.

    Returns
    -------
    joints_cam : (K, 24, 3) tensor on ``device``
    """
    from lib.models.smpl import SMPL  # lazy import to avoid heavy CUDA init

    rotmat = torch.as_tensor(rotmat, dtype=torch.float32, device=device)
    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    trans = torch.as_tensor(trans, dtype=torch.float32, device=device)
    if trans.ndim == 3 and trans.shape[1] == 1:
        trans = trans.squeeze(1)

    kf_idx = torch.as_tensor(list(keyframe_indices), dtype=torch.long)
    rotmat_kf = rotmat[kf_idx]                # (K, 24, 3, 3)
    betas_kf = betas[kf_idx]                  # (K, 10)
    trans_kf = trans[kf_idx]                  # (K, 3)

    smpl = SMPL().to(device)
    smpl.eval()
    with torch.no_grad():
        out = smpl(
            global_orient=rotmat_kf[:, [0]],
            body_pose=rotmat_kf[:, 1:],
            betas=betas_kf,
            pose2rot=False,
            default_smpl=use_default_smpl,
        )
        # default SMPL output has 24 joints already in canonical/zero-trans space
        joints = out.joints[:, :24, :]        # (K, 24, 3)
        joints = joints + trans_kf[:, None, :]  # add camera-frame translation
    return joints


# ---------------------------------------------------------------------------
#  SO(3) helpers
# ---------------------------------------------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    """v: (..., 3) -> (..., 3, 3) skew-symmetric matrix."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z,            -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([v[..., 2],   z,           -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],   v[..., 0],  z], dim=-1),
    ], dim=-2)


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues. omega: (..., 3) -> (..., 3, 3)."""
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-12)
    K = _skew(omega / theta)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape)
    return I + s * K + (1.0 - c) * (K @ K)


# ---------------------------------------------------------------------------
#  Pose-only LM refinement
# ---------------------------------------------------------------------------

def _project_and_jacobian(
    Rt: torch.Tensor,          # (3, 3) c2w
    Tt: torch.Tensor,          # (3,)
    J_world: torch.Tensor,     # (P, 3)
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (proj, J) where proj is (P, 2) and J is (P, 2, 6) Jacobian
    of projection w.r.t. (omega, dT) at (omega=0, dT=0).

    Parameterization (column-vector convention):
        R' = exp(omega) @ R,   T' = T + dT     (c2w)
        p_c = R^T (p_w - T)
        => p_c' = (R')^T (p_w - T') = R^T (I - [omega]_x) (p_w - T - dT)

    Linearizing at (omega, dT) = 0:
        d p_c / d omega = [p_c]_x R^T
        d p_c / d dT    = -R^T
        d (u,v) / d p_c = [[fx/Z, 0, -fx X / Z^2],
                           [0, fy/Z, -fy Y / Z^2]]
    """
    # Camera-frame points
    diff = J_world - Tt[None, :]                   # (P, 3)
    Xc = diff @ Rt                                  # (P, 3) == R^T diff (column conv.)
    Z = Xc[..., 2].clamp(min=1e-6)
    X = Xc[..., 0]
    Y = Xc[..., 1]

    proj = torch.stack([fx * X / Z + cx, fy * Y / Z + cy], dim=-1)   # (P, 2)

    P = Xc.shape[0]
    # d_uv_dXc: (P, 2, 3)
    invZ = 1.0 / Z
    d_uv_dXc = torch.zeros(P, 2, 3, device=Xc.device, dtype=Xc.dtype)
    d_uv_dXc[:, 0, 0] = fx * invZ
    d_uv_dXc[:, 0, 2] = -fx * X * invZ * invZ
    d_uv_dXc[:, 1, 1] = fy * invZ
    d_uv_dXc[:, 1, 2] = -fy * Y * invZ * invZ

    # d_Xc_d_omega = +[Xc]_x  @ R^T    -> shape (P, 3, 3)
    Xc_skew = _skew(Xc)                              # (P, 3, 3)
    Rt_T = Rt.transpose(-1, -2)                      # (3, 3)
    d_Xc_d_omega = Xc_skew @ Rt_T                    # (P, 3, 3)

    # d_Xc_d_dT = -R^T -> (3, 3) broadcast to (P, 3, 3)
    d_Xc_d_dT = (-Rt_T).expand(P, 3, 3)

    # Chain
    d_uv_d_omega = torch.einsum("pij,pjk->pik", d_uv_dXc, d_Xc_d_omega)  # (P, 2, 3)
    d_uv_d_dT = torch.einsum("pij,pjk->pik", d_uv_dXc, d_Xc_d_dT)        # (P, 2, 3)

    J = torch.cat([d_uv_d_omega, d_uv_d_dT], dim=-1)  # (P, 2, 6)
    return proj, J


def _project(
    Rt: torch.Tensor,          # (3, 3) c2w
    Tt: torch.Tensor,          # (3,)   c2w
    J_world: torch.Tensor,     # (P, 3)
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Project world-frame points back to image plane through camera (Rt, Tt) c2w.

    For c2w extrinsics (Rt, Tt) (i.e. p_w = Rt @ p_c + Tt), the inverse is
    p_c = Rt^T @ (p_w - Tt). Implemented via right-multiplication
    ``(p_w - Tt) @ Rt`` which equals ``Rt^T @ (p_w - Tt)`` for orthogonal Rt.
    """
    Xc = (J_world - Tt[None, :]) @ Rt
    z = Xc[..., 2].clamp(min=1e-6)
    u = fx * Xc[..., 0] / z + cx
    v = fy * Xc[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def _huber_weight(r2: torch.Tensor, delta: float) -> torch.Tensor:
    """Iteratively Reweighted Least Squares weight for Huber loss."""
    r = torch.sqrt(r2.clamp(min=1e-12))
    w = torch.where(r <= delta, torch.ones_like(r), delta / r)
    return w


def refine_keyframe_poses_with_smpl(
    cam_R: torch.Tensor,             # (N, 3, 3) c2w trajectory (full frames)
    cam_T: torch.Tensor,             # (N, 3)
    ref_R: torch.Tensor,             # (N, 3, 3) reference c2w (for anchoring J^w)
    ref_T: torch.Tensor,             # (N, 3)
    smpl_rotmat: torch.Tensor,       # (N, 24, 3, 3)
    smpl_betas: torch.Tensor,        # (N, 10)
    smpl_trans: torch.Tensor,        # (N, 3)
    keyframe_indices: Sequence[int],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    weight: float = 1.0,
    huber_delta: float = 4.0,        # pixels
    num_iters: int = 5,
    damping: float = 1e-3,           # LM damping
    pelvis_only: bool = False,
    refine_translation_only: bool = False,
    init_rmse_gate_px: float = 50.0,  # if initial RMSE exceeds this, abort refine
    device: str = "cuda",
    record_debug: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Pose-only LM refinement of warmstart camera extrinsics on keyframes.

    The reference trajectory ``(ref_R, ref_T)`` is used *only* to anchor the
    world-space SMPL joints. Warmstart trajectory ``(cam_R, cam_T)`` is the
    one being refined; non-keyframe poses are returned unchanged.

    All weights are scalar so callers can disable the term entirely by
    passing ``weight=0`` (in which case this function is a no-op).

    Safety gate
    -----------
    Because this term is mathematically a *soft prior* that pulls
    ``(cam_R, cam_T)`` toward ``(ref_R, ref_T)``, it implicitly assumes the
    two trajectories live in the *same* world coordinate system and at the
    same metric scale. If warmstart SLAM has converged to a very different
    coordinate system / scale (which can happen e.g. when the rendered-image
    keyframe stream confuses DROID), the linearization explodes and a single
    LM step can corrupt the trajectory. To avoid this, we measure the
    initial residual RMSE *before* doing any update; if it exceeds
    ``init_rmse_gate_px`` (default 50 px), refinement is aborted and the
    input trajectory is returned untouched.

    Returns
    -------
    cam_R_new, cam_T_new : torch.Tensor
        Refined trajectory (same shapes as input).
    info : dict
        Debug info: per-iter mean residual etc.
    """
    info = {"enabled": True, "num_keyframes": int(len(keyframe_indices))}

    if weight <= 0.0 or len(keyframe_indices) == 0:
        info["enabled"] = False
        info["reason"] = "disabled_or_no_keyframes"
        return cam_R, cam_T, info

    cam_R_t = torch.as_tensor(cam_R, dtype=torch.float32, device=device).clone()
    cam_T_t = torch.as_tensor(cam_T, dtype=torch.float32, device=device).clone()
    ref_R_t = torch.as_tensor(ref_R, dtype=torch.float32, device=device)
    ref_T_t = torch.as_tensor(ref_T, dtype=torch.float32, device=device)

    # SMPL camera-frame joints (frozen during refinement)
    J_cam = compute_camera_frame_joints(
        smpl_rotmat, smpl_betas, smpl_trans,
        keyframe_indices=keyframe_indices,
        device=device,
    )                                          # (K, 24, 3)
    if pelvis_only:
        J_cam = J_cam[:, :1, :]                # (K, 1, 3)

    # World-space anchor: J^w_{t,k} = R_ref_t @ J^c + T_ref_t
    kf_idx = torch.as_tensor(list(keyframe_indices), dtype=torch.long)
    R_ref_kf = ref_R_t[kf_idx]                 # (K, 3, 3)
    T_ref_kf = ref_T_t[kf_idx]                 # (K, 3)
    # J_w = einsum("kij,kpj->kpi", R_ref_kf, J_cam) + T_ref_kf[:, None, :]
    J_w = torch.einsum("kij,kpj->kpi", R_ref_kf, J_cam) + T_ref_kf[:, None, :]  # (K, P, 3)

    # Observation: 2D projection of J_cam through the *current* SMPL camera
    # (intrinsics K), which is exactly the SMPL forward projection. This is
    # what the SMPL pipeline *claims* the joints look like in image space.
    J_cam_z = J_cam[..., 2].clamp(min=1e-6)
    obs_u = fx * J_cam[..., 0] / J_cam_z + cx
    obs_v = fy * J_cam[..., 1] / J_cam_z + cy
    obs = torch.stack([obs_u, obs_v], dim=-1)  # (K, P, 2)

    # Per-keyframe LM iterations (independent problems → loop is fine)
    iter_residuals = []
    P = J_cam.shape[1]

    # ------------------------------------------------------------------
    # Safety gate: compute initial RMSE under the *current* cam_R/T.
    # If it is unreasonably large (typically because cam_R/T and ref_R/T
    # do not share the same world frame / scale), abort refinement.
    # ------------------------------------------------------------------
    with torch.no_grad():
        sq_sum = 0.0
        cnt = 0
        bad_kf = 0
        for ki, t in enumerate(keyframe_indices):
            R_t = cam_R_t[t]
            T_t = cam_T_t[t]
            diff = J_w[ki] - T_t[None, :]
            Xc = diff @ R_t
            Z = Xc[..., 2]
            if (Z <= 1e-3).any():
                bad_kf += 1
                continue
            u = fx * Xc[..., 0] / Z + cx
            v = fy * Xc[..., 1] / Z + cy
            r2d = torch.stack([u, v], dim=-1) - obs[ki]
            sq_sum += float((r2d ** 2).sum().item())
            cnt += int(r2d.numel())
        init_rmse = (sq_sum / max(cnt, 1)) ** 0.5
    info["init_rmse_pixels"] = init_rmse
    info["num_keyframes_with_negative_z"] = bad_kf
    if init_rmse > init_rmse_gate_px or bad_kf > 0:
        info["enabled"] = False
        info["reason"] = (
            f"init_rmse={init_rmse:.1f}px exceeds gate={init_rmse_gate_px}px "
            f"or {bad_kf} keyframes have non-positive Z; aborting refine "
            "(cam_R/T and ref_R/T likely live in different world frames)."
        )
        # Return inputs unchanged
        return cam_R, cam_T, info

    for it in range(num_iters):
        total_res = 0.0
        total_count = 0
        for ki, t in enumerate(keyframe_indices):
            R_t = cam_R_t[t]
            T_t = cam_T_t[t]

            # Forward + analytic Jacobian
            proj, Jmat_3d = _project_and_jacobian(
                R_t, T_t, J_w[ki], fx, fy, cx, cy
            )                                                        # (P,2), (P,2,6)
            r2d = proj - obs[ki]                                     # (P, 2)
            r = r2d.reshape(-1)                                      # (2P,)
            r2 = (r2d ** 2).sum(-1)                                  # (P,)
            w_irls = _huber_weight(r2, huber_delta)                  # (P,)

            total_res += float(r.detach().pow(2).sum().item())
            total_count += int(r.numel())

            Jmat = Jmat_3d.reshape(-1, 6)                            # (2P, 6)
            # IRLS weight per residual (treat sqrt(w) as the residual scaling
            # so that the normal equations J^T W J δ = -J^T W r hold with the
            # implicit diagonal weight matrix W = w_irls (per-joint, per-uv)).
            w_uv = w_irls.repeat_interleave(2)                       # (2P,)
            W = w_uv * float(weight)                                 # (2P,)
            JW = Jmat * W[:, None]                                   # = W * J
            JtWJ = JW.T @ Jmat                                       # (6, 6)
            JtWr = JW.T @ r                                          # (6,)

            if refine_translation_only:
                # Solve only for dT (rotation block frozen).
                # Keep the 3×3 dT submatrix + dT residual.
                A = JtWJ[3:, 3:] + damping * torch.eye(3, device=device)
                b = -JtWr[3:]
                try:
                    d_T = torch.linalg.solve(A, b)
                except RuntimeError:
                    continue
                cam_R_t[t] = R_t  # unchanged
                cam_T_t[t] = T_t + d_T
                continue

            # LM damping (full 6×6)
            JtWJ = JtWJ + damping * torch.eye(6, device=device)

            try:
                delta = torch.linalg.solve(JtWJ, -JtWr)
            except RuntimeError:
                continue

            d_omega = delta[:3]
            d_T = delta[3:]
            cam_R_t[t] = _so3_exp(d_omega) @ R_t
            cam_T_t[t] = T_t + d_T

        mean_res = (total_res / max(total_count, 1)) ** 0.5
        iter_residuals.append(mean_res)

    if record_debug:
        info["iter_rmse_pixels"] = iter_residuals
        info["weight"] = float(weight)
        info["huber_delta"] = float(huber_delta)
        info["num_iters"] = int(num_iters)
        info["pelvis_only"] = bool(pelvis_only)
        info["refine_translation_only"] = bool(refine_translation_only)

    # Move back to caller's device / dtype
    if isinstance(cam_R, torch.Tensor):
        cam_R_out = cam_R_t.to(device=cam_R.device, dtype=cam_R.dtype)
        cam_T_out = cam_T_t.to(device=cam_T.device, dtype=cam_T.dtype)
    else:
        cam_R_out = cam_R_t.cpu()
        cam_T_out = cam_T_t.cpu()

    return cam_R_out, cam_T_out, info

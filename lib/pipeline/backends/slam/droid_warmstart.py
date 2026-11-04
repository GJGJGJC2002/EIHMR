"""DroidSLAM Warmstart Backend - Forced keyframes with warm start initialization

This backend runs DROID-SLAM on full-frame images (from adjacent_smpl_renderer in
keyframe mode), using forced keyframes and warm start poses from a previous SLAM run.
The trajectory filler returns poses for ALL frames, so no interpolation is needed.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from glob import glob
import os

from lib.pipeline.core.component import Backend


class DroidWarmstartBackend(Backend):
    """
    DROID-SLAM Warm Start 后端

    在全帧渲染图像上运行 DROID-SLAM，强制使用第一次 SLAM 的关键帧，
    并用第一次的位姿/视差做 warm start 初始化。
    traj_filler 直接返回所有帧的轨迹，无需插值。

    Config:
        align_to_world: 是否对齐到世界坐标系
        enable_scale_alignment: 是否将尺度对齐到第一次 SLAM
    """

    DEFAULT_CONFIG = {
        'align_to_world': True,
        'enable_scale_alignment': True,
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.align_to_world = self.config['align_to_world']

    def setup(self):
        """初始化"""
        self._is_setup = True
        self.logger.info("DroidSLAM Warmstart backend initialized")

    def estimate_camera(
        self,
        image_dir: str,
        forced_keyframes: list,
        initial_poses_se3: Dict[int, np.ndarray] = None,
        initial_disps: Dict[int, np.ndarray] = None,
        masks=None,
        intrinsics: Optional[list] = None,
        is_static: bool = False,
        original_image_dir: str = None,
        use_rendered_for_keyframes: bool = False,
        keyframe_render_mode: str = 'all',
        texture_threshold: float = 500.0,
        render_fusion_mode: str = 'replace',
        render_alpha: float = 1.0,
        # --- SMPL pose-only refinement passthrough ---
        smpl_refine_cfg: Optional[Dict[str, Any]] = None,
        smpl_rotmat=None,
        smpl_betas=None,
        smpl_trans=None,
        ref_cam_R=None,
        ref_cam_T=None,
        refine_intrinsics: Optional[Tuple[float, float, float, float]] = None,
        adjacent_projection_refine_cfg: Optional[Dict[str, Any]] = None,
        adjacent_projection_info=None,
        return_keyframe_info: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 warm start 运行 DROID-SLAM

        Args:
            image_dir: 全帧渲染图像目录 (adjacent_smpl/)
            forced_keyframes: 强制关键帧的帧索引列表
            initial_poses_se3: 关键帧初始 SE3 pose {tstamp: array[7]}
            initial_disps: 关键帧初始 disparity {tstamp: array[h//8, w//8]}
            masks: 人体 mask
            intrinsics: 相机内参 [fx, fy, cx, cy]
            is_static: 是否为静态相机
            original_image_dir: 原始图像目录
            use_rendered_for_keyframes: 关键帧是否使用渲染图像
            keyframe_render_mode: 'all' (所有关键帧用渲染) 或 'adaptive' (根据纹理自适应)
            texture_threshold: adaptive 模式的纹理阈值
            render_fusion_mode: 'replace' 直接使用渲染图；'alpha' 混合原图和渲染图
            render_alpha: alpha 模式下渲染图权重
            return_keyframe_info: 额外返回关键帧的原始 DROID tstamps / SE3 poses /
                低分辨率 disps，供后续 frontend-only pass 做 warm start

        Returns:
            cam_R: 旋转矩阵 [N_total, 3, 3]
            cam_T: 平移向量 [N_total, 3]
            keyframe_info: 仅当 return_keyframe_info=True 时返回
        """
        self.ensure_setup()

        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")

        self.logger.info(f"Running Warmstart DROID-SLAM on {image_dir}")
        self.logger.info(f"  Forced keyframes: {len(forced_keyframes)}")
        self.logger.info(f"  Warm start poses: {len(initial_poses_se3) if initial_poses_se3 else 0}")
        self.logger.info(f"  Use rendered for keyframes: {use_rendered_for_keyframes}")
        self.logger.info(f"  Keyframe render mode: {keyframe_render_mode}")
        self.logger.info(f"  Render fusion mode: {render_fusion_mode}, alpha={render_alpha}")
        if keyframe_render_mode == 'adaptive':
            self.logger.info(f"  Texture threshold: {texture_threshold}")
        if original_image_dir:
            self.logger.info(f"  Original image source: {original_image_dir}")
        if smpl_refine_cfg is not None and smpl_refine_cfg.get('enabled', False):
            self.logger.info(
                "  SMPL pose-refine: ENABLED "
                f"(weight={smpl_refine_cfg.get('weight', 1.0)}, "
                f"iters={smpl_refine_cfg.get('num_iters', 5)}, "
                f"huber={smpl_refine_cfg.get('huber_delta', 4.0)}, "
                f"pelvis_only={smpl_refine_cfg.get('pelvis_only', False)}, "
                f"refine_translation_only={smpl_refine_cfg.get('refine_translation_only', False)})"
            )
        else:
            self.logger.info("  SMPL pose-refine: DISABLED")

        from lib.camera import run_metric_slam_warmstart

        result = run_metric_slam_warmstart(
            image_dir,
            forced_keyframes=forced_keyframes,
            initial_poses_se3=initial_poses_se3,
            initial_disps=initial_disps,
            masks=masks,
            calib=intrinsics,
            is_static=is_static,
            original_image_dir=original_image_dir,
            use_rendered_for_keyframes=use_rendered_for_keyframes,
            keyframe_render_mode=keyframe_render_mode,
            texture_threshold=texture_threshold,
            render_fusion_mode=render_fusion_mode,
            render_alpha=render_alpha,
            smpl_refine_cfg=smpl_refine_cfg,
            smpl_rotmat=smpl_rotmat,
            smpl_betas=smpl_betas,
            smpl_trans=smpl_trans,
            ref_cam_R=ref_cam_R,
            ref_cam_T=ref_cam_T,
            refine_intrinsics=refine_intrinsics,
            adjacent_projection_refine_cfg=adjacent_projection_refine_cfg,
            adjacent_projection_info=adjacent_projection_info,
            return_keyframe_info=return_keyframe_info,
        )

        if return_keyframe_info:
            cam_R, cam_T, keyframe_info = result
        else:
            cam_R, cam_T = result
            keyframe_info = None

        self.logger.info(f"Warmstart SLAM output: R={cam_R.shape}, T={cam_T.shape}")

        # Align to world frame
        if self.align_to_world:
            self.logger.info("Aligning to world frame...")
            # Use original images for SPEC gravity estimation (avoid SMPL mesh/mask artifacts)
            align_image_dir = original_image_dir if original_image_dir else image_dir
            processed_images = sorted(glob(os.path.join(align_image_dir, "*.jpg")))
            if len(processed_images) == 0:
                processed_images = sorted(glob(os.path.join(align_image_dir, "*.png")))
            if len(processed_images) == 0:
                # Fallback to image_dir
                processed_images = sorted(glob(os.path.join(image_dir, "*.jpg")))
            if len(processed_images) == 0:
                raise ValueError(f"No images found in {align_image_dir} or {image_dir}")

            from lib.camera import align_cam_to_world
            cam_R, cam_T, spec_f = align_cam_to_world(
                processed_images[0], cam_R, cam_T
            )
            self.logger.info(f"Aligned to world frame, spec_focal: {spec_f:.2f}")

        if return_keyframe_info:
            return cam_R, cam_T, keyframe_info
        return cam_R, cam_T

    def align_scale_to_reference(
        self,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor,
        ref_R: torch.Tensor,
        ref_T: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将轨迹尺度对齐到参考轨迹（第一次 SLAM 的结果）

        使用 Umeyama Sim3 对齐求解尺度因子 s。

        Args:
            cam_R: warmstart 的旋转 [N, 3, 3]
            cam_T: warmstart 的平移 [N, 3]
            ref_R: 参考的旋转 [N, 3, 3]
            ref_T: 参考的平移 [N, 3]

        Returns:
            cam_R: 旋转不变
            cam_T_scaled: 尺度校正后的平移
        """
        if isinstance(cam_T, torch.Tensor):
            cam_T_np = cam_T.cpu().numpy()
        else:
            cam_T_np = np.array(cam_T)

        if isinstance(ref_T, torch.Tensor):
            ref_T_np = ref_T.cpu().numpy()
        else:
            ref_T_np = np.array(ref_T)

        try:
            n = min(len(cam_T_np), len(ref_T_np))
            cam_T_np = cam_T_np[:n]
            ref_T_np = ref_T_np[:n]

            s, R_align, t_align = self._umeyama_alignment(cam_T_np, ref_T_np)
            self.logger.info(f"Scale alignment: s={s:.4f}")

            if isinstance(cam_T, torch.Tensor):
                cam_T_scaled = cam_T * s
            else:
                cam_T_scaled = torch.from_numpy(cam_T_np * s).float()

            return cam_R, cam_T_scaled

        except Exception as e:
            self.logger.warning(f"Scale alignment failed: {e}, returning original trajectory")
            return cam_R, cam_T

    @staticmethod
    def _umeyama_alignment(src: np.ndarray, tgt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Umeyama Sim3 alignment: tgt = s * R @ src + t"""
        assert src.shape == tgt.shape
        n, dim = src.shape

        src_mean = src.mean(axis=0)
        tgt_mean = tgt.mean(axis=0)
        src_centered = src - src_mean
        tgt_centered = tgt - tgt_mean

        src_var = np.sum(src_centered ** 2) / n
        H = (tgt_centered.T @ src_centered) / n
        U, D, Vt = np.linalg.svd(H)

        S = np.eye(dim)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[dim - 1, dim - 1] = -1

        R = U @ S @ Vt
        s = np.trace(np.diag(D) @ S) / src_var
        t = tgt_mean - s * R @ src_mean

        return s, R, t

    def cleanup(self):
        """清理资源"""
        super().cleanup()

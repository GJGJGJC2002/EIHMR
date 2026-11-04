"""DroidSLAM Backend - Metric SLAM using DROID-SLAM"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class DroidSLAMBackend(Backend):
    """
    DROID-SLAM 后端
    
    使用 DROID-SLAM 进行相机运动估计。
    支持使用 mask 遮挡动态物体（如人体）。
    """
    
    DEFAULT_CONFIG = {
        'use_masks': True,
        'align_to_world': True,
    }
    
    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        
        self.use_masks = self.config['use_masks']
        self.align_to_world = self.config['align_to_world']
    
    def setup(self):
        """初始化（DROID-SLAM 在每次调用时动态加载）"""
        # DROID-SLAM 是在 run_metric_slam 中动态加载的
        self._is_setup = True
        self.logger.info("DroidSLAM backend initialized")
    
    def estimate_camera(
        self,
        image_folder: str,
        masks: Optional[torch.Tensor] = None,
        intrinsics: Optional[List[float]] = None,
        is_static: bool = False,
        return_keyframe_info: bool = False
    ):
        """
        估计相机运动
        
        Args:
            image_folder: 图像文件夹路径
            masks: 人体 mask [N, H, W]，用于遮挡
            intrinsics: 相机内参 [fx, fy, cx, cy]
            is_static: 是否为静态相机
            return_keyframe_info: 是否额外返回关键帧信息（用于 warm start）
        
        Returns:
            cam_R: 相机旋转矩阵 [N, 3, 3]
            cam_T: 相机平移向量 [N, 3]
            keyframe_info: (仅当 return_keyframe_info=True) 包含 keyframe_tstamps, poses_se3, disps 的 dict
        """
        self.ensure_setup()
        
        from lib.camera import run_metric_slam
        
        # 运行 DROID-SLAM
        kwargs = dict(calib=intrinsics, is_static=is_static, return_keyframe_info=return_keyframe_info)
        if self.use_masks and masks is not None:
            result = run_metric_slam(image_folder, masks=masks, **kwargs)
        else:
            result = run_metric_slam(image_folder, **kwargs)
        
        if return_keyframe_info:
            cam_R, cam_T, keyframe_info = result
            return cam_R, cam_T, keyframe_info
        else:
            cam_R, cam_T = result
            return cam_R, cam_T
    
    def align_to_world_frame(
        self,
        first_image_path: str,
        cam_R: torch.Tensor,
        cam_T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        将相机轨迹对齐到世界坐标系
        
        使用重力方向和尺度估计将相机坐标系对齐到世界坐标系。
        
        Args:
            first_image_path: 第一帧图像路径（用于重力估计）
            cam_R: 相机旋转矩阵 [N, 3, 3]
            cam_T: 相机平移向量 [N, 3]
        
        Returns:
            world_R: 世界坐标系下的旋转 [N, 3, 3]
            world_T: 世界坐标系下的平移 [N, 3]
            spec_focal: 估计的焦距
        """
        from lib.camera import align_cam_to_world
        
        world_R, world_T, spec_f = align_cam_to_world(
            first_image_path, cam_R, cam_T
        )
        
        return world_R, world_T, spec_f
    
    def cleanup(self):
        """清理资源"""
        super().cleanup()



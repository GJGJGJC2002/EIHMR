"""SegmentationComponent - 图像分割组件"""

from typing import Dict, Any, List
import numpy as np
import cv2
import torch
from tqdm import tqdm

from lib.pipeline.core.component import BackendComponent
from lib.pipeline.core.data import PipelineData
from lib.pipeline.backends.segmentation import SAMBackend


class SegmentationComponent(BackendComponent):
    """
    图像分割组件
    
    使用检测到的边界框作为提示，分割出人体区域。
    结果用于后续的 SLAM（遮挡人体区域）。
    
    Config:
        backend: 分割后端类型 ('sam', 'sam2', ...)
    """
    
    COMPONENT_TYPE = "segmentation"
    
    BACKENDS = {
        'sam': SAMBackend,
    }
    
    DEFAULT_BACKEND = 'sam'
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要图像和边界框"""
        has_images = data.images is not None or len(data.image_paths) > 0
        has_boxes = data.bboxes is not None
        return has_images and has_boxes
    
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行分割

        为每一帧图像生成人体 mask，结果存储在 data.masks 中。
        """
        # 获取图像（如果未加载则从路径加载）
        if data.images is not None:
            images = data.images
        else:
            if data.metadata.get('inference_mode') == 'video':
                images = data.image_paths
            else:
                images = self._load_images(data.image_paths)
                data.images = images

        bboxes = data.bboxes  # [N, K, 5] or [N, 4]

        # 兼容 [N, 4] 格式的 GT bboxes（无 score，无 K 维度）
        if bboxes.ndim == 2:
            # [N, 4] -> [N, 1, 5]：增加 K 维度，补 score=1
            scores = np.ones((len(bboxes), 1))
            bboxes = np.hstack([bboxes, scores])[:, np.newaxis, :]
        # Mask every detected person, not only the target: unmasked bystanders are moving
        # features for SLAM.  The target box stays first, so HPE/OAR inputs are unchanged.
        people = data.metadata.get('person_boxes')
        if people is not None:
            people = np.asarray(people, dtype=np.float32)
            if people.shape[:1] != bboxes.shape[:1] or people.shape[-1] != 5:
                raise ValueError('person_boxes must be [N, K, 5] aligned with the frames')
            bboxes = np.concatenate([bboxes, people], axis=1)

        num_frames = len(images)
        masks = []
        
        self.logger.info(f"Segmenting {num_frames} frames...")
        
        for i, img in enumerate(tqdm(images, desc='Segmentation')):
            if isinstance(img, str):
                img = cv2.imread(img)
                if img is None:
                    raise ValueError(f'Failed to load image: {images[i]}')
            # 获取当前帧的有效边界框
            frame_boxes = bboxes[i]  # [K, 5]
            
            # 过滤掉无效的边界框（全0）
            valid_mask = frame_boxes[:, 4] > 0  # score > 0
            valid_boxes = frame_boxes[valid_mask]
            
            # 分割
            mask = self.backend.segment(img, valid_boxes)
            masks.append(mask)
        
        # Stack masks
        data.masks = torch.stack(masks)  # [N, H, W]
        
        # 统计
        total_pixels = data.masks.numel()
        masked_pixels = data.masks.sum().item()
        data.metadata['segmentation_stats'] = {
            'total_frames': num_frames,
            'mask_coverage': masked_pixels / total_pixels if total_pixels > 0 else 0,
        }
        
        self.logger.info(f"Segmentation completed, mask coverage: {masked_pixels/total_pixels*100:.2f}%")

        return data

    def _load_images(self, image_paths: List[str]) -> np.ndarray:
        """加载图像"""
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Failed to load image: {path}")
            images.append(img)
        return np.array(images)



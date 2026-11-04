"""SAM Backend - Segment Anything Model"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class SAMBackend(Backend):
    """
    SAM (Segment Anything Model) 分割后端
    
    使用给定的边界框提示进行人体分割。
    """
    
    DEFAULT_CONFIG = {
        'checkpoint': 'data/pretrain/sam_vit_h_4b8939.pth',
        'model_type': 'vit_h',
    }
    
    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)
        
        self.predictor = None
        self.sam_model = None
    
    def setup(self):
        """初始化 SAM 模型"""
        from segment_anything import SamPredictor, sam_model_registry
        
        self.sam_model = sam_model_registry[self.config['model_type']](
            checkpoint=self.config['checkpoint']
        )
        self.sam_model.to(self.device)
        self.predictor = SamPredictor(self.sam_model)
        
        self._is_setup = True
        self.logger.info("SAM backend initialized")
    
    def segment(
        self, 
        image: np.ndarray, 
        boxes: np.ndarray
    ) -> torch.Tensor:
        """
        使用边界框提示分割图像
        
        Args:
            image: BGR 格式图像 [H, W, 3]
            boxes: 边界框 [K, 4] 或 [K, 5]（最后一列为 score，会被忽略）
        
        Returns:
            合并后的分割 mask [H, W]
        """
        self.ensure_setup()
        
        if len(boxes) == 0:
            return torch.zeros((image.shape[0], image.shape[1]), dtype=torch.uint8)
        
        # 只取前 4 列（坐标）
        if boxes.shape[1] > 4:
            boxes = boxes[:, :4]
        
        with torch.no_grad():
            # 设置图像
            self.predictor.set_image(image, image_format='BGR')
            
            # 转换边界框格式
            bb = torch.tensor(boxes, device=self.device)
            bb = self.predictor.transform.apply_boxes_torch(bb, image.shape[:2])
            
            # 预测 masks
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=bb,
                multimask_output=False
            )
            
            # 合并所有 masks
            masks = masks.cpu().squeeze(1)  # [K, H, W]
            merged_mask = masks.sum(dim=0).clamp(0, 1)  # [H, W]
        
        return merged_mask.byte()
    
    def segment_batch(
        self,
        images: List[np.ndarray],
        boxes_list: List[np.ndarray]
    ) -> List[torch.Tensor]:
        """
        批量分割
        
        Args:
            images: 图像列表
            boxes_list: 每张图像对应的边界框列表
        
        Returns:
            mask 列表
        """
        results = []
        for image, boxes in zip(images, boxes_list):
            mask = self.segment(image, boxes)
            results.append(mask)
        return results
    
    def cleanup(self):
        """清理资源"""
        self.predictor = None
        self.sam_model = None
        super().cleanup()



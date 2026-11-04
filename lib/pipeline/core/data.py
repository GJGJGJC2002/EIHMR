"""PipelineData - Pipeline 中统一的数据容器"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
import pickle
import gzip
import os
from datetime import datetime


@dataclass
class CameraParams:
    """相机参数数据结构"""
    R: Optional[Union[np.ndarray, torch.Tensor]] = None  # 旋转矩阵 [N, 3, 3]
    T: Optional[Union[np.ndarray, torch.Tensor]] = None  # 平移向量 [N, 3]
    intrinsics: Optional[np.ndarray] = None  # 内参矩阵 [3, 3]
    focal_length: Optional[float] = None
    principal_point: Optional[np.ndarray] = None  # [2]
    world_R: Optional[Union[np.ndarray, torch.Tensor]] = None  # 世界坐标系旋转
    world_T: Optional[Union[np.ndarray, torch.Tensor]] = None  # 世界坐标系平移
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key in ['R', 'T', 'intrinsics', 'focal_length', 'principal_point', 'world_R', 'world_T']:
            value = getattr(self, key)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraParams':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SMPLParams:
    """SMPL 参数数据结构"""
    poses: Optional[Union[np.ndarray, torch.Tensor]] = None  # 姿态参数
    betas: Optional[Union[np.ndarray, torch.Tensor]] = None  # 形状参数
    trans: Optional[Union[np.ndarray, torch.Tensor]] = None  # 相机坐标系下的平移
    global_trans: Optional[Union[np.ndarray, torch.Tensor]] = None  # 世界坐标系下的平移
    rotmat: Optional[Union[np.ndarray, torch.Tensor]] = None  # 旋转矩阵
    pred_cam: Optional[Union[np.ndarray, torch.Tensor]] = None  # 预测的相机参数
    vertices: Optional[Union[np.ndarray, torch.Tensor]] = None  # 顶点
    joints: Optional[Union[np.ndarray, torch.Tensor]] = None  # 关节点
    # GVHMR 特有字段
    global_orient_w: Optional[Union[np.ndarray, torch.Tensor]] = None  # 世界坐标系 root orient (F, 3) axis-angle
    global_orient_c: Optional[Union[np.ndarray, torch.Tensor]] = None  # 相机坐标系 root orient (F, 3) axis-angle
    body_pose_aa: Optional[Union[np.ndarray, torch.Tensor]] = None  # body pose (F, 63) axis-angle
    transl_w_raw: Optional[Union[np.ndarray, torch.Tensor]] = None  # 滑步修正前的 world transl (F, 3)
    static_conf_logits: Optional[Union[np.ndarray, torch.Tensor]] = None  # 静止置信度 logits (F, J)
    skeleton_offset: Optional[Union[np.ndarray, torch.Tensor]] = None  # SMPL skeleton root offset (3,)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SMPLParams':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineData:
    """
    Pipeline 中统一的数据容器
    
    在 Pipeline 执行过程中，各组件通过此数据结构传递数据。
    支持序列化、反序列化和中间结果缓存。
    """
    
    # === 输入数据 ===
    sequence_name: str = ""
    sequence_path: str = ""
    image_paths: List[str] = field(default_factory=list)
    images: Optional[np.ndarray] = None  # [N, H, W, 3] BGR 格式
    
    # === 检测和分割结果 ===
    bboxes: Optional[np.ndarray] = None  # [N, K, 5] (x1, y1, x2, y2, score)
    masks: Optional[torch.Tensor] = None  # [N, H, W] 人体 mask
    tracks: Optional[Dict[int, List]] = None  # 跟踪结果
    
    # === 相机参数 ===
    camera_params: Optional[CameraParams] = None
    gt_camera_params: Optional[CameraParams] = None  # GT 相机参数（评估用）
    
    # === SMPL 参数 ===
    smpl_params: Optional[SMPLParams] = None
    gt_smpl_params: Optional[SMPLParams] = None  # GT SMPL 参数（评估用）
    
    # === Ground Truth 标注 ===
    annotations: Optional[Dict[str, Any]] = None
    valid_frames_mask: Optional[np.ndarray] = None
    
    # === 评估结果 ===
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # === 元数据 ===
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # === Pipeline 状态 ===
    current_stage: str = ""
    iteration: int = 0
    error_message: Optional[str] = None
    should_stop: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
        if 'history' not in self.metadata:
            self.metadata['history'] = []
    
    def record_stage(self, stage_name: str, info: Dict[str, Any] = None):
        """记录 Pipeline 执行阶段"""
        record = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'iteration': self.iteration
        }
        if info:
            record.update(info)
        self.metadata['history'].append(record)
        self.current_stage = stage_name
    
    def get_num_frames(self) -> int:
        """获取帧数"""
        if self.images is not None:
            return len(self.images)
        if self.image_paths:
            return len(self.image_paths)
        return 0
    
    def get_image_size(self) -> Optional[tuple]:
        """获取图像尺寸 (H, W)"""
        if self.images is not None and len(self.images) > 0:
            return self.images[0].shape[:2]
        return None
    
    # === 序列化方法 ===
    
    def save(self, path: str, incremental: bool = False, previous_stage: str = ""):
        """保存到文件
        
        Args:
            path: 保存路径
            incremental: 是否使用增量保存（只保存当前阶段新增/变更的字段）
            previous_stage: 上一阶段的名称（增量保存时用于确定哪些字段是新增的）
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # 根据当前阶段确定哪些字段需要保存
        # 基础字段（很小，始终保存）
        data = {
            '_version': 2,  # 标记新格式
            '_incremental': incremental,
            'sequence_name': self.sequence_name,
            'sequence_path': self.sequence_path,
            'image_paths': self.image_paths,
            'current_stage': self.current_stage,
            'iteration': self.iteration,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'valid_frames_mask': self.valid_frames_mask,
        }
        
        # 不保存 annotations（缓存加载时不使用，且体积较大）
        # 不保存原始图像数据（太大）
        
        if incremental:
            # 增量模式：只保存当前阶段新增/变更的字段
            stage = self.current_stage.lower()
            
            if 'detection' in stage:
                data['bboxes'] = self.bboxes
            elif 'segmentation' in stage:
                data['masks'] = self.masks.cpu().numpy() if isinstance(self.masks, torch.Tensor) else self.masks
            elif 'camera' in stage or 'slam' in stage:
                data['camera_params'] = self.camera_params.to_dict() if self.camera_params else None
                data['gt_camera_params'] = self.gt_camera_params.to_dict() if self.gt_camera_params else None
            elif 'hpe' in stage or 'preprocessing' in stage or 'skating' in stage:
                # 匹配 HPE 组件（如 "gvhmr_preprocessing"）和后处理组件（如 "skating_removal"）
                data['smpl_params'] = self.smpl_params.to_dict() if self.smpl_params else None
                data['gt_smpl_params'] = self.gt_smpl_params.to_dict() if self.gt_smpl_params else None
                # HPE 组件也会设置 camera_params（内参），需要一并保存
                data['camera_params'] = self.camera_params.to_dict() if self.camera_params else None
            else:
                # 未知阶段，保存所有字段（安全回退）
                data['bboxes'] = self.bboxes
                data['masks'] = self.masks.cpu().numpy() if isinstance(self.masks, torch.Tensor) else self.masks
                data['tracks'] = self.tracks
                data['camera_params'] = self.camera_params.to_dict() if self.camera_params else None
                data['gt_camera_params'] = self.gt_camera_params.to_dict() if self.gt_camera_params else None
                data['smpl_params'] = self.smpl_params.to_dict() if self.smpl_params else None
                data['gt_smpl_params'] = self.gt_smpl_params.to_dict() if self.gt_smpl_params else None
        else:
            # 完整模式：保存所有字段（向后兼容）
            data['bboxes'] = self.bboxes
            data['masks'] = self.masks.cpu().numpy() if isinstance(self.masks, torch.Tensor) else self.masks
            data['tracks'] = self.tracks
            data['camera_params'] = self.camera_params.to_dict() if self.camera_params else None
            data['gt_camera_params'] = self.gt_camera_params.to_dict() if self.gt_camera_params else None
            data['smpl_params'] = self.smpl_params.to_dict() if self.smpl_params else None
            data['gt_smpl_params'] = self.gt_smpl_params.to_dict() if self.gt_smpl_params else None
        
        # 使用 gzip 压缩保存，显著减少磁盘占用
        with gzip.open(path, 'wb', compresslevel=4) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: str) -> 'PipelineData':
        """从文件加载（兼容新旧格式）"""
        # 尝试用 gzip 打开（新格式），失败则用普通 pickle（旧格式）
        try:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        # 重建数据结构
        pipeline_data = cls(
            sequence_name=data.get('sequence_name', ''),
            sequence_path=data.get('sequence_path', ''),
            image_paths=data.get('image_paths', []),
            bboxes=data.get('bboxes'),
            masks=torch.from_numpy(data['masks']) if data.get('masks') is not None else None,
            tracks=data.get('tracks'),
            annotations=data.get('annotations'),
            valid_frames_mask=data.get('valid_frames_mask'),
            metrics=data.get('metrics', {}),
            metadata=data.get('metadata', {}),
            current_stage=data.get('current_stage', ''),
            iteration=data.get('iteration', 0),
        )
        
        if data.get('camera_params'):
            pipeline_data.camera_params = CameraParams.from_dict(data['camera_params'])
        if data.get('gt_camera_params'):
            pipeline_data.gt_camera_params = CameraParams.from_dict(data['gt_camera_params'])
        if data.get('smpl_params'):
            pipeline_data.smpl_params = SMPLParams.from_dict(data['smpl_params'])
        if data.get('gt_smpl_params'):
            pipeline_data.gt_smpl_params = SMPLParams.from_dict(data['gt_smpl_params'])
        
        return pipeline_data
    
    def save_results(self, output_dir: str, prefix: str = ""):
        """保存各类结果到单独的文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        
        # 保存相机参数（使用压缩格式）
        if self.camera_params:
            np.savez_compressed(
                os.path.join(output_dir, f'{prefix}camera.npz'),
                **self.camera_params.to_dict()
            )
        
        # 保存 SMPL 参数（使用压缩格式）
        if self.smpl_params:
            np.savez_compressed(
                os.path.join(output_dir, f'{prefix}smpl.npz'),
                **self.smpl_params.to_dict()
            )
        
        # 保存评估指标
        if self.metrics:
            import json
            with open(os.path.join(output_dir, f'{prefix}metrics.json'), 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # 保存 masks（不再在 intermediate 阶段单独保存，仅在最终输出时保存）
        # 如果 prefix 为空（最终输出）才保存 masks
        if not prefix and self.masks is not None:
            torch.save(self.masks, os.path.join(output_dir, f'{prefix}masks.pt'))
    
    def clone(self) -> 'PipelineData':
        """创建深拷贝"""
        import copy
        return copy.deepcopy(self)
    
    def __repr__(self) -> str:
        return (
            f"PipelineData("
            f"seq='{self.sequence_name}', "
            f"frames={self.get_num_frames()}, "
            f"stage='{self.current_stage}', "
            f"iteration={self.iteration})"
        )



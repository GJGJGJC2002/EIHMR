"""PromptHMRBackend - PromptHMR 人体运动估计后端

PromptHMR 使用自己的图像模型提取特征（2通道: smpl_token + loc_token），
然后通过 PromptHMR 视频头（GVHMR 改进版）进行时序推理。

输出格式与 GVHMR 完全一致：
  - smpl_params_global: 世界坐标系(ay)下的 global_orient, body_pose, betas, transl
  - smpl_params_incam: 相机坐标系下的 global_orient, body_pose, betas, transl
  - static_conf_logits 用于后续滑步消除

该后端只执行推理（不做后处理），后处理由独立的 SkatingRemovalComponent 完成。

子步骤缓存：
  VitPose 关键点检测和 PromptHMR 图像模型特征提取是最耗时的两个子步骤。
  当 substep_cache_dir 被设置时，这两步的结果会被缓存到磁盘，
  下次处理同一序列（相同图像 + bbox）时直接加载，跳过推理。
"""

import sys
import os
import hashlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch

from lib.pipeline.core.component import Backend


class PromptHMRBackend(Backend):
    """
    PromptHMR 后端 - 基于 PromptHMR 图像特征的全局人体运动估计

    使用 PromptHMR 图像模型提取 2 通道特征，再通过 PromptHMR 视频头
    估计世界坐标系和相机坐标系下的 SMPL 参数。

    Config:
        prompthmr_root: PromptHMR 项目根目录
        static_cam: 是否假设静态相机（跳过 VO）
        f_mm: 相机焦距 (mm)，None 则使用默认估计
        phmr_ckpt: PromptHMR 图像模型 checkpoint 路径
        phmr_vid_cfg: PromptHMR 视频头配置文件路径
        phmr_vid_ckpt: PromptHMR 视频头 checkpoint 路径
        mask_prompt: 是否使用 mask prompt（默认 True）
        img_size: PromptHMR 输入图像尺寸（默认 896）
    """

    DEFAULT_CONFIG = {
        'prompthmr_root': '../PromptHMR',
        'static_cam': True,
        'f_mm': None,
        'phmr_ckpt': 'data/pretrain/phmr/checkpoint.ckpt',
        'phmr_vid_cfg': 'data/pretrain/phmr_vid/prhmr_release_002.yaml',
        'phmr_vid_ckpt': 'data/pretrain/phmr_vid/prhmr_release_002.ckpt',
        # Match PromptHMR/scripts/eval_phmr.py video eval by default: box prompt only.
        'mask_prompt': False,
        'kpt_prompt': False,
        'img_size': 896,
        # Use DemoPL.predict(... postproc=True) with the given calibration K when available.
        'official_video_head': True,
        'use_gt_video_k': True,
        # RICH_TEST 官方评估会在 padding 后把主点移到 896 输入中心。
        'center_principal_point_after_padding': False,
        'smooth_bbx': False,
        # 子步骤缓存配置
        'substep_cache_dir': None,       # 缓存目录，None 则不启用子步骤缓存
        'substep_cache_use': True,       # 是否读取已有缓存
        'substep_cache_overwrite': False, # 是否覆盖已有缓存
    }

    def __init__(self, config: Dict[str, Any]):
        merged_config = {**self.DEFAULT_CONFIG, **config}
        super().__init__(merged_config)

        self.prompthmr_root = self.config['prompthmr_root']
        self.static_cam = self.config['static_cam']
        self.f_mm = self.config['f_mm']
        self.phmr_ckpt = self.config['phmr_ckpt']
        self.phmr_vid_cfg = self.config['phmr_vid_cfg']
        self.phmr_vid_ckpt = self.config['phmr_vid_ckpt']
        self.mask_prompt = self.config['mask_prompt']
        self.kpt_prompt = self.config['kpt_prompt']
        self.img_size = self.config['img_size']
        self.official_video_head = self.config['official_video_head']
        self.use_gt_video_k = self.config['use_gt_video_k']
        self.center_principal_point_after_padding = bool(self.config.get('center_principal_point_after_padding', False))
        self.smooth_bbx = self.config['smooth_bbx']

        # 子步骤缓存
        self.substep_cache_dir = self.config['substep_cache_dir']
        self.substep_cache_use = self.config['substep_cache_use']
        self.substep_cache_overwrite = self.config['substep_cache_overwrite']

        self.phmr_model = None
        self.vid_head = None
        self.smplx_model = None

    def setup(self):
        """初始化 PromptHMR 图像模型和视频头"""
        prompthmr_abs = os.path.abspath(self.prompthmr_root)

        # PromptHMR ships its own hmr4d package (pipeline/gvhmr); it must win over any other hmr4d on sys.path.
        phmr_gvhmr_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr')
        for p in [prompthmr_abs, phmr_gvhmr_path]:
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, prompthmr_abs)
        sys.path.insert(0, phmr_gvhmr_path)

        original_cwd = os.getcwd()
        os.chdir(prompthmr_abs)

        try:
            # 1. 加载 PromptHMR 图像模型
            from prompt_hmr import load_model as load_phmr
            self.logger.info(f"Loading PromptHMR image model from {self.phmr_ckpt}...")
            self.phmr_model = load_phmr(self.phmr_ckpt)
            self.logger.info("PromptHMR image model loaded")

            # 2. 加载 PromptHMR 视频头
            from omegaconf import OmegaConf
            import importlib.util

            # 直接加载 gvhmr_pl_demo 模块，绕过 pipeline/__init__.py（避免触发 droid_slam 导入）
            # 清除可能已缓存的其它 hmr4d 模块，确保从 PromptHMR 版本重新加载
            hmr4d_modules = [k for k in sys.modules if k.startswith('hmr4d')]
            for mod_name in hmr4d_modules:
                del sys.modules[mod_name]

            demo_pl_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr', 'hmr4d', 'model', 'gvhmr', 'gvhmr_pl_demo.py')
            spec = importlib.util.spec_from_file_location("gvhmr_pl_demo", demo_pl_path)
            demo_pl_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo_pl_module)
            DemoPL = demo_pl_module.DemoPL

            self.logger.info(f"Loading PromptHMR video head from {self.phmr_vid_ckpt}...")
            phmr_vid_cfg = OmegaConf.load(self.phmr_vid_cfg)
            self.vid_head = DemoPL(
                pipeline=phmr_vid_cfg.model.pipeline,
                smplx_path='data/body_models/smplx/SMPLX_NEUTRAL.npz',
            )
            self.vid_head = self.vid_head.eval().cuda()
            self.vid_head.load_pretrained_model(self.phmr_vid_ckpt)
            self.logger.info("PromptHMR video head loaded")
        finally:
            os.chdir(original_cwd)

        # 3. 加载 smplx 模型（用于 get_skeleton 计算 offset）
        from lib.gvhmr.body_model import make_smplx
        self.smplx_model = make_smplx("supermotion").cuda()

        self._is_setup = True
        self.logger.info("PromptHMR backend initialized")

    # === 子步骤缓存 ===

    def _compute_cache_key(self, image_paths: List[str], bboxes: np.ndarray,
                           K_override: np.ndarray = None) -> str:
        """根据图像路径列表、bbox 和可选的 K 矩阵计算缓存 key（MD5 hash）"""
        h = hashlib.md5()
        # 用图像文件名（不含目录）+ 帧数 作为主要标识
        h.update(str(len(image_paths)).encode())
        for p in image_paths:
            h.update(os.path.basename(p).encode())
        # bbox 内容也参与 hash，防止同一视频不同检测框混淆
        h.update(bboxes.tobytes())
        # K 矩阵参与 hash，防止 estimate_K 和 GT K 的缓存混淆
        if K_override is not None:
            if isinstance(K_override, torch.Tensor):
                h.update(K_override.cpu().numpy().tobytes())
            else:
                h.update(K_override.tobytes())
        return h.hexdigest()[:16]

    def _get_substep_cache_path(self, cache_key: str, substep_name: str) -> Optional[str]:
        """获取子步骤缓存文件路径，如果 cache_dir 未设置则返回 None"""
        if self.substep_cache_dir is None:
            return None
        cache_dir = os.path.join(self.substep_cache_dir, 'substep_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{cache_key}_{substep_name}.pt")

    def _load_substep_cache(self, cache_key: str, substep_name: str) -> Optional[torch.Tensor]:
        """尝试加载子步骤缓存，返回 tensor 或 None"""
        if not self.substep_cache_use or self.substep_cache_overwrite:
            return None
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            data = torch.load(cache_path, map_location='cpu', weights_only=True)
            self.logger.info(f"Loaded {substep_name} cache: {cache_path} (shape: {data.shape})")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load {substep_name} cache: {e}")
            return None

    def _save_substep_cache(self, cache_key: str, substep_name: str, data: torch.Tensor):
        """保存子步骤缓存到磁盘"""
        cache_path = self._get_substep_cache_path(cache_key, substep_name)
        if cache_path is None:
            return
        try:
            torch.save(data, cache_path)
            self.logger.info(f"Saved {substep_name} cache: {cache_path} (shape: {data.shape})")
        except Exception as e:
            self.logger.warning(f"Failed to save {substep_name} cache: {e}")

    def _run_image_model(self, image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg,
                         return_smpl=False):
        """
        运行 PromptHMR 图像模型，提取每帧的 2 通道特征 (smpl_token, loc_token)

        Args:
            image_paths: 图像路径列表
            bboxes_xyxy: (N, 4) xyxy 格式边界框
            vitpose_kp2d: (N, 17/25, 3) VitPose 关键点
            K_fullimg: (N, 3, 3) 相机内参
            return_smpl: 如果 True，同时返回图像模型的单帧 SMPL 结果
                         (rotmat, betas, transl, smpl_vertices, smpl_j3d)

        Returns:
            features: (N, 2, 1024) PromptHMR 图像特征
            如果 return_smpl=True，额外返回 dict:
                'rotmat': (N, 22, 3, 3), 'betas': (N, 10), 'transl': (N, 3),
                'smpl_vertices': (N, 6890, 3), 'smpl_j3d': (N, 24, 3)
        """
        import cv2
        from PIL import Image, ImageOps
        from torchvision.transforms import Normalize, ToTensor, Compose

        normalization = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

        IMG_SIZE = self.img_size
        CENTER_PP_AFTER_PADDING = bool(self.center_principal_point_after_padding)
        N = len(image_paths)
        all_features = []
        # 用于保存图像模型的单帧 SMPL 输出（当 return_smpl=True 时）
        all_img_rotmat = [] if return_smpl else None
        all_img_betas = [] if return_smpl else None
        all_img_transl = [] if return_smpl else None
        all_img_smpl_verts = [] if return_smpl else None

        self.logger.info(f"Running PromptHMR image model on {N} frames (return_smpl={return_smpl})...")

        from torch.utils.data import Dataset, DataLoader

        class SingleFrameDataset(Dataset):
            def __init__(ds_self, image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg, img_size):
                ds_self.image_paths = image_paths
                ds_self.bboxes_xyxy = bboxes_xyxy
                ds_self.vitpose_kp2d = vitpose_kp2d
                ds_self.K_fullimg = K_fullimg
                ds_self.img_size = img_size
                ds_self.normalization = normalization

            def __len__(ds_self):
                return len(ds_self.image_paths)

            def __getitem__(ds_self, idx):
                img = cv2.imread(ds_self.image_paths[idx])
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {ds_self.image_paths[idx]}")
                img = img[..., ::-1].copy()  # BGR -> RGB

                bbox = ds_self.bboxes_xyxy[idx]  # (4,)
                boxes = torch.from_numpy(bbox).float().unsqueeze(0)  # (1, 4)
                boxes = torch.cat([boxes, torch.ones(1, 1)], dim=-1)  # (1, 5)

                # 关键点 (取前 25 个 joints，如果有)
                # 注意：必须用 .copy() 断开与原始数组的内存共享，
                # 否则后续 inplace 缩放操作会污染原始 vitpose_kp2d
                kpt = ds_self.vitpose_kp2d[idx]  # (J, 3)
                if isinstance(kpt, np.ndarray):
                    kpt = torch.from_numpy(kpt.copy()).float()
                else:
                    kpt = kpt.clone().float()
                kpt = kpt.unsqueeze(0)  # (1, J, 3)

                cam_int = ds_self.K_fullimg[idx]  # (3, 3)
                if isinstance(cam_int, np.ndarray):
                    cam_int = torch.from_numpy(cam_int.copy()).float()
                else:
                    cam_int = cam_int.clone().float()
                cam_int = cam_int.unsqueeze(0)  # (1, 3, 3)

                # pad image to IMG_SIZE
                size = np.array([img.shape[1], img.shape[0]])
                scale = ds_self.img_size / max(size)
                offset = (ds_self.img_size - scale * size) / 2

                img_pil = Image.fromarray(img)
                img_pil = ImageOps.contain(img_pil, (ds_self.img_size, ds_self.img_size))
                img_pil = ImageOps.pad(img_pil, size=(ds_self.img_size, ds_self.img_size))
                img_np = np.array(img_pil)

                # 调整 cam_int
                cam_int = cam_int.mean(dim=0, keepdim=True)
                cam_int[:, :2] *= scale
                cam_int[:, :2, -1] += torch.tensor(offset).float()
                if CENTER_PP_AFTER_PADDING:
                    cam_int[:, 0, 2] = ds_self.img_size / 2.0
                    cam_int[:, 1, 2] = ds_self.img_size / 2.0

                # 调整 boxes
                boxes[:, :4] *= scale
                boxes[:, :2] += torch.tensor(offset).float()
                boxes[:, 2:4] += torch.tensor(offset).float()

                # 调整 keypoints
                kpt[:, :, :2] *= scale
                kpt[:, :, :2] += torch.tensor(offset).float()

                image_tensor = ds_self.normalization(img_np)

                item = {
                    'image': image_tensor,
                    'image_cv': torch.tensor(img_np),
                    'boxes': boxes,
                    'kpts': kpt,
                    'cam_int': cam_int,
                    'masks': None,
                }
                return item

        dataset = SingleFrameDataset(image_paths, bboxes_xyxy, vitpose_kp2d, K_fullimg, IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                                collate_fn=lambda x: x)

        from tqdm import tqdm
        from torch.amp import autocast

        for batch in tqdm(dataloader, desc="PromptHMR image model"):
            with torch.no_grad(), autocast('cuda'):
                output = self.phmr_model(
                    batch,
                    box_prompt=True,
                    mask_prompt=self.mask_prompt,
                    kpt_prompt=self.kpt_prompt,
                )

            for bid in range(len(batch)):
                # output[bid]['features'] 形状: (num_persons, 2, 1024)
                # 由于每帧只有 1 个人, 取 [0]
                feat = output[bid]['features'][0]  # (2, 1024)
                all_features.append(feat.cpu())

                if return_smpl:
                    all_img_rotmat.append(output[bid]['rotmat'][0].cpu())            # (22, 3, 3)
                    all_img_betas.append(output[bid]['betas'][0].cpu())              # (10,)
                    all_img_transl.append(output[bid]['transl'][0].cpu())            # (3,)
                    all_img_smpl_verts.append(output[bid]['smpl_vertices'][0].cpu()) # (6890, 3)

        features = torch.stack(all_features)  # (N, 2, 1024)
        self.logger.info(f"PromptHMR image features extracted: {features.shape}")

        if return_smpl:
            j_regressor = self.phmr_model.smpl.J_regressor[:24].cpu()  # (24, 6890)
            smpl_verts = torch.stack(all_img_smpl_verts)               # (N, 6890, 3)
            img_smpl = {
                'rotmat': torch.stack(all_img_rotmat),       # (N, 22, 3, 3)
                'betas': torch.stack(all_img_betas),         # (N, 10)
                'transl': torch.stack(all_img_transl),       # (N, 3)
                'smpl_vertices': smpl_verts,                 # (N, 6890, 3)
                'smpl_j3d': torch.matmul(j_regressor, smpl_verts),  # (N, 24, 3)
            }
            self.logger.info(
                f"PromptHMR image-model SMPL saved: "
                f"j3d={img_smpl['smpl_j3d'].shape}, rotmat={img_smpl['rotmat'].shape}"
            )
            return features, img_smpl

        return features

    def _pad_video_inputs(self, image_paths, bboxes_xyxy, kp2d, K_fullimg):
        """Pad prompts to 896x896 exactly like eval_phmr.py::_pad_image_and_prompts."""
        import cv2
        N = len(image_paths)
        boxes_padded, kp2d_padded, K_padded = [], [], []
        if isinstance(K_fullimg, np.ndarray):
            K_base = torch.from_numpy(K_fullimg).float()
        else:
            K_base = K_fullimg.float().clone()
        if K_base.ndim == 2:
            K_base = K_base.unsqueeze(0).repeat(N, 1, 1)
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load image: {path}")
            h, w = img.shape[:2]
            size = np.array([w, h], dtype=np.float32)
            scale = self.img_size / max(size)
            offset = torch.tensor((self.img_size - scale * size) / 2, dtype=torch.float32)

            box = torch.as_tensor(bboxes_xyxy[i]).float().clone()
            box[:4] *= scale
            box[:2] += offset
            box[2:4] += offset

            kpt = torch.as_tensor(kp2d[i]).float().clone()
            kpt[:, :2] *= scale
            kpt[:, :2] += offset

            K = K_base[i].clone()
            K[:2] *= scale
            K[:2, -1] += offset

            boxes_padded.append(box[:4])
            kp2d_padded.append(kpt)
            K_padded.append(K)
        return torch.stack(boxes_padded), torch.stack(kp2d_padded), torch.stack(K_padded)

    def estimate_smpl(
        self,
        image_paths: List[str],
        bboxes: np.ndarray,
        img_focal: float = None,
        img_center: np.ndarray = None,
        K_fullimg_override: np.ndarray = None,
        cam_R_w2c_override: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        运行 PromptHMR 推理

        Args:
            image_paths: 图像路径列表
            bboxes: 边界框 [N, 4] 或 [N, K, 5]
            img_focal: 图像焦距（可选）
            img_center: 图像主点（可选）
            K_fullimg_override: 完整的 3x3 相机内参矩阵（可选）
            cam_R_w2c_override: first-pass SLAM world-to-camera rotation，用于 cam_angvel

        Returns:
            包含 PromptHMR 全部输出的字典（格式与 GVHMR 一致）
        """
        self.ensure_setup()

        prompthmr_abs = os.path.abspath(self.prompthmr_root)
        phmr_gvhmr_path = os.path.join(prompthmr_abs, 'pipeline', 'gvhmr')

        # PromptHMR's own hmr4d package first (see setup())
        for p in [prompthmr_abs, phmr_gvhmr_path]:
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, prompthmr_abs)
        sys.path.insert(0, phmr_gvhmr_path)

        # 将图像路径转为绝对路径
        image_paths = [os.path.abspath(p) for p in image_paths]

        original_cwd = os.getcwd()

        try:
            import cv2
            import scipy.signal as signal
            from scipy.ndimage import gaussian_filter1d
            from torch.amp import autocast

            N = len(image_paths)
            first_img = cv2.imread(image_paths[0])
            height, width = first_img.shape[:2]
            self.logger.info(f"PromptHMR processing {N} frames, image size: {width}x{height}")

            # --- 1. 准备 bboxes ---
            if bboxes.ndim == 3:
                bboxes_xyxy = bboxes[:, 0, :4]
            elif bboxes.ndim == 2 and bboxes.shape[1] > 4:
                bboxes_xyxy = bboxes[:, :4]
            else:
                bboxes_xyxy = bboxes

            # 计算缓存 key
            # VitPose 缓存不依赖 K，用基础 key（图像路径 + bbox）
            cache_key = self._compute_cache_key(image_paths, bboxes_xyxy)
            # 图像模型缓存依赖 K（不同的 K 产生不同的特征），需要额外加入 K_fullimg_override
            cache_key_img = self._compute_cache_key(image_paths, bboxes_xyxy, K_fullimg_override)

            # hmr4d.utils.preproc must come from PromptHMR's hmr4d; drop any other cached copy.
            _preproc_mod = sys.modules.get('hmr4d.utils.preproc')
            if _preproc_mod is not None and phmr_gvhmr_path not in (getattr(_preproc_mod, '__file__', '') or ''):
                for k in [k for k in sys.modules if k.startswith('hmr4d.utils.preproc')]:
                    del sys.modules[k]

            # 使用本地 shim：PromptHMR 自带的 25 关节 ViTPose（vitpose-h-coco_25.pth），
            # 避免依赖 GVHMR 约定路径里那个 17 关节 vitpose-h-multi-coco.pth（未下载）。
            # shim 内部会调用 PromptHMR 的 VitInference，再把 coco_25 映射为 coco_17。
            from lib.pipeline.backends.hpe.vitpose_compat import (
                VitPoseExtractorCompat as VitPoseExtractor,
            )
            from hmr4d.utils.geo.hmr_cam import (
                get_bbx_xys_from_xyxy,
                estimate_K,
                create_camera_sensor,
                normalize_kp2d,
            )
            from hmr4d.utils.geo_transform import compute_cam_angvel
            from hmr4d.utils.net_utils import detach_to_cpu

            bbx_xyxy = torch.from_numpy(bboxes_xyxy).float()
            bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

            # --- 2-3. 2D keypoints：ViTPose（从不使用 annotation 关键点）---
            from tqdm import tqdm
            vitpose = self._load_substep_cache(cache_key, 'vitpose')
            if vitpose is not None:
                self.logger.info("VitPose loaded from cache, skipping inference")
            else:
                self.logger.info(f"Loading {N} images for VitPose...")
                imgs_list = []
                for p in tqdm(image_paths, desc="Loading images", leave=False):
                    img = cv2.imread(p)
                    if img is None:
                        raise FileNotFoundError(f"Failed to load image: {p}")
                    imgs_list.append(img[..., ::-1])  # BGR → RGB
                imgs_np = np.stack(imgs_list)
                del imgs_list

                self.logger.info("Running VitPose...")
                vitpose_extractor = VitPoseExtractor(
                    tqdm_leave=False,
                    prompthmr_root=prompthmr_abs,
                )
                # VitPoseExtractorCompat wraps PromptHMR's inference_wbbox, which expects
                # full-resolution RGB images and bbox coordinates in the same image space.
                # Do not pass GVHMR get_batch() crops here: get_batch returns 256x256 crops
                # but keeps bbx_xys in full-image coordinates, which can create empty crops.
                vitpose = vitpose_extractor.extract(imgs_np, bbx_xys)
                del vitpose_extractor, imgs_np
                torch.cuda.empty_cache()
                self.logger.info(f"VitPose complete: {vitpose.shape}")

                # 保存 VitPose 缓存
                self._save_substep_cache(cache_key, 'vitpose', vitpose)

            # --- 4. 构建相机内参 ---
            # estimate_K 只作为 fallback；当 use_gt_video_k=True 且有 annotation K 时，
            # official video head 实际使用 GT K 经 896 padding 后的 K。
            K_fullimg_vid = estimate_K(width, height).repeat(N, 1, 1)
            self.logger.info(
                f"Fallback estimate_K: fx={float(K_fullimg_vid[0, 0, 0]):.2f}, "
                f"cx={float(K_fullimg_vid[0, 0, 2]):.2f}, cy={float(K_fullimg_vid[0, 1, 2]):.2f}"
            )

            # 图像模型的 K：优先用传入的 GT K（CameraEncoder + decode_transl 需要准确的 K），
            # 若未提供则退回到 estimate_K
            if K_fullimg_override is not None:
                K_np = K_fullimg_override
                if isinstance(K_np, np.ndarray):
                    K_np = torch.from_numpy(K_np).float()
                if K_np.ndim == 2:
                    K_fullimg_img = K_np.unsqueeze(0).repeat(N, 1, 1)
                elif K_np.shape[0] == N:
                    K_fullimg_img = K_np.float()
                else:
                    K_fullimg_img = K_np[:1].repeat(N, 1, 1)
                self.logger.info(
                    f"Image model K (GT override): fx={float(K_fullimg_img[0, 0, 0]):.2f}, "
                    f"cx={float(K_fullimg_img[0, 0, 2]):.2f}, cy={float(K_fullimg_img[0, 1, 2]):.2f}"
                )
            elif img_focal is not None:
                from hmr4d.utils.geo.hmr_cam import convert_f_to_K
                K_fullimg_img = convert_f_to_K(img_focal, width, height).repeat(N, 1, 1)
                self.logger.info(f"Image model K (from img_focal={img_focal})")
            elif self.f_mm is not None:
                K_fullimg_img = create_camera_sensor(width, height, self.f_mm)[2].repeat(N, 1, 1)
                self.logger.info(f"Image model K (from f_mm={self.f_mm})")
            else:
                K_fullimg_img = K_fullimg_vid.clone()
                self.logger.info("Image model K (estimate_K, same as video head)")

            # --- 5. 运行 PromptHMR 图像模型提取特征 + 单帧 SMPL（带缓存） ---
            phmr_features = self._load_substep_cache(cache_key_img, 'phmr_features')
            phmr_img_smpl = self._load_substep_cache(cache_key_img, 'phmr_img_smpl')
            if phmr_features is not None:
                self.logger.info("PromptHMR image features loaded from cache, skipping inference")
                if phmr_img_smpl is not None:
                    self.logger.info("PromptHMR image-model SMPL loaded from cache")
            else:
                # 先将视频头卸载到 CPU 以腾出显存给图像模型（DINOv2 backbone 很大）
                self.vid_head.cpu()
                self.smplx_model.cpu()
                torch.cuda.empty_cache()
                # 确保图像模型在 GPU 上（处理上一个序列后可能被卸到了 CPU）
                self.phmr_model.cuda()

                os.chdir(prompthmr_abs)
                result = self._run_image_model(
                    image_paths, bboxes_xyxy, vitpose.numpy(), K_fullimg_img.numpy(),
                    return_smpl=True,
                )
                phmr_features, phmr_img_smpl = result  # (N, 2, 1024), dict
                torch.cuda.empty_cache()

                # 保存 Image Model 缓存
                self._save_substep_cache(cache_key_img, 'phmr_features', phmr_features)
                self._save_substep_cache(cache_key_img, 'phmr_img_smpl', phmr_img_smpl)

                # 图像特征提取完毕，将图像模型卸载到 CPU
                self.phmr_model.cpu()
                torch.cuda.empty_cache()

            # 确保视频头和 smplx 在 GPU 上
            self.vid_head.cuda()
            self.smplx_model.cuda()

            # --- 6. 相机旋转：第一遍 SLAM（从不使用 GT 相机），否则 static ---
            if cam_R_w2c_override is not None:
                R_w2c = torch.as_tensor(cam_R_w2c_override).float()
                if R_w2c.ndim == 4:
                    R_w2c = R_w2c[:, :3, :3]
                if R_w2c.shape[0] != N:
                    self.logger.warning(
                        f"cam_R_w2c_override length mismatch ({R_w2c.shape[0]} vs {N}); falling back to static camera"
                    )
                    R_w2c = torch.eye(3).repeat(N, 1, 1)
                else:
                    self.logger.info("Using first-pass SLAM camera rotation for PromptHMR video head cam_angvel")
            elif self.static_cam:
                self.logger.info("Using static camera assumption (skipping VO)")
                R_w2c = torch.eye(3).repeat(N, 1, 1)
            else:
                raise RuntimeError('PromptHMR video head needs camera rotations: pass cam_R_w2c_override '
                                   '(first-pass SLAM camera) or set static_cam')

            # --- 7. 准备视频头输入 ---
            self.logger.info("Preparing video head input...")
            if self.official_video_head:
                # Match PromptHMR/scripts/eval_phmr.py: pad boxes/kp2d/K to 896 and use DemoPL.predict(postproc=True).
                K_video_base = K_fullimg_img if (self.use_gt_video_k and K_fullimg_override is not None) else K_fullimg_vid
                boxes_video, kp2d_video, K_video = self._pad_video_inputs(
                    image_paths, bboxes_xyxy, vitpose, K_video_base
                )
                self.logger.info(
                    f"Actual video head K after 896 padding: fx={float(K_video[0,0,0]):.2f}, "
                    f"fy={float(K_video[0,1,1]):.2f}, cx={float(K_video[0,0,2]):.2f}, "
                    f"cy={float(K_video[0,1,2]):.2f}, "
                    f"source={'GT/image K' if (self.use_gt_video_k and K_fullimg_override is not None) else 'estimate_K'}"
                )
                bbx_xys_video = get_bbx_xys_from_xyxy(boxes_video).float()
                cam_angvel = compute_cam_angvel(R_w2c).float()

                self.logger.info("Running PromptHMR video head via DemoPL.predict(postproc=True)...")
                os.chdir(prompthmr_abs)
                pred_raw = self.vid_head.predict({
                    'length': torch.tensor(N),
                    'kp2d': kp2d_video.cuda(),
                    'bbx_xys': bbx_xys_video.cuda(),
                    'K_fullimg': K_video.cuda(),
                    'cam_angvel': cam_angvel.cuda(),
                    'f_imgseq': phmr_features.cuda(),
                }, static_cam=bool(self.static_cam))
                pred = {
                    'smpl_params_global': {k: v.cpu() for k, v in pred_raw['smpl_params_global'].items()},
                    'smpl_params_incam': {k: v.cpu() for k, v in pred_raw['smpl_params_incam'].items()},
                }
                net_outputs = pred_raw['net_outputs']
                static_conf_logits = net_outputs.get('static_conf_logits', None)
                if static_conf_logits is None:
                    static_conf_logits = net_outputs['model_output']['static_conf_logits']
                static_conf_logits = static_conf_logits[0].cpu() if static_conf_logits.ndim == 3 else static_conf_logits.cpu()
            else:
                bbx_xys_video = bbx_xys.float()
                if self.smooth_bbx:
                    bbx_xys_np = bbx_xys.numpy()
                    smoothed = np.array([signal.medfilt(param, 11) for param in bbx_xys_np.T]).T
                    bbx_xys_video = np.array([gaussian_filter1d(traj, 3) for traj in smoothed.T]).T
                    bbx_xys_video = torch.from_numpy(bbx_xys_video).float()

                K_video = K_fullimg_vid
                cam_angvel = compute_cam_angvel(R_w2c)
                vitpose_norm = normalize_kp2d(vitpose, bbx_xys_video).float()
                batch = {
                    "length": torch.tensor([N]),
                    "obs": vitpose_norm[None],
                    "bbx_xys": bbx_xys_video[None],
                    "K_fullimg": K_video[None],
                    "cam_angvel": cam_angvel[None],
                    "f_imgseq": phmr_features[None],
                }
                batch = {k: v.cuda() for k, v in batch.items()}

                self.logger.info("Running PromptHMR video head inference (legacy two-forward mode)...")
                os.chdir(prompthmr_abs)

                with torch.no_grad(), autocast('cuda'):
                    output_w_kpts = self.vid_head.pipeline.forward(
                        batch, train=False, postproc=False, static_cam=bool(self.static_cam)
                    )
                batch_no_kpts = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_no_kpts['obs'] = torch.zeros_like(batch['obs'])
                with torch.no_grad(), autocast('cuda'):
                    output_no_kpts = self.vid_head.pipeline.forward(
                        batch_no_kpts, train=False, postproc=False, static_cam=bool(self.static_cam)
                    )

                pred = {
                    "smpl_params_global": {
                        "body_pose": output_no_kpts["pred_smpl_params_global"]["body_pose"][0].cpu(),
                        "betas": output_no_kpts["pred_smpl_params_global"]["betas"][0].cpu(),
                        "global_orient": output_no_kpts["pred_smpl_params_global"]["global_orient"][0].cpu(),
                        "transl": output_w_kpts["pred_smpl_params_global"]["transl"][0].cpu(),
                    },
                    "smpl_params_incam": {
                        "body_pose": output_no_kpts["pred_smpl_params_incam"]["body_pose"][0].cpu(),
                        "betas": output_no_kpts["pred_smpl_params_incam"]["betas"][0].cpu(),
                        "global_orient": output_no_kpts["pred_smpl_params_incam"]["global_orient"][0].cpu(),
                        "transl": output_w_kpts["pred_smpl_params_incam"]["transl"][0].cpu(),
                    },
                }
                static_conf_logits = output_no_kpts["model_output"]["static_conf_logits"][0].cpu()
                kp2d_video = vitpose

            self.logger.info("PromptHMR video head inference complete")

            # --- 8. 计算 skeleton offset ---
            betas_global = pred["smpl_params_global"]["betas"]  # (F, 10)
            skeleton_offset = self.smplx_model.get_skeleton(
                betas_global[0:1].cuda()
            )[0, 0].cpu()

            self.logger.info(
                f"PromptHMR inference complete. "
                f"global orient shape: {pred['smpl_params_global']['global_orient'].shape}, "
                f"skeleton offset: {skeleton_offset}"
            )

            return {
                'smpl_params_global': pred['smpl_params_global'],
                'smpl_params_incam': pred['smpl_params_incam'],
                'static_conf_logits': static_conf_logits,
                'K_fullimg': K_video,
                'skeleton_offset': skeleton_offset,
                'vitpose_kp2d': kp2d_video.cpu() if isinstance(kp2d_video, torch.Tensor) else kp2d_video,
                'phmr_img_smpl': phmr_img_smpl,
            }
        finally:
            os.chdir(original_cwd)

    def cleanup(self):
        """清理资源"""
        self.phmr_model = None
        self.vid_head = None
        self.smplx_model = None
        super().cleanup()

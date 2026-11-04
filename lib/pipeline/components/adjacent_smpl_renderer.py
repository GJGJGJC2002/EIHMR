"""
AdjacentSMPLRenderer - 渲染相邻帧SMPL mesh的组件

这个组件复现adjTram的功能，对每一帧渲染当前帧、前x帧、后x帧的SMPL mesh，
并将它们在当前帧的相机视角下拼接显示。

支持四种渲染模式（render_mode）：
- 'camera'（默认，适合 TRAM）: 使用相机坐标系的 trans (local_trans) 生成
  vertices，再通过 c2w 做帧间相对变换。适合 VIMO 等估计方法产生的相机坐标系参数。
- 'world'（适合 GT）: 直接使用世界坐标系的 trans (global_trans) + 世界坐标系
  root orientation 生成 vertices，再用 GT 的 w2c 投影到当前帧。避免了两种方法
  对世界坐标系 Y 轴约定不同导致的渲染偏移问题。
- 'gvhmr_world'（适合 GVHMR）: 从 GVHMR 的 global+incam 两套输出反推每帧
  T_w2c 变换矩阵，用修正后的全局 vertices 投影到当前帧相机视角下。
  完全不需要外部 SLAM 的相机参数。
- 'prompthmr_camera'（适合 PromptHMR + SLAM）: 使用 PromptHMR 的相机坐标系参数
  (global_orient_c + body_pose_aa + trans) 通过 SMPL-X 生成 incam vertices，
  再用 DROID-SLAM 的 c2w 做帧间相对变换投影到中心帧视角。
  使用 GVHMRRenderer + 完整 K 矩阵渲染。
"""

import os
import cv2
import colorsys
import numpy as np
import torch
import shutil
from typing import Dict, Any
from tqdm import tqdm

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData
from lib.models.smpl import SMPL
from pytorch3d.renderer import TexturesVertex, Materials
from pytorch3d.structures import Meshes


class AdjacentSMPLRenderer(Component):
    """
    渲染相邻帧SMPL mesh的组件

    对于每一帧，渲染当前帧、前x帧、后x帧的SMPL mesh，
    在当前帧的相机视角下显示，用于可视化和调试。

    Config:
        pre_dis: 相邻帧间隔（默认20）
        device: 计算设备（默认'cpu'）
        output_dir: 输出目录（默认'results/adjacent_smpl'）
        render_mode: 渲染模式 ('camera' 或 'world' 或 'gvhmr_world' 或 'prompthmr_camera')
            - 'camera': TRAM 模式，使用相机坐标系 trans + c2w 帧间变换
            - 'world': GT 模式，使用世界坐标系 trans + w2c 投影
            - 'gvhmr_world': GVHMR 模式，反推 T_w2c 投影
            - 'prompthmr_camera': PromptHMR + SLAM 模式，incam vertices + SLAM c2w 帧间变换
    """

    COMPONENT_TYPE = "adjacent_smpl_renderer"

    DEFAULT_CONFIG = {
        'pre_dis': 20,  # 相邻帧间隔
        'num_adjacent_frames': 1,  # 前后渲染的帧数（渲染 i-x*pre_dis 到 i+x*pre_dis）
        'device': 'cpu',
        'output_dir': 'results/adjacent_smpl',
        'render_mode': 'camera',  # 'camera' (TRAM), 'world' (GT), 'gvhmr_world' (GVHMR), 'prompthmr_camera' (PromptHMR + SLAM c2w)
        # prompthmr_camera 模式下 mesh 平移的来源：
        #   'sequence'   : 序列级 smpl_params.trans（跨窗口高斯融合后的结果）
        #   'oar_window' : depth_scene_refine 每个窗口自己的局部 transl/body_pose，
        #                  按窗口中心帧渲染到该中心的相机视角。要求 pre_dis 与
        #                  depth_scene_refine.stride 一致，否则中心帧对不上。
        'transl_source': 'sequence',
        'frame_selection_mode': 'uniform',  # 'uniform' (均匀 pre_dis 采样) 或 'keyframe' (使用 SLAM 关键帧)
        'render_frame_indices': None,
        'render_only_keyframes': False,  # keyframe 模式下是否只渲染关键帧（跳过非关键帧以节省时间）
        'mesh_color_mode': 'colorful',  # 'colorful' (每帧不同彩色) 或 'uniform' (统一灰白色)
        'uniform_mesh_color': [0.75, 0.75, 0.75],  # mesh_color_mode='uniform' 时使用的 RGB 颜色 [0-1]
        'mesh_render_style': 'solid',  # 'solid' (实心填充) 或 'contour' (只画轮廓边缘，保留背景纹理)
        'mesh_alpha': 1.0,  # solid 模式下 mesh 透明度；1.0=完全覆盖，0.5=半透明融合
        'smpl_trans_scale': 1.0,  # 渲染前缩放 SMPL 相机/世界平移；>1 使人物投影变小
        'prompthmr_rescale_transl_to_render_k': True,  # PHMR transl_z 按 cache/render focal 比例重解码
        'contour_thickness': 3,  # contour 模式下轮廓线宽度（像素），值越大轮廓越粗
        # Export local adjacent-SMPL projection factors for post-SLAM camera refinement.
        'export_projection_factors': False,
        'projection_factor_point_type': 'joints24',  # currently supports 'joints24'
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

        self.pre_dis = self.config['pre_dis']
        self.num_adjacent_frames = self.config['num_adjacent_frames']
        self.device = self.config['device']
        self.output_dir = self.config['output_dir']
        self.render_mode = self.config['render_mode']
        self.transl_source = self.config.get('transl_source', 'sequence')
        self.frame_selection_mode = self.config['frame_selection_mode']
        self.render_only_keyframes = self.config['render_only_keyframes']
        self.mesh_color_mode = self.config['mesh_color_mode']
        self.uniform_mesh_color = self.config['uniform_mesh_color']
        self.mesh_render_style = self.config.get('mesh_render_style', 'solid')
        self.mesh_alpha = float(self.config.get('mesh_alpha', 1.0))
        self.smpl_trans_scale = float(self.config.get('smpl_trans_scale', 1.0))
        self.prompthmr_rescale_transl_to_render_k = bool(self.config.get('prompthmr_rescale_transl_to_render_k', True))
        self.contour_thickness = self.config.get('contour_thickness', 3)
        self.export_projection_factors = self.config.get('export_projection_factors', False)
        self.projection_factor_point_type = self.config.get('projection_factor_point_type', 'joints24')

        # SMPL模型和faces
        self.smpl_model = None
        self.smpl_faces = None

        # 颜色缓存（为每一帧分配的颜色）
        self.frame_colors = None

    def _get_render_frame_indices(self, num_frames: int):
        indices = self.config.get('render_frame_indices')
        if indices is None:
            return list(range(0, num_frames, self.pre_dis))
        if self.render_mode != 'prompthmr_camera' or self.frame_selection_mode != 'uniform':
            raise ValueError('render_frame_indices requires uniform prompthmr_camera mode')
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices):
            raise ValueError('render_frame_indices must contain integer timestamps')
        indices = sorted(set(int(i) for i in indices))
        if any(i < 0 or i >= num_frames for i in indices):
            raise ValueError('render_frame_indices contains an out-of-range timestamp')
        return indices

    def _generate_frame_colors(self, num_frames: int, window_size: int = 100):
        """为每一帧生成独特的颜色

        使用HSV颜色空间，确保相邻帧颜色不同。
        如果总帧数超过window_size，则周期性重复颜色，
        但保证局部窗口内不会重复。

        Args:
            num_frames: 总帧数
            window_size: 颜色不重复的窗口大小，默认100

        Returns:
            List of RGB colors, each in [0, 1] range
        """
        colors = []
        for i in range(num_frames):
            # 使用黄金角度 (~137.5度) 确保相邻颜色在HSV空间中分布均匀
            # 黄金角度 = 2π * (1 - 1/φ) ≈ 2.39996 弧度 ≈ 137.5度
            golden_angle = 2.3999631
            hue = (i * golden_angle / (2 * np.pi)) % 1.0

            # 饱和度和亮度保持较高，确保颜色鲜艳且可见
            saturation = 0.8
            value = 0.9

            # 转换为RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)

        return colors

    def _get_mesh_color(self, frame_idx: int):
        """根据 mesh_color_mode 配置获取渲染颜色

        Returns:
            RGB color as list, each in [0, 1] range
        """
        if self.mesh_color_mode == 'uniform':
            return self.uniform_mesh_color
        else:
            return self.frame_colors[frame_idx]

    def _render_solid(self, renderer, vertices, background, colors):
        """渲染 solid mesh，并按 mesh_alpha 与当前背景融合后保存。"""
        rendered = renderer.render_mesh(vertices, background, colors=colors)
        alpha = float(np.clip(self.mesh_alpha, 0.0, 1.0))
        if alpha >= 1.0:
            return rendered
        if alpha <= 0.0:
            return background.copy()
        return cv2.addWeighted(background, 1.0 - alpha, rendered, alpha, 0.0)

    def _render_contour(self, renderer, vertices, background, colors, thickness=3):
        """渲染 mesh 的轮廓边缘（不填充实心区域），保留背景纹理

        原理：
        1. 用标准 render_mesh 渲染获得实心 mask
        2. 对 mask 做膨胀-腐蚀差分提取边缘轮廓
        3. 只在边缘像素位置用指定颜色覆盖原始背景

        Args:
            renderer: GVHMRRenderer 实例
            vertices: (V, 3) 相机坐标系下的 mesh vertices
            background: (H, W, 3) BGR 背景图像
            colors: RGB 颜色 [r, g, b]，范围 [0, 1]
            thickness: 轮廓线宽度（像素）

        Returns:
            叠加了轮廓线的背景图像
        """
        # Step 1: 渲染实心 mesh 获取 silhouette mask
        renderer.update_bbox(vertices[::50], scale=1.2)
        verts = vertices.unsqueeze(0)

        colors_normalized = colors
        if isinstance(colors_normalized, list) and len(colors_normalized) > 0 and colors_normalized[0] > 1:
            colors_normalized = [c / 255.0 for c in colors_normalized]

        verts_features = torch.tensor(colors_normalized).reshape(1, 1, 3).to(
            device=verts.device, dtype=verts.dtype)
        verts_features = verts_features.repeat(1, verts.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(verts=verts, faces=renderer.faces, textures=textures)
        materials = Materials(device=renderer.device,
                              specular_color=(colors_normalized,), shininess=0)

        results = torch.flip(
            renderer.renderer(mesh, materials=materials,
                              cameras=renderer.cameras, lights=renderer.lights),
            [1, 2]
        )

        # 获取 mask 和 bbox
        mask_full = (results[0, ..., -1] > 1e-3).cpu().numpy().astype(np.uint8)
        bbox = renderer.bboxes[0].int().cpu().numpy()
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Step 2: 在 bbox 区域内提取轮廓（膨胀 - 腐蚀 = 边缘带）
        mask_roi = mask_full[:y2 - y1, :x2 - x1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
        dilated = cv2.dilate(mask_roi, kernel, iterations=1)
        eroded = cv2.erode(mask_roi, kernel, iterations=1)
        contour_mask = ((dilated - eroded) > 0)

        # Step 3: 在轮廓位置叠加颜色到背景上
        output = background.copy()
        roi = output[y1:y2, x1:x2]

        # RGB [0,1] -> BGR [0,255]
        color_bgr = [int(colors_normalized[2] * 255),
                     int(colors_normalized[1] * 255),
                     int(colors_normalized[0] * 255)]

        h_roi, w_roi = roi.shape[:2]
        h_mask, w_mask = contour_mask.shape[:2]
        h_use = min(h_roi, h_mask)
        w_use = min(w_roi, w_mask)

        contour_crop = contour_mask[:h_use, :w_use]
        roi[:h_use, :w_use][contour_crop] = color_bgr
        output[y1:y2, x1:x2] = roi

        renderer.reset_bbox()
        return output

    def validate_input(self, data: PipelineData) -> bool:
        """验证输入"""
        # 需要SMPL参数
        has_smpl = data.smpl_params is not None
        if not has_smpl:
            self.logger.warning("No SMPL params found in data")
            return False

        # 需要图像路径
        has_images = hasattr(data, 'image_paths') or data.metadata.get('image_paths')
        if not has_images:
            self.logger.warning("No image paths found in data")
            return False

        # gvhmr_world 模式不需要外部 camera_params（自行从 GVHMR 输出反推）
        if self.render_mode == 'gvhmr_world':
            if data.smpl_params.global_orient_w is None or data.smpl_params.global_orient_c is None:
                self.logger.warning("gvhmr_world mode requires GVHMR fields in smpl_params")
                return False
            return True

        # 其他模式需要相机参数
        if data.camera_params is None:
            self.logger.warning("No camera_params found in data")
            return False

        return True

    def setup(self):
        """初始化SMPL模型"""
        self.smpl_model = SMPL(
            gender='neutral',
            model_type='smpl',
        ).to(self.device)

        self.smpl_faces = self.smpl_model.faces
        self.logger.info(f"SMPL faces shape: {self.smpl_faces.shape}")

        self._is_setup = True
        self.logger.info("Adjacent SMPL Renderer component initialized")

    def cleanup(self):
        """释放 SMPL 模型资源"""
        self.smpl_model = None
        self.smpl_faces = None
        self.frame_colors = None
        super().cleanup()

    def execute(self, data: PipelineData) -> PipelineData:
        """执行渲染，根据 render_mode 和 frame_selection_mode 选择不同的渲染策略"""
        # 检查缓存配置
        cache_config = self.config.get('cache', {})
        cache_enabled = cache_config.get('enabled', False)
        use_cache = cache_config.get('use_cache', cache_enabled)
        overwrite = cache_config.get('overwrite', False)
        
        output_dir = self._get_output_dir(data)
        
        # 如果 use_cache=True 且输出目录存在且有图像文件，直接使用缓存
        if use_cache and not overwrite and os.path.exists(output_dir):
            existing_images = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            explicit_indices = self.config.get('render_frame_indices')
            coverage_ok = True
            if explicit_indices is not None:
                requested = self._get_render_frame_indices(len(data.image_paths))
                coverage_ok = all(
                    os.path.basename(data.image_paths[i]) in existing_images for i in requested
                )
            if len(existing_images) > 0 and coverage_ok:
                self.logger.info(f"Using cached rendered images from {output_dir}")
                self.logger.info(f"  Found {len(existing_images)} cached images")
                
                # 更新 metadata 以供后续组件使用
                self._update_metadata_from_cache(data, output_dir, existing_images)
                return data
        
        # 如果 overwrite=True，清空输出目录
        if overwrite and os.path.exists(output_dir):
            self.logger.info(f"Overwrite enabled, clearing {output_dir}")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # 执行渲染
        if self.frame_selection_mode == 'keyframe':
            return self._execute_keyframe_mode(data)
        elif self.render_mode == 'gvhmr_world':
            return self._execute_gvhmr_world_mode(data)
        elif self.render_mode == 'prompthmr_camera':
            return self._execute_prompthmr_camera_mode(data)
        elif self.render_mode == 'world':
            return self._execute_world_mode(data)
        else:
            return self._execute_camera_mode(data)

    def _execute_camera_mode(self, data: PipelineData) -> PipelineData:
        """相机坐标系模式渲染（适合 TRAM/VIMO）

        策略：在相机坐标系下生成每帧的 vertices，然后通过帧间相对变换
        （cam_j -> world -> cam_i）将其他帧的 vertices 投影到当前帧的相机坐标系下渲染。

        这种方式使用 c2w 做帧间相对变换，世界坐标系约定会被消掉。
        适合 DROID-SLAM 估计的相机参数 + VIMO 估计的相机坐标系 SMPL 参数。
        """
        from lib.vis.renderer import Renderer

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)
        if intrinsics is None:
            return data
        focal_length = float(intrinsics[0, 0])

        # 获取 c2w 相机外参（用于计算帧间相对变换）
        R_c2w, t_c2w = self._get_c2w(data)

        # 获取SMPL参数（相机坐标系）
        smpl_params = data.smpl_params
        if smpl_params.rotmat is None:
            self.logger.error("SMPL rotmat is required for rendering")
            return data

        rotmat = smpl_params.rotmat  # [N, 24, 3, 3] - 相机坐标系
        betas = smpl_params.betas  # [N, 10]
        trans = smpl_params.trans
        if trans is None:
            self.logger.error("trans (camera-coordinate) is not available in smpl_params")
            return data

        if trans.ndim == 3 and trans.shape[1] == 1:
            trans = trans.squeeze(1)

        self.logger.info(f"[camera mode] SMPL params: rotmat={rotmat.shape}, betas={betas.shape}, trans={trans.shape}")

        # 生成SMPL vertices（在各自的相机坐标系下）
        vertices = self._generate_vertices(rotmat, betas, trans)
        self.logger.info(f"Generated vertices: {vertices.shape} (camera coords, per-frame)")

        # 可选：生成用于相邻投影 BA 的稀疏 SMPL 点（默认 joints24）。
        # 这些点与 render 中使用的相机坐标系 SMPL 保持一致；后续会用同一组
        # reference c2w 生成 (center <- source) 的 2D 观测。
        projection_points = None
        if self.export_projection_factors:
            projection_points = self._generate_projection_points_camera(
                rotmat, betas, trans
            )
            self.logger.info(
                f"Generated projection factors points: {projection_points.shape} "
                f"({self.projection_factor_point_type})"
            )

        # 创建渲染器
        render = Renderer(W, H, focal_length, self.device, self.smpl_faces)

        # 为每一帧生成颜色
        self.frame_colors = self._generate_frame_colors(N)

        # 创建输出目录
        output_dir = self._get_output_dir(data)

        # 计算渲染的帧索引
        rendered_indices = list(range(0, N, self.pre_dis))

        # 保存最后一帧（如果不在渲染索引中）
        last_frame_saved = self._save_last_frame_if_needed(
            img_paths, rendered_indices, N, output_dir
        )

        has_c2w = R_c2w is not None
        projection_records = []

        # 渲染
        self.logger.info(f"Rendering {len(rendered_indices)} frames (every {self.pre_dis}th frame)...")

        for i in tqdm(rendered_indices, desc="Rendering adjacent frames"):
            img = cv2.imread(img_paths[i])
            final_img = img.copy()
            img_name = os.path.basename(img_paths[i])

            for offset in range(-self.num_adjacent_frames * self.pre_dis,
                                self.num_adjacent_frames * self.pre_dis + 1,
                                self.pre_dis):
                frame_idx = i + offset
                if 0 <= frame_idx < N:
                    verts_j = torch.tensor(vertices[frame_idx]).float().to(self.device)

                    if offset == 0 or not has_c2w:
                        verts_in_cam_i = verts_j
                    else:
                        # cam_j -> world -> cam_i
                        R_c2w_j = torch.from_numpy(R_c2w[frame_idx]).float().to(self.device)
                        t_c2w_j = torch.from_numpy(t_c2w[frame_idx]).float().to(self.device)
                        R_c2w_i = torch.from_numpy(R_c2w[i]).float().to(self.device)
                        t_c2w_i = torch.from_numpy(t_c2w[i]).float().to(self.device)

                        R_w2c_i = R_c2w_i.T
                        t_w2c_i = -R_w2c_i @ t_c2w_i

                        verts_world = torch.einsum('ij,vj->vi', R_c2w_j, verts_j) + t_c2w_j
                        verts_in_cam_i = torch.einsum('ij,vj->vi', R_w2c_i, verts_world) + t_w2c_i

                    if self.export_projection_factors and projection_points is not None and has_c2w:
                        projection_records.append({
                            'center': int(i),
                            'source': int(frame_idx),
                            'points_cam': projection_points[frame_idx],
                        })

                    if self.mesh_render_style == 'contour':
                        final_img = self._render_contour(
                            render, verts_in_cam_i, final_img,
                            colors=self._get_mesh_color(frame_idx),
                            thickness=self.contour_thickness
                        )
                    else:
                        final_img = self._render_solid(
                            render, verts_in_cam_i, final_img, colors=self._get_mesh_color(frame_idx)
                        )

            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_img)

        projection_info = self._pack_projection_factors(projection_records)
        self._log_render_summary(rendered_indices, N, last_frame_saved, output_dir)
        self._save_render_info(
            data, N, rendered_indices, last_frame_saved, output_dir,
            projection_info=projection_info,
        )

        return data

    def _execute_prompthmr_camera_mode(self, data: PipelineData) -> PipelineData:
        """PromptHMR 相机坐标系模式渲染（PromptHMR + DROID-SLAM c2w）

        策略：使用 PromptHMR 的相机坐标系参数 (global_orient_c + body_pose_aa + trans)
        通过 SMPL-X 模型生成每帧在各自相机坐标系下的 vertices，然后用 DROID-SLAM
        估计的 c2w 做帧间相对变换 (cam_j -> world -> cam_i)，将相邻帧的 SMPL mesh
        投影到中心帧的视角下渲染。

        与 camera 模式的区别：
        - 使用 SMPL-X (supermotion) + smplx2smpl 而非标准 SMPL（与 PromptHMR 一致）
        - 使用 axis-angle 参数 (global_orient_c, body_pose_aa) 而非 rotmat
        - 使用 GVHMRRenderer + 完整 K 矩阵（保留真实 cx/cy）

        与 gvhmr_world 模式的区别：
        - 使用 DROID-SLAM 的 c2w 而非从 incam/global 参数反推 T_w2c
        - vertices 在相机坐标系下生成，通过 c2w 做帧间变换
        """
        from lib.gvhmr import ASSETS
        from lib.gvhmr.body_model import make_smplx
        from lib.gvhmr.renderer import Renderer as GVHMRRenderer

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # PromptHMR demo 在渲染回原图时使用原图坐标系的 K，而不是 896 padded K。
        # PHMR cache 中的 metadata['hpe_K'] / camera_params.intrinsics 通常是 eval 阶段
        # pad 到 896x896 后的 K；这里只在渲染 720x960 等原图背景时使用原图 K。
        K_fullimg = None
        K_decode = None
        hpe_K = data.metadata.get('hpe_K', None) if data.metadata else None
        if hpe_K is not None:
            K_decode = torch.from_numpy(hpe_K).float() if isinstance(hpe_K, np.ndarray) else hpe_K.float()

        if data.annotations is not None and 'camera' in data.annotations:
            K_ann = data.annotations['camera'].get('intrinsics')
            if K_ann is not None:
                K_fullimg = torch.from_numpy(K_ann).float() if isinstance(K_ann, np.ndarray) else K_ann.float()
                self.logger.info(
                    f"[prompthmr_camera] Using annotation original-image K for PHMR-style render: "
                    f"fx={float(K_fullimg[0,0]):.2f}, fy={float(K_fullimg[1,1]):.2f}, "
                    f"cx={float(K_fullimg[0,2]):.2f}, cy={float(K_fullimg[1,2]):.2f}"
                )

        if K_fullimg is None and K_decode is not None:
            K_pad = K_decode
            img_size = 896.0
            scale = img_size / float(max(W, H))
            offset_x = (img_size - scale * float(W)) / 2.0
            offset_y = (img_size - scale * float(H)) / 2.0
            K_fullimg = K_pad.clone()
            K_fullimg[:2, :] /= scale
            K_fullimg[0, 2] = (K_pad[0, 2] - offset_x) / scale
            K_fullimg[1, 2] = (K_pad[1, 2] - offset_y) / scale
            self.logger.warning(
                f"[prompthmr_camera] No annotation K found; inverse-unpadding hpe_K to original-image K: "
                f"scale={scale:.6f}, offset=({offset_x:.2f},{offset_y:.2f}), "
                f"fx={float(K_fullimg[0,0]):.2f}, fy={float(K_fullimg[1,1]):.2f}, "
                f"cx={float(K_fullimg[0,2]):.2f}, cy={float(K_fullimg[1,2]):.2f}"
            )

        if K_fullimg is None:
            intrinsics = self._get_intrinsics(data)
            if intrinsics is not None:
                K_fullimg = torch.from_numpy(intrinsics).float() if isinstance(intrinsics, np.ndarray) else intrinsics.float()
                self.logger.warning(
                    f"[prompthmr_camera] Falling back to data camera intrinsics: "
                    f"fx={float(K_fullimg[0,0]):.2f}, fy={float(K_fullimg[1,1]):.2f}, "
                    f"cx={float(K_fullimg[0,2]):.2f}, cy={float(K_fullimg[1,2]):.2f}"
                )
        if K_fullimg is None:
            self.logger.error("No camera intrinsics found for prompthmr_camera mode")
            return data

        # 获取 DROID-SLAM 的 c2w
        R_c2w, t_c2w = self._get_c2w(data)
        if R_c2w is None:
            self.logger.error("prompthmr_camera mode requires SLAM c2w (data.camera_params.R/T)")
            return data

        # 获取 PromptHMR 的 incam 参数
        sp = data.smpl_params
        if sp.global_orient_c is None or sp.body_pose_aa is None or sp.trans is None:
            self.logger.error(
                "prompthmr_camera mode requires global_orient_c, body_pose_aa, trans in smpl_params"
            )
            return data

        device = torch.device(self.device)

        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            return x.float().to(device)

        global_orient_c = _to_tensor(sp.global_orient_c)  # (F, 3)
        body_pose_aa = _to_tensor(sp.body_pose_aa)         # (F, 63)
        betas = _to_tensor(sp.betas)                       # (F, 10)
        trans_c = _to_tensor(sp.trans)                      # (F, 3)
        # 平移的分量级重标定系数，供 oar_window 路径复用（那条路径的平移来自世界系，
        # 需要在中心相机系下施加同样的修正）。
        trans_component_scale = [1.0, 1.0, 1.0]
        if self.prompthmr_rescale_transl_to_render_k and K_decode is not None:
            # PromptHMR/GVHMR camt_type='prompthmr' 的 z 由 fx 解码，y 平移还依赖 fx/fy。
            # 渲染时若从 padded/cache K 切换到原图 K，需要同步重标定 z 和 y，
            # 否则 x 方向基本自洽，但 y 方向会出现明显缩放。
            fx_decode = max(float(K_decode[0, 0]), 1e-6)
            fy_decode = max(float(K_decode[1, 1]), 1e-6)
            fx_render = max(float(K_fullimg[0, 0]), 1e-6)
            fy_render = max(float(K_fullimg[1, 1]), 1e-6)
            z_scale = fx_render / fx_decode
            y_scale = (fx_render / fy_render) / (fx_decode / fy_decode)
            if abs(z_scale - 1.0) > 1e-4 or abs(y_scale - 1.0) > 1e-4:
                trans_c = trans_c.clone()
                trans_c[:, 1] *= y_scale
                trans_c[:, 2] *= z_scale
                trans_component_scale = [1.0, y_scale, z_scale]
                self.logger.info(
                    f"[prompthmr_camera] Rescaled transl for render K: "
                    f"z_scale={z_scale:.6f}, y_scale={y_scale:.6f}, "
                    f"K_decode=(fx={fx_decode:.2f}, fy={fy_decode:.2f}), "
                    f"K_render=(fx={fx_render:.2f}, fy={fy_render:.2f})"
                )
        if self.smpl_trans_scale != 1.0:
            self.logger.info(
                f"[prompthmr_camera] Scaling SMPL camera translation by {self.smpl_trans_scale:.3f} "
                "before rendering"
            )
            trans_c = trans_c * self.smpl_trans_scale

        self.logger.info(
            f"[prompthmr_camera] SMPL params: global_orient_c={global_orient_c.shape}, "
            f"body_pose={body_pose_aa.shape}, betas={betas.shape}, trans={trans_c.shape}"
        )

        # --- 1. 用 SMPL-X 生成相机坐标系下的 vertices ---
        self.logger.info("[prompthmr_camera] Loading SMPL-X model (supermotion)...")
        smplx_model = make_smplx("supermotion").to(device)
        smplx2smpl = torch.load(ASSETS / "smplx2smpl_sparse.pt").to(device)
        faces_smpl = make_smplx("smpl").faces

        with torch.no_grad():
            smplx_out = smplx_model(
                global_orient=global_orient_c,
                body_pose=body_pose_aa,
                betas=betas,
                transl=trans_c,
            )
            vertices_incam = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in smplx_out.vertices]
            )  # (F, V_smpl, 3)

        self.logger.info(f"Generated incam vertices via SMPL-X: {vertices_incam.shape}")

        # --- 1b. 可选：改用 depth_scene_refine 的窗口局部 transl ---
        window_by_center = None
        global_orient_w = None
        if self.transl_source == 'oar_window':
            window_results = data.metadata.get('oar_window_results') or []
            if not window_results:
                self.logger.warning(
                    "transl_source='oar_window' but metadata has no oar_window_results; "
                    "falling back to sequence-level transl. Did depth_scene_refine run?"
                )
            elif sp.global_orient_w is None:
                self.logger.warning(
                    "transl_source='oar_window' requires smpl_params.global_orient_w; "
                    "falling back to sequence-level transl."
                )
            else:
                window_by_center = {int(wr['center']): wr for wr in window_results}
                global_orient_w = _to_tensor(sp.global_orient_w)  # (F, 3)
                centers = sorted(window_by_center)
                oar_stride = centers[1] - centers[0] if len(centers) > 1 else None
                self.logger.info(
                    f"[prompthmr_camera] Window-local transl: {len(centers)} OAR windows, "
                    f"stride={oar_stride}, render pre_dis={self.pre_dis}"
                )
                if oar_stride is not None and oar_stride != self.pre_dis:
                    self.logger.warning(
                        f"OAR window stride ({oar_stride}) != pre_dis ({self.pre_dis}); "
                        "render centers will not align with window centers and most frames "
                        "will fall back to sequence-level transl. Set pre_dis to match "
                        "depth_scene_refine.stride."
                    )

        # --- 2. 渲染：用 SLAM c2w 做帧间相对变换 ---
        render = GVHMRRenderer(W, H, device=device, faces=faces_smpl, K=K_fullimg)
        self.frame_colors = self._generate_frame_colors(N)
        output_dir = self._get_output_dir(data)
        rendered_indices = self._get_render_frame_indices(N)
        last_frame_saved = False
        if self.config.get('render_frame_indices') is None:
            last_frame_saved = self._save_last_frame_if_needed(
                img_paths, rendered_indices, N, output_dir
            )

        self.logger.info(f"Rendering {len(rendered_indices)} selected frames...")

        offsets = list(range(-self.num_adjacent_frames * self.pre_dis,
                             self.num_adjacent_frames * self.pre_dis + 1,
                             self.pre_dis))
        num_window_centers = 0

        for i in tqdm(rendered_indices, desc="Rendering adjacent frames (prompthmr_camera)"):
            img = cv2.imread(img_paths[i])
            final_img = img.copy()
            img_name = os.path.basename(img_paths[i])

            # 当前帧 cam_i 的 c2w / w2c
            R_c2w_i = torch.from_numpy(R_c2w[i]).float().to(device)
            t_c2w_i = torch.from_numpy(t_c2w[i]).float().to(device)
            R_w2c_i = R_c2w_i.T
            t_w2c_i = -R_w2c_i @ t_c2w_i

            window_record = window_by_center.get(i) if window_by_center is not None else None

            if window_record is not None:
                # 窗口局部：mesh 用该窗口自己修正过的 transl/body_pose，直接摆到窗口中心视角
                start, end = int(window_record['start']), int(window_record['end'])
                frame_indices = [
                    i + o for o in offsets
                    if 0 <= i + o < N and start <= i + o < end
                ]
                verts_batch = self._window_verts_in_center_cam(
                    smplx_model, smplx2smpl, window_record, frame_indices,
                    global_orient_w, betas, R_w2c_i, t_w2c_i,
                    trans_component_scale, device,
                )
                verts_by_frame = {j: verts_batch[k] for k, j in enumerate(frame_indices)}
                num_window_centers += 1
            else:
                verts_by_frame = {}
                for offset in offsets:
                    frame_idx = i + offset
                    if not (0 <= frame_idx < N):
                        continue
                    verts_cam_j = vertices_incam[frame_idx].to(device)
                    if offset == 0:
                        verts_by_frame[frame_idx] = verts_cam_j
                    else:
                        # cam_j -> world -> cam_i
                        R_c2w_j = torch.from_numpy(R_c2w[frame_idx]).float().to(device)
                        t_c2w_j = torch.from_numpy(t_c2w[frame_idx]).float().to(device)

                        verts_world = torch.einsum('ij,vj->vi', R_c2w_j, verts_cam_j) + t_c2w_j
                        verts_by_frame[frame_idx] = (
                            torch.einsum('ij,vj->vi', R_w2c_i, verts_world) + t_w2c_i
                        )

            for frame_idx in sorted(verts_by_frame):
                verts_in_cam_i = verts_by_frame[frame_idx]
                if self.mesh_render_style == 'contour':
                    final_img = self._render_contour(
                        render, verts_in_cam_i, final_img,
                        colors=self._get_mesh_color(frame_idx),
                        thickness=self.contour_thickness
                    )
                else:
                    final_img = self._render_solid(
                        render, verts_in_cam_i, final_img, colors=self._get_mesh_color(frame_idx)
                    )

            output_path = os.path.join(output_dir, img_name)
            if not cv2.imwrite(output_path, final_img):
                raise OSError(f'Failed to write rendered keyframe: {output_path}')

        if window_by_center is not None:
            self.logger.info(
                f"[prompthmr_camera] Window-local transl used on {num_window_centers}/"
                f"{len(rendered_indices)} rendered centers"
            )

        del smplx_model, smplx2smpl, vertices_incam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log_render_summary(rendered_indices, N, last_frame_saved, output_dir)
        self._save_render_info(
            data, N, rendered_indices, last_frame_saved, output_dir,
            num_window_centers=num_window_centers if window_by_center is not None else None,
        )

        return data

    def _window_verts_in_center_cam(
        self,
        smplx_model,
        smplx2smpl,
        window_record,
        frame_indices,
        global_orient_w,
        betas,
        R_w2c_center,
        t_w2c_center,
        trans_component_scale,
        device,
    ):
        """Pose one OAR window's local SMPL and express it in the window-center camera.

        The window's ``transl`` / ``body_pose`` are the window-local corrected values
        (post pp_static_joint + contact IK) living in the world frame, so the mesh is
        built in world and then rigidly moved into the center camera - mirroring how
        depth_scene_refine builds ``transl_center_cam`` / ``joints_center_cam``.

        The render-K translation rescale is applied as a pure mesh translation rather
        than by re-parameterizing ``transl``, which would shift the mesh by
        ``(R - I) @ J_pelvis`` because SMPL rotates about the pelvis, not the origin.

        Returns:
            verts_cam: (len(frame_indices), V_smpl, 3) in the center camera frame
        """
        start = int(window_record['start'])
        idx_global = torch.as_tensor([int(j) for j in frame_indices], dtype=torch.long)
        idx_local = torch.as_tensor([int(j) - start for j in frame_indices], dtype=torch.long)

        win_transl = window_record['transl'].to(device).float()      # (win_len, 3)
        win_body_pose = window_record['body_pose'].to(device).float()  # (win_len, 63)

        transl_w = win_transl[idx_local.to(win_transl.device)]
        body_pose = win_body_pose[idx_local.to(win_body_pose.device)]
        orient_w = global_orient_w[idx_global.to(global_orient_w.device)].to(device)
        betas_sel = betas[idx_global.to(betas.device)].to(device)

        with torch.no_grad():
            smplx_out = smplx_model(
                global_orient=orient_w,
                body_pose=body_pose,
                betas=betas_sel,
                transl=transl_w,
            )
            verts_world = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in smplx_out.vertices]
            )  # (n, V_smpl, 3)

            verts_cam = (
                torch.einsum('ij,nvj->nvi', R_w2c_center, verts_world)
                + t_w2c_center[None, None, :]
            )

            root_cam = torch.einsum('ij,nj->ni', R_w2c_center, transl_w) + t_w2c_center[None, :]
            scale_vec = torch.tensor(
                trans_component_scale, device=root_cam.device, dtype=root_cam.dtype
            )
            root_fixed = root_cam * scale_vec * float(self.smpl_trans_scale)
            verts_cam = verts_cam + (root_fixed - root_cam)[:, None, :]

        return verts_cam

    def _execute_world_mode(self, data: PipelineData) -> PipelineData:
        """世界坐标系模式渲染（适合 GT）

        策略：直接使用世界坐标系的 trans (global_trans) 和世界坐标系的 root orientation
        生成 vertices，所有帧的 vertices 都在同一个世界坐标系下，然后用 GT 的 w2c
        (extrinsics) 投影到当前帧的相机坐标系下渲染。

        这样避免了 TRAM 方法和 GT 对 Y 轴约定不同导致的渲染偏移：
        - TRAM: 相机坐标系下 trans_cam 是由 VIMO 估计的，c2w 由 DROID-SLAM 估计
        - GT: 世界坐标系下 trans 是真值，extrinsics 是真值的 w2c 变换
        两者的世界坐标系 Y 轴约定可能不同，camera mode 下通过 trans_cam -> c2w -> w2c
        的链路会引入 Y 轴偏移。world mode 直接在 GT 世界坐标系下操作，避免了这个问题。
        """
        from lib.vis.renderer import Renderer
        from lib.utils.rotation_conversions import axis_angle_to_matrix

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)
        if intrinsics is None:
            return data
        focal_length = float(intrinsics[0, 0])

        # 获取 GT extrinsics (w2c)
        if data.annotations is None or 'camera' not in data.annotations:
            self.logger.error("world mode requires annotations with camera extrinsics")
            return data

        ext = data.annotations['camera']['extrinsics']  # [N_total, 3, 4] w2c

        # 检查帧采样
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
            ext = ext[sampled_indices]

        R_w2c = ext[:, :3, :3]  # [N, 3, 3]
        t_w2c = ext[:, :3, 3]   # [N, 3]

        # 获取 GT 世界坐标系 SMPL 参数
        ann = data.annotations
        poses_body = ann["smpl"]["poses_body"]   # [N_total, 23, 3] axis-angle
        poses_root = ann["smpl"]["poses_root"]   # [N_total, 3] axis-angle
        betas_raw = ann["smpl"]["betas"]         # [10]
        trans_world = ann["smpl"]["trans"]        # [N_total, 3]

        if sampling_info:
            poses_body = poses_body[sampled_indices]
            poses_root = poses_root[sampled_indices]
            trans_world = trans_world[sampled_indices]

        betas = np.repeat(betas_raw.reshape((1, -1)), repeats=len(poses_body), axis=0)

        self.logger.info(
            f"[world mode] GT SMPL params: poses_root={poses_root.shape}, "
            f"poses_body={poses_body.shape}, betas={betas.shape}, trans={trans_world.shape}"
        )

        # 在世界坐标系下生成 vertices
        # 使用世界坐标系的 root orientation + trans，pose2rot=True (axis-angle 输入)
        vertices_world = self._generate_vertices_world(
            poses_root, poses_body, betas, trans_world
        )
        self.logger.info(f"Generated vertices: {vertices_world.shape} (world coords)")

        # 创建渲染器
        render = Renderer(W, H, focal_length, self.device, self.smpl_faces)

        # 为每一帧生成颜色
        self.frame_colors = self._generate_frame_colors(N)

        # 创建输出目录
        output_dir = self._get_output_dir(data)

        # 计算渲染的帧索引
        rendered_indices = list(range(0, N, self.pre_dis))

        # 保存最后一帧
        last_frame_saved = self._save_last_frame_if_needed(
            img_paths, rendered_indices, N, output_dir
        )

        # 渲染
        self.logger.info(f"Rendering {len(rendered_indices)} frames (every {self.pre_dis}th frame)...")

        for i in tqdm(rendered_indices, desc="Rendering adjacent frames (world mode)"):
            img = cv2.imread(img_paths[i])
            final_img = img.copy()
            img_name = os.path.basename(img_paths[i])

            # 当前帧的 w2c
            R_w2c_i = torch.from_numpy(R_w2c[i]).float().to(self.device)  # [3, 3]
            t_w2c_i = torch.from_numpy(t_w2c[i]).float().to(self.device)  # [3]

            for offset in range(-self.num_adjacent_frames * self.pre_dis,
                                self.num_adjacent_frames * self.pre_dis + 1,
                                self.pre_dis):
                frame_idx = i + offset
                if 0 <= frame_idx < N:
                    verts_w = torch.tensor(vertices_world[frame_idx]).float().to(self.device)

                    # 世界坐标系 -> 当前帧相机坐标系: p_cam = R_w2c @ p_world + t_w2c
                    verts_in_cam_i = torch.einsum('ij,vj->vi', R_w2c_i, verts_w) + t_w2c_i

                    if self.mesh_render_style == 'contour':
                        final_img = self._render_contour(
                            render, verts_in_cam_i, final_img,
                            colors=self._get_mesh_color(frame_idx),
                            thickness=self.contour_thickness
                        )
                    else:
                        final_img = self._render_solid(
                            render, verts_in_cam_i, final_img, colors=self._get_mesh_color(frame_idx)
                        )

            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_img)

        self._log_render_summary(rendered_indices, N, last_frame_saved, output_dir)
        self._save_render_info(data, N, rendered_indices, last_frame_saved, output_dir)

        return data

    def _execute_gvhmr_world_mode(self, data: PipelineData) -> PipelineData:
        """GVHMR 世界坐标系模式渲染

        策略：从 GVHMR 的 global+incam 两套输出反推每帧的 T_w2c（世界到相机的变换矩阵），
        使用 GVHMR 原生的 SMPL-X 模型生成世界坐标系下的 vertices（通过 smplx2smpl 转换），
        再用原始的 T_w2c 投影到中心帧的相机视角下渲染。

        关键公式（来自 GVHMR 的 get_T_w2c_from_wcparams）：
            R_w2c = R_c @ R_w^T
            t_w2c = t_c + offset - R_w2c @ (t_w + offset)

        注意：
        - T_w2c 使用滑步修正前的 global 参数（transl_w_raw）反推，
          因为 incam 参数未被修正，这样才能保证 T_w2c 的精确性。
        - 使用 GVHMR 原生的 SMPL-X 模型 (make_smplx("supermotion")) + smplx2smpl
          转换，与 hooks 中 incam 渲染保持一致，避免 SMPL vs SMPL-X 参数空间不兼容。
        - 使用 GVHMRRenderer（支持完整 K 矩阵），避免只取 fx 丢失 cx/cy 导致像素偏移。

        完全不需要外部 SLAM 的相机参数。
        """
        from lib.gvhmr import ASSETS
        from lib.gvhmr.body_model import make_smplx
        from lib.gvhmr.geometry import get_T_w2c_from_wcparams
        from lib.gvhmr.renderer import Renderer as GVHMRRenderer

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # 获取相机内参（完整 K 矩阵）— 优先使用 HPE estimate_K
        K_fullimg = None
        hpe_K = data.metadata.get('hpe_K', None) if data.metadata else None
        if hpe_K is not None:
            if isinstance(hpe_K, np.ndarray):
                K_fullimg = torch.from_numpy(hpe_K).float()
            elif isinstance(hpe_K, torch.Tensor):
                K_fullimg = hpe_K.float()
            self.logger.info(
                f"[gvhmr_world] Using HPE estimate_K: fx={float(K_fullimg[0,0]):.2f}, "
                f"fy={float(K_fullimg[1,1]):.2f}, "
                f"cx={float(K_fullimg[0,2]):.2f}, cy={float(K_fullimg[1,2]):.2f}"
            )
        if K_fullimg is None and data.camera_params and data.camera_params.intrinsics is not None:
            K_fullimg = data.camera_params.intrinsics
            if isinstance(K_fullimg, np.ndarray):
                K_fullimg = torch.from_numpy(K_fullimg).float()
            self.logger.warning(
                f"[gvhmr_world] No hpe_K found, falling back to GT K: "
                f"fx={float(K_fullimg[0,0]):.2f} — may cause projection mismatch!"
            )
        if K_fullimg is None:
            # fallback: 从 annotations 获取
            intrinsics = self._get_intrinsics(data)
            if intrinsics is not None:
                if isinstance(intrinsics, np.ndarray):
                    K_fullimg = torch.from_numpy(intrinsics).float()
                else:
                    K_fullimg = intrinsics.float()
        if K_fullimg is None:
            self.logger.error("No camera intrinsics found for gvhmr_world mode")
            return data

        # 获取 GVHMR 参数
        sp = data.smpl_params
        if sp.global_orient_w is None or sp.global_orient_c is None:
            self.logger.error(
                "gvhmr_world mode requires GVHMR-specific fields in smpl_params "
                "(global_orient_w, global_orient_c, etc.)"
            )
            return data

        global_orient_w = sp.global_orient_w  # (F, 3) axis-angle
        global_orient_c = sp.global_orient_c  # (F, 3) axis-angle
        transl_w_raw = sp.transl_w_raw  # (F, 3) 滑步修正前的 world transl
        global_trans = sp.global_trans  # (F, 3) 滑步修正后的 world transl
        transl_c = sp.trans  # (F, 3) incam transl
        skeleton_offset = sp.skeleton_offset  # (3,) root joint offset

        # 修正后的参数（经过 SkatingRemoval）
        body_pose_fixed = sp.body_pose_aa  # (F, 63) 修正后的 body pose
        betas = sp.betas  # (F, 10)

        # 确保在正确的 device 上
        device = torch.device(self.device)

        # --- 1. 反推 T_w2c（使用修正后的 global_trans + incam 参数） ---
        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(device)
            return x.float().to(device)

        T_w2c = get_T_w2c_from_wcparams(
            _to_tensor(global_orient_w),
            _to_tensor(global_trans),
            _to_tensor(global_orient_c),
            _to_tensor(transl_c),
            _to_tensor(skeleton_offset),
        )  # (F, 4, 4)

        self.logger.info(f"[gvhmr_world mode] Computed T_w2c: {T_w2c.shape}")

        R_w2c = T_w2c[:, :3, :3].cpu().numpy()  # (F, 3, 3)
        t_w2c = T_w2c[:, :3, 3].cpu().numpy()  # (F, 3)

        # --- 2. 使用 SMPL-X 模型生成世界坐标系下的 vertices ---
        # 使用 GVHMR 原生的 SMPL-X 模型，与 hooks 中 incam 渲染保持一致
        self.logger.info("[gvhmr_world mode] Loading SMPL-X model (supermotion)...")
        smplx_model = make_smplx("supermotion").to(device)
        smplx2smpl = torch.load(ASSETS / "smplx2smpl_sparse.pt").to(device)
        faces_smpl = make_smplx("smpl").faces

        smpl_params_world = {
            'global_orient': _to_tensor(global_orient_w),
            'body_pose': _to_tensor(body_pose_fixed),
            'betas': _to_tensor(betas),
            'transl': _to_tensor(global_trans),
        }
        with torch.no_grad():
            smplx_out_w = smplx_model(**smpl_params_world)
            # SMPL-X vertices -> SMPL vertices (通过 smplx2smpl 稀疏矩阵)
            vertices_world = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in smplx_out_w.vertices]
            )  # (F, V_smpl, 3)

        self.logger.info(
            f"Generated vertices via SMPL-X: {vertices_world.shape} "
            f"(world coords, using global_trans for skating-corrected rendering)"
        )

        # --- 3. 渲染（使用 GVHMRRenderer + 完整 K 矩阵） ---
        render = GVHMRRenderer(W, H, device=device, faces=faces_smpl, K=K_fullimg)
        self.frame_colors = self._generate_frame_colors(N)
        output_dir = self._get_output_dir(data)
        rendered_indices = list(range(0, N, self.pre_dis))
        last_frame_saved = self._save_last_frame_if_needed(img_paths, rendered_indices, N, output_dir)

        self.logger.info(f"Rendering {len(rendered_indices)} frames (every {self.pre_dis}th frame)...")

        for i in tqdm(rendered_indices, desc="Rendering adjacent frames (gvhmr_world mode)"):
            img = cv2.imread(img_paths[i])
            final_img = img.copy()
            img_name = os.path.basename(img_paths[i])

            # 当前帧的 w2c
            R_w2c_i = torch.from_numpy(R_w2c[i]).float().to(device)
            t_w2c_i = torch.from_numpy(t_w2c[i]).float().to(device)

            for offset in range(-self.num_adjacent_frames * self.pre_dis,
                                self.num_adjacent_frames * self.pre_dis + 1,
                                self.pre_dis):
                frame_idx = i + offset
                if 0 <= frame_idx < N:
                    verts_w = vertices_world[frame_idx].to(device)
                    # world -> cam_i: p_cam = R_w2c_i @ p_world + t_w2c_i
                    verts_in_cam_i = torch.einsum('ij,vj->vi', R_w2c_i, verts_w) + t_w2c_i
                    if self.mesh_render_style == 'contour':
                        final_img = self._render_contour(
                            render, verts_in_cam_i, final_img,
                            colors=self._get_mesh_color(frame_idx),
                            thickness=self.contour_thickness
                        )
                    else:
                        final_img = self._render_solid(
                            render, verts_in_cam_i, final_img, colors=self._get_mesh_color(frame_idx)
                        )

            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_img)

        # 释放 SMPL-X 模型（gvhmr_world 专用，不影响 self.smpl_model）
        del smplx_model, smplx2smpl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._log_render_summary(rendered_indices, N, last_frame_saved, output_dir)
        self._save_render_info(data, N, rendered_indices, last_frame_saved, output_dir)

        return data

    def _execute_keyframe_mode(self, data: PipelineData) -> PipelineData:
        """关键帧模式渲染

        当 render_only_keyframes=False（默认）时，输出所有帧的图像：
        - 关键帧：渲染该帧在关键帧列表中前后 num_adjacent_frames 个邻居关键帧的 SMPL mesh
        - 非关键帧：使用 mask 将人体区域填充为黑色（去人）

        当 render_only_keyframes=True 时，只输出关键帧的渲染图像（跳过非关键帧以节省时间）。
        适用于下游 SLAM 使用 use_rendered_for_keyframes=True 的场景，因为非关键帧的渲染结果不会被使用。

        支持 'camera' 和 'world' 两种 render_mode。
        """
        from lib.vis.renderer import Renderer

        # 获取关键帧信息
        keyframe_info = data.metadata.get('slam_keyframe_info')
        if keyframe_info is None:
            raise ValueError(
                "keyframe frame_selection_mode requires slam_keyframe_info in metadata. "
                "Ensure the first SLAM component has return_keyframe_info: true."
            )

        # 获取图像路径
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])

        N = len(img_paths)
        first_img = cv2.imread(img_paths[0])
        H, W = first_img.shape[:2]

        # 获取相机内参
        intrinsics = self._get_intrinsics(data)
        if intrinsics is None:
            return data
        focal_length = float(intrinsics[0, 0])

        # 获取关键帧索引（已排序、去重、范围检查）
        keyframe_indices = sorted(set(
            int(idx) for idx in keyframe_info['keyframe_tstamps'] if int(idx) < N
        ))
        keyframe_set = set(keyframe_indices)
        self.logger.info(f"Keyframe mode: {len(keyframe_indices)} keyframes out of {N} total frames")

        # 根据 render_mode 准备 vertices 和相机参数
        if self.render_mode == 'world':
            vertices, R_w2c, t_w2c = self._prepare_world_mode_data(data, N)
        else:
            vertices, R_c2w, t_c2w = self._prepare_camera_mode_data(data, N)

        # 创建渲染器
        render = Renderer(W, H, focal_length, self.device, self.smpl_faces)

        # 为每一帧生成颜色
        self.frame_colors = self._generate_frame_colors(N)

        # 创建输出目录
        output_dir = self._get_output_dir(data)

        # 确定需要处理的帧
        render_only = self.render_only_keyframes
        if render_only:
            frames_to_process = keyframe_indices
            self.logger.info(
                f"render_only_keyframes=True: only rendering {len(keyframe_indices)} keyframes "
                f"(skipping {N - len(keyframe_indices)} non-keyframes)"
            )
        else:
            frames_to_process = list(range(N))
            # 获取 masks 用于非关键帧去人
            masks = data.masks
            if masks is None:
                self.logger.warning("No masks available for non-keyframe human removal, using original images")
            self.logger.info(f"Rendering all {N} frames (keyframe mode)...")

        for i in tqdm(frames_to_process, desc="Rendering (keyframe mode)"):
            img = cv2.imread(img_paths[i])
            img_name = os.path.basename(img_paths[i])

            if i in keyframe_set:
                # 关键帧：渲染邻居关键帧的 SMPL mesh
                final_img = img.copy()
                pos = keyframe_indices.index(i)
                neighbor_start = max(0, pos - self.num_adjacent_frames)
                neighbor_end = min(len(keyframe_indices), pos + self.num_adjacent_frames + 1)

                for ni in range(neighbor_start, neighbor_end):
                    frame_idx = keyframe_indices[ni]
                    verts = torch.tensor(vertices[frame_idx]).float().to(self.device)

                    if self.render_mode == 'world':
                        # world -> cam_i
                        R_w2c_i = torch.from_numpy(R_w2c[i]).float().to(self.device)
                        t_w2c_i = torch.from_numpy(t_w2c[i]).float().to(self.device)
                        verts_in_cam = torch.einsum('ij,vj->vi', R_w2c_i, verts) + t_w2c_i
                    else:
                        # camera mode: cam_j -> world -> cam_i
                        if frame_idx == i:
                            verts_in_cam = verts
                        else:
                            R_c2w_j = torch.from_numpy(R_c2w[frame_idx]).float().to(self.device)
                            t_c2w_j = torch.from_numpy(t_c2w[frame_idx]).float().to(self.device)
                            R_c2w_i = torch.from_numpy(R_c2w[i]).float().to(self.device)
                            t_c2w_i = torch.from_numpy(t_c2w[i]).float().to(self.device)
                            R_w2c_i = R_c2w_i.T
                            t_w2c_i = -R_w2c_i @ t_c2w_i
                            verts_world = torch.einsum('ij,vj->vi', R_c2w_j, verts) + t_c2w_j
                            verts_in_cam = torch.einsum('ij,vj->vi', R_w2c_i, verts_world) + t_w2c_i

                    if self.mesh_render_style == 'contour':
                        final_img = self._render_contour(
                            render, verts_in_cam, final_img,
                            colors=self._get_mesh_color(frame_idx),
                            thickness=self.contour_thickness
                        )
                    else:
                        final_img = self._render_solid(
                            render, verts_in_cam, final_img, colors=self._get_mesh_color(frame_idx)
                        )
            else:
                # 非关键帧（仅当 render_only=False 时才会进入此分支）
                if masks is not None:
                    mask_i = masks[i]
                    if isinstance(mask_i, torch.Tensor):
                        mask_i = mask_i.cpu().numpy()
                    if mask_i.ndim == 3:
                        mask_i = mask_i[0]
                    if mask_i.shape[0] != H or mask_i.shape[1] != W:
                        mask_i = cv2.resize(mask_i, (W, H))
                    final_img = img.copy()
                    final_img[mask_i > 0.5] = 0
                else:
                    final_img = img.copy()

            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, final_img)

        # Log and save render info
        num_output = len(frames_to_process)
        self.logger.info(f"Keyframe mode rendering complete:")
        self.logger.info(f"  - Total frames output: {num_output}")
        self.logger.info(f"  - Keyframes with SMPL mesh: {len(keyframe_indices)}")
        if not render_only:
            self.logger.info(f"  - Non-keyframes with mask removal: {N - len(keyframe_indices)}")
        else:
            self.logger.info(f"  - Non-keyframes skipped (render_only_keyframes=True)")
        self.logger.info(f"  - Output directory: {output_dir}")

        # Save render info
        data.metadata['adjacent_render_info'] = {
            'total_frames': N,
            'pre_dis': None,
            'rendered_indices': keyframe_indices if render_only else list(range(N)),
            'keyframe_indices': keyframe_indices,
            'last_frame_idx': None,
            'num_processed_frames': num_output,
            'output_dir': output_dir,
            'frame_selection_mode': 'keyframe',
            'render_only_keyframes': render_only,
        }

        return data

    def _prepare_world_mode_data(self, data, N):
        """为 world mode keyframe 渲染准备数据"""
        from lib.utils.rotation_conversions import axis_angle_to_matrix

        if data.annotations is None or 'camera' not in data.annotations:
            raise ValueError("world mode requires annotations with camera extrinsics")

        ext = data.annotations['camera']['extrinsics']
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
            ext = ext[sampled_indices]

        R_w2c = ext[:, :3, :3]
        t_w2c = ext[:, :3, 3]

        ann = data.annotations
        poses_root = ann["smpl"]["poses_root"]
        poses_body = ann["smpl"]["poses_body"]
        betas_raw = ann["smpl"]["betas"]
        trans_world = ann["smpl"]["trans"]

        if sampling_info:
            poses_body = poses_body[sampled_indices]
            poses_root = poses_root[sampled_indices]
            trans_world = trans_world[sampled_indices]

        betas = np.repeat(betas_raw.reshape((1, -1)), repeats=len(poses_body), axis=0)
        vertices = self._generate_vertices_world(poses_root, poses_body, betas, trans_world)

        return vertices, R_w2c, t_w2c

    def _prepare_camera_mode_data(self, data, N):
        """为 camera mode keyframe 渲染准备数据"""
        R_c2w, t_c2w = self._get_c2w(data)

        smpl_params = data.smpl_params
        if smpl_params.rotmat is None:
            raise ValueError("SMPL rotmat is required for camera mode rendering")

        rotmat = smpl_params.rotmat
        betas = smpl_params.betas
        trans = smpl_params.trans
        if trans is None:
            raise ValueError("trans (camera-coordinate) is not available in smpl_params")
        if trans.ndim == 3 and trans.shape[1] == 1:
            trans = trans.squeeze(1)

        vertices = self._generate_vertices(rotmat, betas, trans)

        return vertices, R_c2w, t_c2w

    def _get_intrinsics(self, data):
        """获取相机内参

        优先使用 HPE 推理时的 estimate_K (hpe_K)，因为 SMPL 参数
        (global_orient_w, global_trans 等) 是基于 estimate_K 推理的，
        渲染时必须用相同的 K 才能正确对齐到图像。
        仅在 hpe_K 不可用时 fallback 到 GT K。
        """
        # 优先级1: HPE estimate_K（与 SMPL 参数推理一致）
        hpe_K = data.metadata.get('hpe_K', None) if data.metadata else None
        if hpe_K is not None:
            self.logger.info("[_get_intrinsics] Using HPE estimate_K (hpe_K) for projection consistency")
            if isinstance(hpe_K, np.ndarray):
                return hpe_K
            return hpe_K.numpy() if hasattr(hpe_K, 'numpy') else hpe_K
        # 优先级2: camera_params.intrinsics (GT K / SLAM K)
        if data.camera_params is not None and data.camera_params.intrinsics is not None:
            return data.camera_params.intrinsics
        # 优先级3: annotations GT K
        elif data.annotations and 'camera' in data.annotations and 'intrinsics' in data.annotations['camera']:
            return data.annotations['camera']['intrinsics']
        else:
            self.logger.error("No camera intrinsics found in hpe_K, camera_params or annotations")
            return None

    def _get_c2w(self, data):
        """获取 c2w 相机外参"""
        R_c2w = None
        t_c2w = None

        if data.camera_params is not None and data.camera_params.R is not None and data.camera_params.T is not None:
            R_c2w = data.camera_params.R
            t_c2w = data.camera_params.T

            if isinstance(R_c2w, torch.Tensor):
                R_c2w = R_c2w.cpu().numpy()
            if isinstance(t_c2w, torch.Tensor):
                t_c2w = t_c2w.cpu().numpy()
            self.logger.info("Using c2w from camera_params for inter-frame transforms")
        else:
            self.logger.warning("No c2w camera params available")

        return R_c2w, t_c2w

    def _get_output_dir(self, data):
        """创建并返回输出目录"""
        seq_name = data.sequence_name or 'unnamed'
        base_output_dir = data.metadata.get('output_dir', self.output_dir)
        output_dir = os.path.join(base_output_dir, seq_name, 'adjacent_smpl')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_last_frame_if_needed(self, img_paths, rendered_indices, N, output_dir):
        """如果最后一帧不在渲染索引中，保存原始图像供 SLAM 使用"""
        last_frame_idx = N - 1
        if last_frame_idx not in rendered_indices:
            self.logger.info(f"Last frame {last_frame_idx} will be saved (not rendered) for SLAM")
            last_img = cv2.imread(img_paths[last_frame_idx])
            last_img_name = os.path.basename(img_paths[last_frame_idx])
            last_output_path = os.path.join(output_dir, last_img_name)
            cv2.imwrite(last_output_path, last_img)
            self.logger.info(f"Saved last frame to {last_output_path} for SLAM")
            return True
        return False

    def _log_render_summary(self, rendered_indices, N, last_frame_saved, output_dir):
        """打印渲染摘要日志"""
        self.logger.info(f"Rendered images saved to {output_dir}")
        total_adjacent = 2 * self.num_adjacent_frames + 1
        self.logger.info(
            f"  - Each frame shows {total_adjacent} SMPL meshes: "
            f"from -{self.num_adjacent_frames * self.pre_dis} to "
            f"+{self.num_adjacent_frames * self.pre_dis} (step={self.pre_dis})"
        )
        self.logger.info(f"  - Total frames rendered: {len(rendered_indices)} out of {N} total frames")
        if last_frame_saved:
            last_frame_idx = N - 1
            self.logger.info(f"  - Last frame {last_frame_idx} saved without mesh rendering (will be used by DROID-SLAM)")

    def _save_render_info(self, data, N, rendered_indices, last_frame_saved, output_dir,
                          projection_info=None, num_window_centers=None):
        """保存渲染信息到 metadata"""
        last_frame_idx = N - 1
        info = {
            'total_frames': N,
            'pre_dis': self.pre_dis,
            'num_adjacent_frames': self.num_adjacent_frames,
            'transl_source': self.transl_source,
            'rendered_indices': rendered_indices,
            'last_frame_idx': last_frame_idx if last_frame_saved else None,
            'num_processed_frames': len(rendered_indices) + (1 if last_frame_saved else 0),
            'output_dir': output_dir,
        }
        if num_window_centers is not None:
            info['num_window_local_centers'] = int(num_window_centers)
        if projection_info is not None:
            info['projection_info'] = projection_info
        data.metadata['adjacent_render_info'] = info

    def _pack_projection_factors(self, records):
        """Pack adjacent projection factor records into numpy arrays.

        The packed data stores source-frame SMPL points in their own camera
        coordinates. The 2D observations are regenerated later with the same
        reference c2w trajectory that rendered the adjacent SMPL image.
        """
        if not records:
            return None
        center_indices = np.array([r['center'] for r in records], dtype=np.int64)
        source_indices = np.array([r['source'] for r in records], dtype=np.int64)
        source_points_cam = np.stack([r['points_cam'] for r in records], axis=0).astype(np.float32)
        valid = np.ones(source_points_cam.shape[:2], dtype=np.bool_)
        return {
            'point_type': self.projection_factor_point_type,
            'center_indices': center_indices,
            'source_indices': source_indices,
            'source_points_cam': source_points_cam,
            'valid': valid,
        }

    def _generate_projection_points_camera(self, rotmat, betas, trans):
        """Generate sparse SMPL points in each frame's camera coordinates."""
        if self.projection_factor_point_type != 'joints24':
            raise ValueError(
                f"Unsupported projection_factor_point_type: {self.projection_factor_point_type}"
            )
        with torch.no_grad():
            if isinstance(rotmat, np.ndarray):
                rotmat = torch.from_numpy(rotmat).float()
            if isinstance(betas, np.ndarray):
                betas = torch.from_numpy(betas).float()
            if isinstance(trans, np.ndarray):
                trans = torch.from_numpy(trans).float()
            if trans.ndim == 3 and trans.shape[1] == 1:
                trans = trans.squeeze(1)

            rotmat = rotmat.to(self.device)
            betas = betas.to(self.device)
            trans = trans.to(self.device)
            output = self.smpl_model(
                global_orient=rotmat[:, [0]],
                body_pose=rotmat[:, 1:],
                betas=betas,
                pose2rot=False,
                default_smpl=True,
            )
            joints = output.joints[:, :24, :] + trans[:, None, :]
            return joints.detach().cpu().numpy().astype(np.float32)

    def _generate_vertices(self, rotmat, betas, trans):
        """从相机坐标系 SMPL 参数生成 vertices（camera mode 使用）

        输入为 rotation matrix 格式，pose2rot=False
        """
        with torch.no_grad():
            if isinstance(rotmat, np.ndarray):
                rotmat_tensor = torch.from_numpy(rotmat).float().to(self.device)
            else:
                rotmat_tensor = rotmat.float().to(self.device)

            if isinstance(betas, np.ndarray):
                betas_tensor = torch.from_numpy(betas).float().to(self.device)
            else:
                betas_tensor = betas.float().to(self.device)

            if isinstance(trans, np.ndarray):
                trans_tensor = torch.from_numpy(trans).float().to(self.device)
            else:
                trans_tensor = trans.float().to(self.device)

            global_orient = rotmat_tensor[:, [0]]
            body_pose = rotmat_tensor[:, 1:]

            smpl_output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas_tensor,
                transl=trans_tensor,
                pose2rot=False,
            )

            vertices = smpl_output.vertices

        return vertices.cpu().numpy()

    def _generate_vertices_world(self, poses_root, poses_body, betas, trans_world):
        """从世界坐标系 GT SMPL 参数生成 vertices（world mode 使用）

        输入为 axis-angle 格式，pose2rot=True
        """
        with torch.no_grad():
            poses_root_t = torch.from_numpy(poses_root).float().to(self.device)
            poses_body_t = torch.from_numpy(poses_body).float().to(self.device)
            betas_t = torch.from_numpy(betas).float().to(self.device)
            trans_t = torch.from_numpy(trans_world).float().to(self.device)

            smpl_output = self.smpl_model(
                global_orient=poses_root_t,
                body_pose=poses_body_t,
                betas=betas_t,
                transl=trans_t,
                pose2rot=True,
            )

            vertices = smpl_output.vertices

        return vertices.cpu().numpy()

    def _generate_vertices_gvhmr(self, global_orient_aa, body_pose_aa, betas, transl):
        """从 GVHMR 的 axis-angle 参数生成 vertices（gvhmr_world mode 使用）

        GVHMR 输出的 global_orient 和 body_pose 都是 axis-angle 格式。
        SMPL 模型需要：
          - global_orient: (F, 3) axis-angle
          - body_pose: (F, 69) 或 (F, 23*3) axis-angle
          - betas: (F, 10)
          - transl: (F, 3)

        注意：GVHMR 的 body_pose 是 (F, 63)，即 21 个关节。
        pipeline 中使用的 SMPL 模型期望 23 个关节 (69 维)。
        需要做 zero-padding。

        Args:
            global_orient_aa: (F, 3) axis-angle，世界坐标系 root orientation
            body_pose_aa: (F, 63) axis-angle，body pose（GVHMR: 21 joints）
            betas: (F, 10) shape
            transl: (F, 3) world translation（修正后）
        """
        with torch.no_grad():
            def to_tensor(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x).float().to(self.device)
                return x.float().to(self.device)

            global_orient_t = to_tensor(global_orient_aa)  # (F, 3)
            body_pose_t = to_tensor(body_pose_aa)  # (F, 63)
            betas_t = to_tensor(betas)  # (F, 10)
            transl_t = to_tensor(transl)  # (F, 3)

            F_len = global_orient_t.shape[0]

            # GVHMR body_pose 是 21 joints (63 dim)
            # SMPL 模型期望 23 joints (69 dim)，补 2 个零关节
            if body_pose_t.shape[-1] == 63:
                padding = torch.zeros(F_len, 6, device=self.device)
                body_pose_t = torch.cat([body_pose_t, padding], dim=-1)  # (F, 69)

            smpl_output = self.smpl_model(
                global_orient=global_orient_t,
                body_pose=body_pose_t,
                betas=betas_t,
                transl=transl_t,
                pose2rot=True,
            )

            vertices = smpl_output.vertices

        return vertices.cpu().numpy()
    
    def _update_metadata_from_cache(self, data: PipelineData, output_dir: str, existing_images: list):
        """从缓存的图像更新 metadata"""
        # 获取图像路径以判断总帧数
        if hasattr(data, 'image_paths'):
            img_paths = data.image_paths
        else:
            img_paths = data.metadata.get('image_paths', [])
        
        N = len(img_paths)
        
        if self.frame_selection_mode == 'keyframe':
            # keyframe 模式：所有帧都输出
            keyframe_info = data.metadata.get('slam_keyframe_info')
            if keyframe_info:
                keyframe_indices = sorted(set(
                    int(idx) for idx in keyframe_info['keyframe_tstamps'] if int(idx) < N
                ))
            else:
                keyframe_indices = []
            
            data.metadata['adjacent_render_info'] = {
                'total_frames': N,
                'pre_dis': None,
                'rendered_indices': list(range(N)),
                'keyframe_indices': keyframe_indices,
                'last_frame_idx': None,
                'num_processed_frames': N,
                'output_dir': output_dir,
                'frame_selection_mode': 'keyframe',
            }
        else:
            # uniform 模式：根据 pre_dis 采样
            rendered_indices = self._get_render_frame_indices(N)
            last_frame_idx = N - 1
            last_frame_saved = (
                self.config.get('render_frame_indices') is None and last_frame_idx not in rendered_indices
            )
            
            data.metadata['adjacent_render_info'] = {
                'total_frames': N,
                'pre_dis': self.pre_dis,
                'rendered_indices': rendered_indices,
                'last_frame_idx': last_frame_idx if last_frame_saved else None,
                'num_processed_frames': len(rendered_indices) + (1 if last_frame_saved else 0),
                'output_dir': output_dir,
            }

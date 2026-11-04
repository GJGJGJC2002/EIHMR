"""DepthSceneRefineComponent - 光学锚定 + 场景感知的滑动窗口精修组件 (OAR)

基于 Metric3D 深度图 + 2D 重投影约束，以滑动窗口方式精修
global_trans / body_pose。

核心流程:
1. 加载 Metric3D 模型（窗口循环期间保持常驻）
2. FK 预计算世界坐标系关节点
3. 滑动窗口遍历序列，每个窗口：
   a. 对中心帧按需推理单张深度图
   b. Adam 优化中心帧 delta_t:
      - 重投影 loss: proj(joints) vs 2D 关键点
      - 深度 loss: 相机坐标系 z_smpl vs z_scene（深度 scale 对齐）
      - 平滑 + 正则化
   c. 高斯衰减传播 delta_t 到窗口邻居帧
   d. 窗口内滑步修正（ik: 旋转矩阵加权平均 / pp_static_joint: transl 修正）
4. 卸载 Metric3D
5. 全局高斯合并写回 data 供 eval
6. 最终重投影可视化
"""

import sys
import os
import shutil
import gc
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData


def gmof(x, sigma=100):
    """Geman-McClure robust error function."""
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


class DepthSceneRefineComponent(Component):
    """OAR: Optical-Anchored Refinement with Scene Depth Constraints.

    在 WorldTransform 之后插入，精修 global_orient_w / body_pose_aa / global_trans。
    """

    COMPONENT_TYPE = "depth_scene_refine"

    DEFAULT_CONFIG = {
        # Metric3D
        'metric3d_model': 'metric3d_vit_small',
        'metric3d_root': 'thirdparty/PromptHMR/pipeline/yvanyin_metric3d_main',
        'prompthmr_root': 'thirdparty/PromptHMR',
        # 滑动窗口
        'window_size': 21,
        'stride': 5,
        'sigma': 5.0,
        # 优化
        'opt_steps': 50,
        'opt_lr': 0.01,
        # 损失权重
        'loss_reproj_w': 10.0,
        'loss_depth_w': 1.0,
        'loss_smooth_vel_w': 0.01,
        'loss_smooth_acc_w': 0.01,
        'loss_reg_w': 0.01,
        'reproj_sigma': 50,
        'depth_sigma': 1.0,
        # 功能开关
        'enable_depth_constraint': True,
        'enable_ik': True,
        # 在 IK / skating 修正前，固定 pose/betas/orient，仅优化相机系 transl_c 以贴合 2D keypoints
        'enable_transl_2d_refine': False,
        'transl_2d_opt_steps': 60,
        'transl_2d_opt_lr': 0.02,
        'transl_2d_reproj_sigma': 50.0,
        'loss_transl_2d_w': 1.0,
        'loss_transl_2d_reg_w': 0.05,
        'loss_transl_2d_smooth_w': 0.1,
        'transl_2d_conf_thr': 0.5,
        'transl_2d_optimize_z': True,
        # 窗口内滑步修正模式: 'ik' / 'pp_static_joint' / 'none'
        #   ik:              旋转矩阵空间加权平均修正 body_pose
        #   pp_static_joint: 只修正 transl（静态关节位移修正平移轨迹）
        #   none:            不做窗口内滑步修正
        'window_skating_mode': 'ik',
        'pp_static_joint_use_batch': True,
        'pp_static_joint_batch_size': 64,
        'enable_orient_refine': True,
        # 图像模型 3D body pose 纠正
        'enable_img_pose_refine': True,        # 是否用图像模型单帧 SMPL 纠正四肢 body_pose
        'img_pose_limb_joints': [1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21],  # FK-22 中的四肢关节
        'img_pose_opt_steps': 30,               # body pose 优化迭代次数
        'img_pose_opt_lr': 0.005,               # body pose 优化学习率
        'loss_img3d_w': 5.0,                    # 3D 关节 loss 权重
        'loss_img3d_sigma': 0.1,                # gmof sigma（米）
        'loss_bodypose_reg_w': 0.1,             # body pose 正则化权重
        # 深度推理批处理：窗口循环前一次性批量推理各中心帧深度图（align_mode=none 时安全）
        'depth_infer_use_batch': True,
        'depth_infer_batch_size': 8,
        # Contact IK 批处理：将 local-plane 系 IK 的 Adam 优化按窗口批量并行
        'contact_ik_use_batch': True,
        'contact_ik_batch_size': 16,
        # Depth Contact IK
        'enable_depth_contact_ik': True,        # 是否用深度图穿透信号做 Contact IK
        'contact_ik_steps': 30,                 # Contact IK 优化迭代次数
        'contact_ik_lr': 0.003,                 # Contact IK 学习率
        'contact_ik_blend': 1.0,                # 将 Contact IK 结果混回窗口 pose/transl 的比例
        'loss_contact_w': 5.0,                  # 接触关节位置 loss 权重
        'loss_contact_sigma': 0.05,             # gmof sigma（米）
        'loss_contact_bp_reg_w': 0.1,           # body pose 正则化权重
        'loss_contact_smooth_w': 0.5,           # 时序平滑权重
        'contact_ik_scene_model': 'raw_depth',  # raw_depth / ground_plane
        'contact_ik_conf_thr': 0.5,
        'contact_ik_ground_joint_ids': [10, 11],
        'contact_ik_ground_joint_offsets': {7: 0.08, 8: 0.08, 10: 0.02, 11: 0.02},
        'contact_ik_ground_offset_mode': 'fixed',  # fixed / joint_median
        'contact_ik_opt_joints': [1, 2, 4, 5, 7, 8, 10, 11],
        'contact_ik_ground_loss': 'l2',         # l2 / penetration
        'contact_ik_scale_mode': 'joint_median',  # joint_median / none
        'contact_ik_depth_align_mode': 'none',   # none / joint_scale: align Metric3D depth to current SMPL transl before plane fitting
        'contact_ik_depth_align_min_joints': 3,
        'contact_ik_ground_use_human_mask': True,
        'local_plane_radius': 90,
        'local_plane_sample_stride': 3,
        'local_plane_min_points': 40,
        'local_plane_inlier_thresh': 0.04,
        'local_plane_ransac_iters': 64,
        'local_plane_min_inlier_ratio': 0.0,
        'local_plane_max_residual': 1e9,
        'contact_ik_point_type': 'joints',       # joints / vertices; vertices uses SMPL toe/heel mesh vertices
        'contact_ik_vertex_ids': [3216, 3387, 6617, 6787],
        'contact_ik_vertex_conf_indices': [1, 0, 3, 2],  # LToe->LFoot, LHeel->LAnkle, RToe->RFoot, RHeel->RAnkle
        'contact_ik_vertex_names': ['LToe', 'LHeel', 'RToe', 'RHeel'],
        'contact_ik_vertex_offsets': [0.0, 0.0, 0.0, 0.0],
        'contact_ik_sole_max_vertices': 32,
        'contact_ik_sole_axis': 'y',
        'contact_ik_weighted_conf': False,
        'contact_ik_vel_sigma': 0.04,
        'contact_ik_height_sigma': 0.06,
        'contact_ik_min_weight': 0.05,
        'contact_ik_optimize_transl': False,
        'contact_ik_transl_dims': [0, 1, 2],
        'loss_contact_transl_reg_w': 2.0,
        'loss_contact_transl_smooth_w': 5.0,
        'josh_anchor_w': 0.0,
        'josh_anchor_sigma': 0.08,
        'ground_plane_lower_frac': 0.45,
        'ground_plane_sample_stride': 6,
        'ground_plane_ransac_iters': 96,
        'ground_plane_inlier_thresh': 0.05,
        'ground_plane_min_inliers': 200,
        'ground_plane_max_points': 6000,
        # Debug / experiment window limiting
        'debug_window_centers': None,
        'max_debug_windows': 0,
        'include_last_window_center': False,
        'enable_window_visualization': True,
        # EnDecoder
        # 杂项
        'save_depth_maps': False,
        'fps': 30,
        'device': 'cuda',
        # OAR 诊断可视化
        'vis_oar_diagnostics': False,
        'vis_oar_frame_stride': 5,
        'vis_contact_ik_compare': True,          # 保存 Contact IK 前后窗口对比，并标注 IK 关节
        # 保存窗口中心视角下的 pp_static_joint 后局部人体观测，供下游 human mesh camera refine 使用
        'save_window_mesh_observations': True,
        'window_mesh_point_type': 'joints22',
        'window_mesh_offsets': None,
    }

    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        self.endecoder = None
        self.metric3d_model = None
        self._contact_smplx_model = None
        self._contact_smplx2smpl = None

    def validate_input(self, data: PipelineData) -> bool:
        if data.smpl_params is None:
            self.logger.warning("No smpl_params found")
            return False
        sp = data.smpl_params
        if sp.global_orient_w is None or sp.body_pose_aa is None or sp.global_trans is None:
            self.logger.warning("smpl_params missing world-space fields")
            return False
        if data.camera_params is None:
            self.logger.warning("No camera_params found")
            return False
        if not data.image_paths:
            self.logger.warning("No image_paths found")
            return False
        return True

    def setup(self):
        self._setup_endecoder()
        self._is_setup = True
        self.logger.info("DepthSceneRefine (OAR) component initialized")

    def _setup_endecoder(self):
        """Load the GVHMR skeleton forward kinematics (EnDecoder.fk_v2) for FK / IK."""
        from lib.gvhmr.kinematics import EnDecoder
        self.endecoder = EnDecoder().to(self.device)

    # ------------------------------------------------------------------
    # Metric3D depth inference
    # ------------------------------------------------------------------
    def _load_metric3d(self):
        """Load Metric3D model on demand."""
        import copy
        import types

        metric3d_abs = os.path.abspath(self.config['metric3d_root'])
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        for p in [metric3d_abs, prompthmr_abs]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        # Also add the camera sub-dir for depth_utils
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)

        # Metric3D 旧代码同时兼容 mmcv/mmengine；EIHMR 只依赖 mmengine（requirements），
        # 这里补一个最小 mmcv.utils 兼容层，供 Config/DictAction/collect_env 导入使用。
        try:
            import mmcv.utils  # type: ignore  # noqa: F401
        except Exception:
            try:
                from mmengine import Config, DictAction
                mmcv_mod = types.ModuleType('mmcv')
                mmcv_utils_mod = types.ModuleType('mmcv.utils')
                mmcv_utils_mod.Config = Config
                mmcv_utils_mod.DictAction = DictAction
                mmcv_utils_mod.collect_env = lambda *args, **kwargs: {}
                mmcv_mod.utils = mmcv_utils_mod
                sys.modules.setdefault('mmcv', mmcv_mod)
                sys.modules.setdefault('mmcv.utils', mmcv_utils_mod)
            except Exception:
                pass

        old_cwd = os.getcwd()
        os.chdir(metric3d_abs)

        # Register a custom deepcopy dispatcher for ModuleType so that
        # mmcv Config.fromfile -> copy.deepcopy does not choke on numpy/etc
        # modules that appear in Metric3D config .py files.
        _deepcopy_dispatch = copy._deepcopy_dispatch
        _had_module = types.ModuleType in _deepcopy_dispatch
        _old_module_copier = _deepcopy_dispatch.get(types.ModuleType)

        def _copy_module(x, memo):
            return x  # modules are singletons

        _deepcopy_dispatch[types.ModuleType] = _copy_module
        try:
            from hubconf import metric3d_vit_small, metric3d_vit_large, metric3d_vit_giant2
            model_fn = {
                'metric3d_vit_small': metric3d_vit_small,
                'metric3d_vit_large': metric3d_vit_large,
                'metric3d_vit_giant2': metric3d_vit_giant2,
            }[self.config['metric3d_model']]
            model = model_fn(pretrain=True)
            model = model.cuda().half().eval()
            self.metric3d_model = model
            self.logger.info(f"Loaded Metric3D ({self.config['metric3d_model']})")
        finally:
            if _had_module:
                _deepcopy_dispatch[types.ModuleType] = _old_module_copier
            else:
                _deepcopy_dispatch.pop(types.ModuleType, None)
            os.chdir(old_cwd)

    def _unload_metric3d(self):
        if self.metric3d_model is not None:
            del self.metric3d_model
            self.metric3d_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def _infer_single_depth(
        self,
        image_path: str,
        intrinsics: np.ndarray,
    ) -> torch.Tensor:
        """对单帧推理 Metric3D 深度图（假设模型已加载）。

        Args:
            image_path: 图片路径
            intrinsics: (3, 3) 相机内参

        Returns:
            depth: (H, W) CPU float tensor（米制深度）
        """
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)
        from depth_utils import prep_metric3d, post_metric3d

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        calib = [fx, fy, cx, cy]
        model_version = self.config['metric3d_model']

        img = np.array(Image.open(image_path).convert('RGB'))
        rgb_prep, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(img, calib, model_version)
        rgb_batch = rgb_prep.cuda().half()

        with torch.inference_mode():
            pred_depth, confidence, _ = self.metric3d_model.inference({'input': rgb_batch})

        depth = post_metric3d(
            pred_depth, confidence if confidence is not None else None,
            pad_info, rgb_origin, intrinsic_prep
        )
        depth_cpu = depth.cpu().squeeze()

        del rgb_batch, pred_depth, confidence
        torch.cuda.empty_cache()

        return depth_cpu

    def _infer_center_depths_batch(
        self,
        centers: List[int],
        image_paths: List[str],
        intrinsics: np.ndarray,
    ) -> Dict[int, torch.Tensor]:
        """批量推理多个中心帧的 Metric3D 深度图（假设模型已加载）。

        因为深度对齐模式为 none 时，各窗口中心帧的原始深度与 pose 无关，可在
        窗口循环前一次性批量推理，显著提升 GPU 利用率。相机内参对整段序列相同，
        且各帧分辨率一致，因此 prep 后的张量形状相同、可直接堆叠成 batch。

        Args:
            centers: 需要推理深度的中心帧索引列表
            image_paths: 全序列图片路径
            intrinsics: (3, 3) 相机内参

        Returns:
            dict: center -> (H, W) CPU float tensor（米制深度）
        """
        prompthmr_abs = os.path.abspath(self.config['prompthmr_root'])
        cam_dir = os.path.join(prompthmr_abs, 'pipeline', 'camera')
        if cam_dir not in sys.path:
            sys.path.insert(0, cam_dir)
        from depth_utils import prep_metric3d, post_metric3d_batch

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        calib = [fx, fy, cx, cy]
        model_version = self.config['metric3d_model']
        batch_size = max(1, int(self.config.get('depth_infer_batch_size', 8)))

        result: Dict[int, torch.Tensor] = {}
        for chunk_start in range(0, len(centers), batch_size):
            chunk = centers[chunk_start:chunk_start + batch_size]
            rgb_list = []
            pad_info = None
            intrinsic_prep = None
            origin_shape = None
            for c in chunk:
                img = np.array(Image.open(image_paths[c]).convert('RGB'))
                rgb_prep, intrinsic_prep, pad_info, rgb_origin = prep_metric3d(img, calib, model_version)
                origin_shape = rgb_origin.shape[:2]
                rgb_list.append(rgb_prep)

            rgb_batch = torch.cat(rgb_list, dim=0).cuda().half()  # (B, 3, H, W)
            with torch.inference_mode():
                pred_depth, confidence, _ = self.metric3d_model.inference({'input': rgb_batch})

            depth_b = post_metric3d_batch(
                pred_depth, confidence if confidence is not None else None,
                pad_info, None, intrinsic_prep, resize_shape=origin_shape,
            )  # (B, H, W) or (H, W) when B==1
            if depth_b.dim() == 2:
                depth_b = depth_b.unsqueeze(0)
            depth_b = depth_b.detach().cpu().float()
            for i, c in enumerate(chunk):
                result[c] = depth_b[i].clone()

            del rgb_batch, pred_depth, confidence, depth_b
            torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_w2c(R_wc: torch.Tensor, T_wc: torch.Tensor):
        """Convert c2w to w2c: R_cw = R_wc^T, t_cw = -R_cw @ T_wc."""
        R_cw = R_wc.transpose(-1, -2)
        t_cw = -(R_cw @ T_wc.unsqueeze(-1)).squeeze(-1)
        return R_cw, t_cw

    @staticmethod
    def _project_joints(j3d_world, R_cw, t_cw, K):
        """World joints -> 2D pixel coords. All (F, J, 3) or (F, 3, 3)."""
        j_cam = torch.einsum('fij,fkj->fki', R_cw, j3d_world) + t_cw[:, None, :]
        pj = torch.einsum('ij,fkj->fki', K, j_cam)
        pj_2d = pj[..., :2] / (pj[..., 2:3] + 1e-6)
        return pj_2d, j_cam

    @staticmethod
    def _inverse_unpad_K(K_pad, img_w: int, img_h: int, img_size: float = 896.0):
        """Convert PromptHMR 896-padded K back to original-image coordinates."""
        K_full = K_pad.clone() if isinstance(K_pad, torch.Tensor) else torch.from_numpy(K_pad).float().clone()
        scale = img_size / float(max(img_w, img_h))
        offset_x = (img_size - scale * float(img_w)) / 2.0
        offset_y = (img_size - scale * float(img_h)) / 2.0
        K_orig = K_full.clone()
        K_orig[:2, :] /= scale
        K_orig[0, 2] = (K_full[0, 2] - offset_x) / scale
        K_orig[1, 2] = (K_full[1, 2] - offset_y) / scale
        return K_orig

    @staticmethod
    def _inverse_unpad_kp2d(kp2d, img_w: int, img_h: int, img_size: float = 896.0):
        """Convert PromptHMR 896-padded keypoints back to original-image coordinates."""
        kp = kp2d.clone() if isinstance(kp2d, torch.Tensor) else torch.from_numpy(kp2d).float().clone()
        scale = img_size / float(max(img_w, img_h))
        offset = torch.tensor(
            [(img_size - scale * float(img_w)) / 2.0, (img_size - scale * float(img_h)) / 2.0],
            dtype=kp.dtype,
            device=kp.device,
        )
        kp[..., :2] = (kp[..., :2] - offset) / scale
        return kp

    def _refine_incam_transl_with_2d(
        self,
        global_orient_c: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        transl_c: torch.Tensor,
        kp2d: torch.Tensor,
        K: torch.Tensor,
    ):
        """固定 pose/betas/root orient，只优化 camera-space transl_c 贴合 2D keypoints。

        kp2d 与 K 必须处于同一图像坐标系；PromptHMR video 路径下通常是 896 padded 坐标。
        """
        if kp2d is None:
            return transl_c, {'enabled': False, 'reason': 'no_kp2d'}
        if global_orient_c is None or transl_c is None:
            return transl_c, {'enabled': False, 'reason': 'no_incam_params'}

        device = torch.device(self.device)
        go_c = global_orient_c.to(device)
        bp = body_pose_aa.to(device)
        beta = betas.to(device)
        trans_init = transl_c.to(device)
        kp = kp2d.to(device)
        K_dev = K.to(device)
        F_total = trans_init.shape[0]

        fk_coco_pairs = [
            (1, 11), (2, 12), (4, 13), (5, 14), (7, 15), (8, 16),
            (16, 5), (17, 6), (18, 7), (19, 8), (20, 9), (21, 10),
        ]
        fk_pair_idxs = [p[0] for p in fk_coco_pairs]
        coco_pair_idxs = [p[1] for p in fk_coco_pairs]

        with torch.no_grad():
            zero_trans = torch.zeros_like(trans_init)
            base_joints = self.endecoder.fk_v2(
                body_pose=bp.unsqueeze(0),
                betas=beta.unsqueeze(0),
                global_orient=go_c.unsqueeze(0),
                transl=zero_trans.unsqueeze(0),
            )[0]  # (F, 22, 3), camera-oriented but root untranslated

        def _project_with_trans(trans):
            joints_cam = base_joints + trans[:, None, :]
            proj = torch.einsum('ij,fkj->fki', K_dev, joints_cam)
            proj_2d = proj[..., :2] / (proj[..., 2:3] + 1e-6)
            return proj_2d

        conf_thr = float(self.config.get('transl_2d_conf_thr', 0.5))
        gt_2d = kp[:, coco_pair_idxs, :2]
        conf = kp[:, coco_pair_idxs, 2]
        conf_mask = conf > conf_thr
        if not conf_mask.any():
            return transl_c, {'enabled': False, 'reason': 'no_valid_kp2d'}

        with torch.no_grad():
            proj_init = _project_with_trans(trans_init)[:, fk_pair_idxs]
            init_err = torch.norm(proj_init - gt_2d, dim=-1)[conf_mask]
            init_mean = float(init_err.mean().detach().cpu())
            init_median = float(init_err.median().detach().cpu())

        delta = torch.zeros_like(trans_init, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=float(self.config.get('transl_2d_opt_lr', 0.02)))
        opt_z = bool(self.config.get('transl_2d_optimize_z', True))
        steps = int(self.config.get('transl_2d_opt_steps', 60))
        reproj_sigma = float(self.config.get('transl_2d_reproj_sigma', 50.0))
        w_reproj = float(self.config.get('loss_transl_2d_w', 1.0))
        w_reg = float(self.config.get('loss_transl_2d_reg_w', 0.05))
        w_smooth = float(self.config.get('loss_transl_2d_smooth_w', 0.1))

        for _ in range(steps):
            optimizer.zero_grad()
            delta_eff = delta
            if not opt_z:
                delta_eff = delta_eff.clone()
                delta_eff[:, 2] = 0.0
            trans = trans_init + delta_eff
            proj_2d = _project_with_trans(trans)[:, fk_pair_idxs]
            reproj = gmof(proj_2d - gt_2d, sigma=reproj_sigma).sum(dim=-1)
            loss_reproj = (reproj * conf_mask.float()).sum() / conf_mask.float().sum().clamp_min(1.0)
            loss = w_reproj * loss_reproj / 1000.0
            loss = loss + w_reg * (delta_eff.pow(2).sum(dim=-1).mean())
            if F_total >= 3 and w_smooth > 0:
                acc = delta_eff[2:] + delta_eff[:-2] - 2.0 * delta_eff[1:-1]
                loss = loss + w_smooth * acc.pow(2).sum(dim=-1).mean()
            z_penalty = torch.relu(0.2 - trans[:, 2]).pow(2).mean()
            loss = loss + 10.0 * z_penalty
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            delta_eff = delta.detach()
            if not opt_z:
                delta_eff = delta_eff.clone()
                delta_eff[:, 2] = 0.0
            trans_refined = (trans_init + delta_eff).detach()
            proj_final = _project_with_trans(trans_refined)[:, fk_pair_idxs]
            final_err = torch.norm(proj_final - gt_2d, dim=-1)[conf_mask]
            stats = {
                'enabled': True,
                'init_mean_px': init_mean,
                'init_median_px': init_median,
                'final_mean_px': float(final_err.mean().detach().cpu()),
                'final_median_px': float(final_err.median().detach().cpu()),
                'delta_mean_m': float(delta_eff.norm(dim=-1).mean().detach().cpu()),
                'delta_max_m': float(delta_eff.norm(dim=-1).max().detach().cpu()),
                'optimize_z': opt_z,
                'steps': steps,
            }

        return trans_refined.cpu(), stats

    # ------------------------------------------------------------------
    # Window-level SMPL rendering (5 frames per window)
    # ------------------------------------------------------------------
    def _load_smplx_for_render(self):
        """一次性加载 SMPL-X 模型和渲染器组件，返回 (smplx_model, smplx2smpl, faces_smpl)。"""
        from lib.gvhmr import ASSETS
        from lib.gvhmr.body_model import make_smplx
        smplx_model = make_smplx("supermotion").to(self.device)
        smplx2smpl = torch.load(ASSETS / "smplx2smpl_sparse.pt").to(self.device)
        faces_smpl = make_smplx("smpl").faces
        return smplx_model, smplx2smpl, faces_smpl

    def _get_contact_smplx_assets(self):
        """Load/cache the differentiable SMPL-X model and SMPL vertex mapper for vertex Contact IK."""
        if self._contact_smplx_model is None or self._contact_smplx2smpl is None:
            smplx_model, smplx2smpl, _ = self._load_smplx_for_render()
            smplx_model.eval()
            self._contact_smplx_model = smplx_model
            self._contact_smplx2smpl = smplx2smpl.coalesce() if smplx2smpl.is_sparse else smplx2smpl
        return self._contact_smplx_model, self._contact_smplx2smpl

    @staticmethod
    def _normalize_contact_vertex_defs(vertex_ids, conf_indices=None, names=None, offsets=None):
        vertex_ids = [int(v) for v in (vertex_ids or [])]
        if not vertex_ids:
            return []
        if conf_indices is None:
            conf_indices = [1, 0, 3, 2][:len(vertex_ids)]
        conf_indices = [int(x) for x in conf_indices]
        if len(conf_indices) < len(vertex_ids):
            conf_indices = conf_indices + [conf_indices[-1] if conf_indices else 1] * (len(vertex_ids) - len(conf_indices))
        if names is None:
            names = [f'V{v}' for v in vertex_ids]
        names = [str(x) for x in names]
        if len(names) < len(vertex_ids):
            names = names + [f'V{v}' for v in vertex_ids[len(names):]]
        if offsets is None:
            offsets = [0.0] * len(vertex_ids)
        offsets = [float(x) for x in offsets]
        if len(offsets) < len(vertex_ids):
            offsets = offsets + [0.0] * (len(vertex_ids) - len(offsets))
        return [
            {'vid': vid, 'conf_idx': conf_indices[i], 'name': names[i], 'offset': offsets[i]}
            for i, vid in enumerate(vertex_ids)
        ]

    def _build_sole_vertex_defs(self, max_vertices_per_foot: int = 32):
        """Build fixed SMPL sole contact vertices from foot/toe segmentation and template low-height vertices."""
        import json
        from lib.gvhmr import ASSETS
        from lib.gvhmr.body_model import make_smplx
        smpl_model = make_smplx("smpl")
        with open(ASSETS / "smpl_vert_segmentation.json", 'r') as f:
            seg = json.load(f)
        v_template = smpl_model.bm.v_template.detach().cpu() if hasattr(smpl_model, 'bm') else None
        if v_template is None:
            return self._normalize_contact_vertex_defs(
                self.config.get('contact_ik_vertex_ids', [3216, 3387, 6617, 6787]),
                self.config.get('contact_ik_vertex_conf_indices', [1, 0, 3, 2]),
                self.config.get('contact_ik_vertex_names', ['LToe', 'LHeel', 'RToe', 'RHeel']),
                self.config.get('contact_ik_vertex_offsets', [0.0, 0.0, 0.0, 0.0]),
            )

        axis_name = str(self.config.get('contact_ik_sole_axis', 'y')).lower()
        axis = {'x': 0, 'y': 1, 'z': 2}.get(axis_name, 1)
        defs = []
        sides = [
            ('L', ['leftFoot', 'leftToeBase'], 1),
            ('R', ['rightFoot', 'rightToeBase'], 3),
        ]
        per_foot = max(1, int(max_vertices_per_foot))
        for side_name, keys, conf_idx in sides:
            ids = sorted(set(sum([seg.get(k, []) for k in keys], [])))
            if not ids:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long)
            coords = v_template[ids_t, axis]
            keep = ids_t[torch.argsort(coords)[:min(per_foot, len(ids))]].tolist()
            # Keep the proven toe/heel representatives even if template sorting misses them.
            keep.extend([3216, 3387] if side_name == 'L' else [6617, 6787])
            keep = sorted(set(int(x) for x in keep))
            for rank, vid in enumerate(keep):
                defs.append({'vid': vid, 'conf_idx': conf_idx, 'name': f'{side_name}Sole{rank:02d}', 'offset': 0.0})
        return defs

    @staticmethod
    def _forward_selected_smpl_vertices(smplx_model, smplx2smpl_sel, global_orient, body_pose, betas, transl):
        """Forward SMPL-X and return selected mapped SMPL vertices, shape (F, Nv, 3)."""
        out = smplx_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
        verts = []
        mapper = smplx2smpl_sel.coalesce() if smplx2smpl_sel.is_sparse else smplx2smpl_sel
        for fidx in range(out.vertices.shape[0]):
            if mapper.is_sparse:
                verts.append(torch.sparse.mm(mapper, out.vertices[fidx]))
            else:
                verts.append(mapper @ out.vertices[fidx])
        return torch.stack(verts, dim=0)

    def _render_window_smpl(
        self,
        center: int,
        global_orient_w: torch.Tensor,   # (F, 3) aa, full sequence
        body_pose_aa: torch.Tensor,       # (win_len, 63) window-local body pose
        betas: torch.Tensor,              # (F, 10) full sequence
        win_transl: torch.Tensor,         # (win_len, 3) window-local transl
        win_start: int,
        win_end: int,
        R_cw: torch.Tensor,               # (F, 3, 3)
        t_cw: torch.Tensor,               # (F, 3)
        K: torch.Tensor,                   # (3, 3)
        image_paths: List[str],
        output_dir: str,
        tag: str = 'pre',
        render_offsets: List[int] = None,
        smplx_model=None,
        smplx2smpl=None,
        faces_smpl=None,
        renderer=None,
        static_conf_logits: Optional[torch.Tensor] = None,
        ik_joint_ids: Optional[List[int]] = None,
        contact_joint_ids: Optional[List[int]] = None,
        marker_note: str = '',
        save: bool = True,
    ):
        """渲染窗口内指定偏移帧的 SMPL mesh 到中心帧视角，保存 debug 图。

        对中心帧，渲染 render_offsets 指定偏移的帧（默认 -10,-5,0,+5,+10），
        所有 SMPL 用 w2c 投影到中心帧的相机坐标系下。

        Args:
            body_pose_aa: (win_len, 63) 窗口级 body pose（可能被 IK 修正过）
            global_orient_w: (F, 3) 全局序列的 orient（用全局索引访问）
            betas: (F, 10) 全局序列的 shape（用全局索引访问）
            win_transl: (win_len, 3) 窗口内独立 transl
            smplx_model, smplx2smpl, faces_smpl: 外部传入避免重复加载
            renderer: 外部传入的 Pytorch3dRenderer 实例
        """
        import cv2

        if render_offsets is None:
            render_offsets = [-10, -5, 0, 5, 10]

        vis_dir = os.path.join(output_dir, f'oar_window_{tag}')
        if save:
            os.makedirs(vis_dir, exist_ok=True)

        device = torch.device(self.device)

        # 中心帧的 w2c
        R_w2c_center = R_cw[center].to(device)
        t_w2c_center = t_cw[center].to(device)

        img_bg = cv2.imread(image_paths[center])

        # 颜色表：不同偏移用不同颜色（从远过去到远未来：蓝→绿→红）
        # 显式覆盖长窗口常用的 ±20/±30，避免它们落到灰色 default 后看起来像没渲染。
        offset_colors = {
            -30: [0.05, 0.15, 1.00],
            -20: [0.10, 0.55, 1.00],
            -10: [0.20, 0.90, 0.95],
            -5:  [0.30, 0.70, 0.90],
            0:   [0.20, 1.00, 0.20],
            5:   [0.90, 0.70, 0.30],
            10:  [1.00, 0.45, 0.10],
            20:  [1.00, 0.10, 0.55],
            30:  [0.80, 0.10, 1.00],
        }
        default_color = [0.9, 0.9, 0.1]
        static_joint_ids = [7, 10, 8, 11, 20, 21]
        static_joint_names = ['LAnk', 'LFoot', 'RAnk', 'RFoot', 'LWri', 'RWri']
        joint_names = {
            0: 'Pelvis', 1: 'LHip', 2: 'RHip', 3: 'Spine1', 4: 'LKnee', 5: 'RKnee',
            6: 'Spine2', 7: 'LAnk', 8: 'RAnk', 9: 'Spine3', 10: 'LFoot', 11: 'RFoot',
            12: 'Neck', 13: 'LCollar', 14: 'RCollar', 15: 'Head', 16: 'LShoulder',
            17: 'RShoulder', 18: 'LElbow', 19: 'RElbow', 20: 'LWrist', 21: 'RWrist',
        }
        ik_joint_ids = [int(j) for j in (ik_joint_ids or [])]
        contact_joint_ids = [int(j) for j in (contact_joint_ids or [])]
        rendered_offsets = []

        canvas = img_bg.copy()

        for offset in render_offsets:
            global_idx = center + offset
            if global_idx < win_start or global_idx >= win_end:
                continue
            if global_idx < 0 or global_idx >= len(image_paths):
                continue

            local_idx = global_idx - win_start

            with torch.no_grad():
                out = smplx_model(
                    global_orient=global_orient_w[global_idx:global_idx+1].to(device),
                    body_pose=body_pose_aa[local_idx:local_idx+1].to(device),
                    betas=betas[global_idx:global_idx+1].to(device),
                    transl=win_transl[local_idx:local_idx+1].to(device),
                )
                verts_world = torch.matmul(smplx2smpl, out.vertices[0])

            verts_cam = torch.einsum('ij,vj->vi', R_w2c_center, verts_world) + t_w2c_center

            color = offset_colors.get(offset, default_color)
            try:
                canvas = renderer.render_mesh(verts_cam, canvas, colors=color)
                rendered_offsets.append(offset)
            except Exception as e:
                self.logger.warning(f"Render failed for offset {offset}: {e}")
                continue

            if static_conf_logits is not None or ik_joint_ids or contact_joint_ids:
                try:
                    logits = None
                    if static_conf_logits is not None:
                        if static_conf_logits.shape[0] == len(image_paths):
                            logits = static_conf_logits[global_idx]
                        elif static_conf_logits.shape[0] == (win_end - win_start):
                            logits = static_conf_logits[local_idx]
                    static_conf = torch.sigmoid(logits.to(device)) if logits is not None else None

                    joints_world = self.endecoder.fk_v2(
                        body_pose=body_pose_aa[local_idx:local_idx+1].unsqueeze(0).to(device),
                        betas=betas[global_idx:global_idx+1].unsqueeze(0).to(device),
                        global_orient=global_orient_w[global_idx:global_idx+1].unsqueeze(0).to(device),
                        transl=win_transl[local_idx:local_idx+1].unsqueeze(0).to(device),
                    )[0, 0]
                    joints_cam = torch.einsum('ij,kj->ki', R_w2c_center, joints_world) + t_w2c_center
                    proj = torch.einsum('ij,kj->ki', K.to(device), joints_cam)
                    uv = proj[:, :2] / (proj[:, 2:3] + 1e-6)

                    def _draw_joint_marker(jid: int, color, label: str, shape: str = 'circle'):
                        if jid < 0 or jid >= uv.shape[0]:
                            return
                        u, v = uv[jid]
                        if not (torch.isfinite(u) and torch.isfinite(v)):
                            return
                        x, y = int(round(u.item())), int(round(v.item()))
                        if not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
                            return
                        if shape == 'square':
                            cv2.rectangle(canvas, (x - 7, y - 7), (x + 7, y + 7), color, 2)
                        else:
                            cv2.circle(canvas, (x, y), 7, color, 2)
                        cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1)
                        cv2.putText(
                            canvas, label, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1,
                        )

                    # Contact IK 的目标接触关节：黄色圆圈。
                    for jid in contact_joint_ids:
                        _draw_joint_marker(jid, (0, 220, 255), f"CONTACT:{joint_names.get(jid, jid)}", 'circle')

                    # Contact IK 实际优化的 body-pose 关节：青色方框。
                    for jid in ik_joint_ids:
                        _draw_joint_marker(jid, (255, 255, 0), f"IK:{joint_names.get(jid, jid)}", 'square')

                    if static_conf is not None:
                        for si, jid in enumerate(static_joint_ids):
                            if si >= static_conf.shape[0] or static_conf[si].item() <= 0.5:
                                continue
                            _draw_joint_marker(
                                jid,
                                (255, 0, 255),
                                f"STATIC:{static_joint_names[si]}:{static_conf[si].item():.2f}",
                                'circle',
                            )
                except Exception as e:
                    self.logger.warning(f"Joint marker failed for offset {offset}: {e}")

        # 标题和图例
        rendered_text = ','.join([f'{x:+d}' for x in rendered_offsets]) if rendered_offsets else 'none'
        title = f"Frame {center} | Window [{win_start}:{win_end}] | {tag.upper()} | rendered {len(rendered_offsets)}/{len(render_offsets)}: {rendered_text}"
        cv2.putText(canvas, title, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_legend = 50
        for offset in render_offsets:
            color = offset_colors.get(offset, default_color)
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            label = f"offset={offset:+d}"
            if offset == 0:
                label += " (center)"
            cv2.putText(canvas, label, (10, y_legend),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1)
            y_legend += 18
        cv2.putText(canvas, "Magenta=static joints used by pp_static_joint", (10, y_legend),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        y_legend += 18
        if ik_joint_ids or contact_joint_ids:
            cv2.putText(canvas, "Cyan square=IK optimized joints; Yellow circle=contact target joints", (10, y_legend),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y_legend += 18
        if marker_note:
            cv2.putText(canvas, marker_note, (10, y_legend),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        if save:
            out_path = os.path.join(vis_dir, f'{center:05d}.jpg')
            cv2.imwrite(out_path, canvas)
        return canvas

    def _save_window_concat(
        self,
        center: int,
        pre_canvas,
        post_canvas,
        output_dir: str,
        metrics_pre: Optional[Dict[str, Any]] = None,
        metrics_post: Optional[Dict[str, Any]] = None,
        subdir: str = 'oar_window_compare',
        left_label: str = 'BEFORE pp_static_joint',
        right_label: str = 'AFTER pp_static_joint',
    ):
        """Save side-by-side window visualisation: left=pre, right=post."""
        import cv2
        if pre_canvas is None or post_canvas is None:
            return
        vis_dir = os.path.join(output_dir, subdir)
        os.makedirs(vis_dir, exist_ok=True)
        h = min(pre_canvas.shape[0], post_canvas.shape[0])
        w = min(pre_canvas.shape[1], post_canvas.shape[1])
        pre = pre_canvas[:h, :w].copy()
        post = post_canvas[:h, :w].copy()
        cv2.putText(pre, left_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(post, right_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        def _draw_metrics(canvas, metrics, x0=10, y0=55):
            if not metrics:
                return
            lines = [
                f"jitter={metrics.get('jitter', float('nan')):.3f}",
                f"fs_static={metrics.get('fs_static', float('nan')):.3f} mm",
                f"fs_all={metrics.get('fs_all', float('nan')):.3f} mm",
                f"static_contacts={int(metrics.get('static_contact_count', 0))}",
            ]
            y = y0
            for line in lines:
                cv2.putText(canvas, line, (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(canvas, line, (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                y += 18

        _draw_metrics(pre, metrics_pre)
        _draw_metrics(post, metrics_post)
        if metrics_pre and metrics_post:
            delta_fs = metrics_post.get('fs_static', float('nan')) - metrics_pre.get('fs_static', float('nan'))
            delta_jitter = metrics_post.get('jitter', float('nan')) - metrics_pre.get('jitter', float('nan'))
            summary = f"delta: fs_static={delta_fs:+.3f} mm, jitter={delta_jitter:+.3f}"
            cv2.putText(post, summary, (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

        combined = np.concatenate([pre, post], axis=1)
        out_path = os.path.join(vis_dir, f'{center:05d}.jpg')
        cv2.imwrite(out_path, combined)

    def _compute_window_skating_metrics(
        self,
        global_orient_w: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        static_conf_logits: Optional[torch.Tensor],
        smplx_model,
        smplx2smpl,
    ) -> Dict[str, Any]:
        """Compute window-local jitter and static-contact foot sliding proxy."""
        device = torch.device(self.device)
        win_len = win_transl.shape[0]
        metrics = {
            'jitter': float('nan'),
            'fs_static': float('nan'),
            'fs_all': float('nan'),
            'static_contact_count': 0,
        }

        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=body_pose_aa.unsqueeze(0).to(device),
                betas=betas.unsqueeze(0).to(device),
                global_orient=global_orient_w.unsqueeze(0).to(device),
                transl=win_transl.unsqueeze(0).to(device),
            )[0]

            if win_len >= 4:
                jitter = torch.norm(
                    (joints_world[3:] - 3 * joints_world[2:-1] + 3 * joints_world[1:-2] - joints_world[:-3])
                    * (float(self.config.get('fps', 30)) ** 3),
                    dim=2,
                ).mean(dim=-1) / 10.0
                metrics['jitter'] = float(jitter.mean().detach().cpu())

            out = smplx_model(
                global_orient=global_orient_w.to(device),
                body_pose=body_pose_aa.to(device),
                betas=betas.to(device),
                transl=win_transl.to(device),
            )
            verts = torch.stack([torch.matmul(smplx2smpl, v) for v in out.vertices])
            foot_idxs = [3216, 3387, 6617, 6787]
            if verts.shape[0] >= 2 and verts.shape[1] > max(foot_idxs):
                foot_disp = (verts[1:, foot_idxs] - verts[:-1, foot_idxs]).norm(2, dim=-1) * 1000.0
                metrics['fs_all'] = float(foot_disp.mean().detach().cpu())
                if static_conf_logits is not None and static_conf_logits.shape[0] >= 2:
                    foot_static = static_conf_logits[:-1, :4].to(device) > 0
                    selected = foot_disp[foot_static]
                    metrics['static_contact_count'] = int(selected.numel())
                    if selected.numel() > 0:
                        metrics['fs_static'] = float(selected.mean().detach().cpu())

        return metrics

    def _build_window_mesh_observation(
        self,
        center: int,
        start: int,
        end: int,
        global_orient_w: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        static_conf_logits: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Save corrected window human points in the center camera/view frame.

        `win_transl` remains the window-local corrected translation.  The saved
        `transl_center_cam` and `joints_center_cam` are expressed in the window
        center view and intentionally are not projected back to a global world.
        """
        device = torch.device(self.device)
        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        win_bp = body_pose_aa.to(device)
        win_tr = win_transl.to(device)
        R_center = R_cw[center].to(device)
        t_center = t_cw[center].to(device)

        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=win_bp.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]
            joints_center_cam = torch.einsum('ij,fkj->fki', R_center, joints_world) + t_center[None, None, :]
            transl_center_cam = torch.einsum('ij,fj->fi', R_center, win_tr) + t_center[None, :]

        frame_indices = torch.arange(start, end, dtype=torch.long)
        offsets = self.config.get('window_mesh_offsets', None)
        if offsets is not None:
            keep = torch.zeros(end - start, dtype=torch.bool)
            for off in offsets:
                local_idx = int(center + int(off) - start)
                if 0 <= local_idx < keep.numel():
                    keep[local_idx] = True
            if keep.any():
                frame_indices = frame_indices[keep]
                joints_center_cam = joints_center_cam.detach().cpu()[keep]
                transl_center_cam = transl_center_cam.detach().cpu()[keep]
            else:
                joints_center_cam = joints_center_cam.detach().cpu()
                transl_center_cam = transl_center_cam.detach().cpu()
        else:
            joints_center_cam = joints_center_cam.detach().cpu()
            transl_center_cam = transl_center_cam.detach().cpu()

        obs = {
            'center': int(center),
            'start': int(start),
            'end': int(end),
            'frame_indices': frame_indices.cpu(),
            'point_type': self.config.get('window_mesh_point_type', 'joints22'),
            'transl_center_cam': transl_center_cam,
            'joints_center_cam': joints_center_cam,
        }
        if static_conf_logits is not None:
            obs['static_conf_logits'] = static_conf_logits[start:end].detach().cpu()
        return obs

    # ------------------------------------------------------------------
    # IK correction (旋转矩阵空间加权平均)
    # ------------------------------------------------------------------
    def _apply_ik_correction(
        self,
        global_orient_w: torch.Tensor,  # (F, 3) aa
        body_pose_aa: torch.Tensor,      # (F, 63)
        betas: torch.Tensor,             # (F, 10)
        global_trans: torch.Tensor,      # (F, 3)
        static_conf_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """旋转矩阵空间加权平均 IK：对静态关节的 local rotation 做帧间 SLERP。

        与原 process_ik（位置空间 rollout merge + CCD IK）不同，
        直接在旋转矩阵空间做加权平均修改 body_pose：
        1. FK 得到 local rotation matrix (B, L, 22, 3, 3)
        2. 对静态关节（脚踝、脚趾、手腕）沿运动链的 local rotation，
           用 static_conf 在相邻帧之间做加权平均
        3. 转回 axis-angle 作为修正后的 body_pose

        Returns:
            corrected body_pose: (F, 63)
        """
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

        device = self.device
        F_total = global_orient_w.shape[0]

        # 构造 static_conf (F, 6) → sigmoid
        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits.to(device))  # (F, 6)
        else:
            static_conf = torch.zeros(F_total, 6, device=device)

        # FK 获取 local rotation matrices
        with torch.no_grad():
            _, local_mat, _ = self.endecoder.fk_v2(
                body_pose=body_pose_aa.unsqueeze(0).to(device),
                betas=betas.unsqueeze(0).to(device),
                global_orient=global_orient_w.unsqueeze(0).to(device),
                transl=global_trans.unsqueeze(0).to(device),
                get_intermediate=True,
            )
            # local_mat: (1, F, 22, 4, 4), local_rotmat: (1, F, 22, 3, 3)

        local_rotmat = local_mat[0, :, :, :3, :3].clone()  # (F, 22, 3, 3)

        # 静态关节 IDs 和对应的运动链关节
        # joint_ids = [7, 10, 8, 11, 20, 21]
        # 对应 static_conf 列: 0=L_Ankle(7), 1=L_Foot(10), 2=R_Ankle(8),
        #                       3=R_Foot(11), 4=L_Wrist(20), 5=R_Wrist(21)
        # 修正运动链上的关节（不修 root=0）:
        chain_map = {
            0: [1, 4, 7, 10],      # L_Ankle: L_Hip(1)→L_Knee(4)→L_Ankle(7)→L_Foot(10)
            1: [1, 4, 7, 10],      # L_Foot: 同上
            2: [2, 5, 8, 11],      # R_Ankle: R_Hip(2)→R_Knee(5)→R_Ankle(8)→R_Foot(11)
            3: [2, 5, 8, 11],      # R_Foot: 同上
            4: [13, 16, 18, 20],   # L_Wrist: L_Collar(13)→L_Shoulder(16)→L_Elbow(18)→L_Wrist(20)
            5: [14, 17, 19, 21],   # R_Wrist: R_Collar(14)→R_Shoulder(17)→R_Elbow(19)→R_Wrist(21)
        }

        # 旋转矩阵空间 rollout merge:
        # 对每帧 i，如果 static_conf[i-1, j] 高，则链上关节的 local_rotmat
        # 趋近前一帧（已平均）的值
        # R_new[i] = (1 - c) * R_orig[i] + c * R_prev[i-1]
        # 在旋转矩阵空间做加权平均后 SVD 投影回 SO(3)
        merged_rotmat = local_rotmat.clone()  # (F, 22, 3, 3)

        for i in range(1, F_total):
            for conf_idx, chain_joints in chain_map.items():
                c = static_conf[min(i - 1, static_conf.shape[0] - 1), conf_idx].item()
                if c < 0.1:
                    continue
                for jid in chain_joints:
                    if jid >= merged_rotmat.shape[1]:
                        continue
                    R_prev = merged_rotmat[i - 1, jid]  # (3, 3)
                    R_curr = local_rotmat[i, jid]         # (3, 3)
                    # 旋转矩阵加权平均: R_avg = (1-c)*R_curr + c*R_prev，然后 SVD 投影到 SO(3)
                    R_avg = (1.0 - c) * R_curr + c * R_prev
                    U, _, Vh = torch.linalg.svd(R_avg)
                    R_proj = U @ Vh
                    # 确保 det > 0 (proper rotation)
                    if torch.det(R_proj) < 0:
                        U[:, -1] *= -1
                        R_proj = U @ Vh
                    merged_rotmat[i, jid] = R_proj

        # 转回 axis-angle: body_pose = joints 1-21 的 local rotation
        body_rotmat = merged_rotmat[:, 1:, :, :]  # (F, 21, 3, 3)
        body_aa = matrix_to_axis_angle(body_rotmat)  # (F, 21, 3)
        corrected_body_pose = body_aa.reshape(F_total, 63).cpu()

        self.logger.debug(
            f"[RotMat IK] Corrected body_pose for {F_total} frames, "
            f"delta_norm={((corrected_body_pose - body_pose_aa.cpu()).norm(dim=-1)).mean():.4f}"
        )

        return corrected_body_pose

    def _apply_pp_static_joint(
        self,
        global_orient_w: torch.Tensor,  # (F, 3) aa
        body_pose_aa: torch.Tensor,      # (F, 63)
        betas: torch.Tensor,             # (F, 10)
        global_trans: torch.Tensor,      # (F, 3)
        static_conf_logits: Optional[torch.Tensor],
        center_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Run GVHMR pp_static_joint to correct transl (not body_pose).

        Args:
            center_idx: 窗口中心帧的局部索引。若提供，则以中心帧为锚点对齐，
                        即保证中心帧 transl 不变，只修正其他帧的相对位移。
                        若为 None，则保持 pp_static_joint 原始行为（第一帧对齐）。

        Returns:
            corrected transl: (F, 3)
        """
        result = self._apply_pp_static_joint_batch(
            global_orient_w.unsqueeze(0),
            body_pose_aa.unsqueeze(0),
            betas.unsqueeze(0),
            global_trans.unsqueeze(0),
            static_conf_logits.unsqueeze(0) if static_conf_logits is not None else None,
            center_idxs=(torch.tensor([center_idx], dtype=torch.long) if center_idx is not None else None),
        )[0]
        return result

    def _apply_pp_static_joint_batch(
        self,
        global_orient_w: torch.Tensor,  # (B, F, 3) aa
        body_pose_aa: torch.Tensor,      # (B, F, 63)
        betas: torch.Tensor,             # (B, F, 10)
        global_trans: torch.Tensor,      # (B, F, 3)
        static_conf_logits: Optional[torch.Tensor],
        center_idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batch GVHMR pp_static_joint for windows with the same length."""
        from lib.gvhmr.kinematics import pp_static_joint

        batch_size, win_len = global_trans.shape[:2]
        outputs = {
            "pred_smpl_params_global": {
                "global_orient": global_orient_w.to(self.device),
                "body_pose": body_pose_aa.to(self.device),
                "betas": betas.to(self.device),
                "transl": global_trans.to(self.device),
            },
            "static_conf_logits": (
                static_conf_logits.to(self.device)
                if static_conf_logits is not None
                else torch.zeros(batch_size, win_len, 6, device=self.device)
            ),
        }
        post_w_transl = pp_static_joint(outputs, self.endecoder).detach().cpu()  # (B, F, 3)
        global_trans_cpu = global_trans.detach().cpu()

        if center_idxs is not None:
            # pp_static_joint 以第一帧为锚点 cumsum，并做了地面对齐 (y -= min_y)。
            # 窗口模式希望以每个窗口中心帧为锚点，保持中心帧 transl 不变。
            center_idxs = center_idxs.detach().cpu().long()
            batch_idxs = torch.arange(batch_size, dtype=torch.long)
            anchor_offset = global_trans_cpu[batch_idxs, center_idxs] - post_w_transl[batch_idxs, center_idxs]
            post_w_transl = post_w_transl + anchor_offset[:, None, :]
        else:
            # 全局调用：只还原 y 轴（地面对齐由下游处理）
            post_w_transl[..., 1] = global_trans_cpu[..., 1]

        return post_w_transl

    # ------------------------------------------------------------------
    # Depth Contact 检测 + Debug 可视化
    # ------------------------------------------------------------------
    def _vis_depth_contact_debug(
        self,
        center: int,
        start: int,
        end: int,
        global_orient_w: torch.Tensor,   # (F_total, 3) aa
        body_pose_aa: torch.Tensor,      # (win_len, 63) 窗口 body_pose
        betas: torch.Tensor,             # (F_total, 10)
        win_transl: torch.Tensor,        # (win_len, 3) 窗口 transl
        static_conf_logits: Optional[torch.Tensor],  # (F_total, 6)
        depth_map: torch.Tensor,         # (H, W) 中心帧深度图
        R_cw: torch.Tensor,              # (F_total, 3, 3) w2c rotation
        t_cw: torch.Tensor,              # (F_total, 3) w2c translation
        K: torch.Tensor,                 # (3, 3) intrinsics
        image_path: str,                 # 中心帧图片路径
        output_dir: str,
    ):
        """检测窗口内穿透场景的静态接触关节，生成 Contact IK 信号并可视化。

        不依赖 R_cw/t_cw 的绝对精度——仅用于投影到像素。
        通过窗口内所有关节的 z_scene/z_smpl 中位数做局部 scale 对齐，
        然后在对齐后的深度空间检测脚部穿透。

        流程：
        1. FK → 世界关节 → 中心帧相机坐标 → 像素 (u,v) + z_smpl
        2. 用中心帧所有 22 个关节计算 scale_factor = median(z_scene / z_smpl)
        3. z_smpl_aligned = z_smpl * scale_factor
        4. 对静态脚部关节：若 z_smpl_aligned > z_scene + eps → 穿透
        5. 可视化 + 返回 contact_info_list 作为 IK 信号

        颜色编码：
        - 蓝色小点：用于 scale 对齐的中心帧所有关节
        - 绿色圆圈：静态脚部、对齐后未穿透
        - 红色圆圈 + 箭头：对齐后仍穿透，需要 Contact IK
        """
        import cv2

        vis_dir = os.path.join(output_dir, 'oar_depth_contact_debug')
        os.makedirs(vis_dir, exist_ok=True)

        device = self.device
        win_len = end - start
        center_local = center - start

        # static_conf 列索引 → FK-22 关节 ID
        CONTACT_JOINTS = {
            0: (7,  'L_Ankle'),
            1: (10, 'L_Foot'),
            2: (8,  'R_Ankle'),
            3: (11, 'R_Foot'),
        }
        PENETRATION_EPS = 0.03  # 3cm 穿透阈值（对齐后用更小阈值）

        # 构造 static_conf (win_len, 6)
        if static_conf_logits is not None:
            win_static = static_conf_logits[start:end].to(device)
            static_conf = torch.sigmoid(win_static)  # (win_len, 6)
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        # FK 计算窗口内关节的世界坐标系位置
        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        win_bp = body_pose_aa.to(device)
        win_tr = win_transl.to(device)

        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=win_bp.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]  # (win_len, 22, 3)

        R_center = R_cw[center].to(device)  # (3, 3)
        t_center = t_cw[center].to(device)  # (3,)
        K_dev = K.to(device)
        depth_dev = depth_map.to(device)
        dH, dW = depth_dev.shape

        img = cv2.imread(image_path)
        if img is None:
            self.logger.warning(f"Cannot read image: {image_path}")
            return
        H, W = img.shape[:2]
        canvas = img.copy()

        # ================================================================
        # Phase 1: 用中心帧所有 22 个关节计算局部 scale_factor
        # ================================================================
        z_smpl_all = []
        z_scene_all = []
        center_joint_pixels = []  # 用于可视化 scale 对齐的采样点

        j_center_world = joints_world[center_local]  # (22, 3)
        j_center_cam = (R_center @ j_center_world.T).T + t_center  # (22, 3)
        pj_center = (K_dev @ j_center_cam.T).T  # (22, 3)
        pj_center_2d = pj_center[:, :2] / (pj_center[:, 2:3] + 1e-6)  # (22, 2)

        for jid in range(22):
            z_s = j_center_cam[jid, 2].item()
            if z_s <= 0.01:
                continue
            u_px = int(round(pj_center_2d[jid, 0].item()))
            v_px = int(round(pj_center_2d[jid, 1].item()))
            if u_px < 0 or u_px >= dW or v_px < 0 or v_px >= dH:
                continue
            z_d = depth_dev[v_px, u_px].item()
            if z_d < 0.1 or z_d > 100.0:
                continue
            z_smpl_all.append(z_s)
            z_scene_all.append(z_d)
            center_joint_pixels.append((u_px, v_px, jid, z_s, z_d))

        if len(z_smpl_all) < 3:
            self.logger.warning(f"Window {center}: too few valid joints for scale alignment ({len(z_smpl_all)})")
            return

        z_smpl_arr = np.array(z_smpl_all)
        z_scene_arr = np.array(z_scene_all)
        ratios = z_scene_arr / (z_smpl_arr + 1e-8)
        scale_factor = float(np.median(ratios))

        self.logger.debug(
            f"[DepthContact] Window {center}: scale_factor={scale_factor:.4f} "
            f"(median of {len(ratios)} joints, range=[{ratios.min():.3f}, {ratios.max():.3f}])"
        )

        # 在画布上画 scale 对齐采样点（蓝色小点）
        for u_px, v_px, jid, zs, zd in center_joint_pixels:
            if 0 <= u_px < W and 0 <= v_px < H:
                cv2.circle(canvas, (u_px, v_px), 2, (255, 150, 0), -1)

        # ================================================================
        # Phase 2: 对齐后检测脚部静态关节穿透
        # ================================================================
        n_static = 0
        n_penetrate = 0
        contact_info_list = []

        for fi in range(win_len):
            frame_global = start + fi
            for conf_idx, (jid, jname) in CONTACT_JOINTS.items():
                conf_val = static_conf[fi, conf_idx].item()
                if conf_val < 0.5:
                    continue

                n_static += 1

                j_world = joints_world[fi, jid]
                j_cam = R_center @ j_world + t_center
                z_smpl_raw = j_cam[2].item()
                z_smpl_aligned = z_smpl_raw * scale_factor

                pj = K_dev @ j_cam
                u = pj[0].item() / (pj[2].item() + 1e-6)
                v = pj[1].item() / (pj[2].item() + 1e-6)
                ui, vi = int(round(u)), int(round(v))

                if ui < 0 or ui >= dW or vi < 0 or vi >= dH:
                    continue
                if ui < 0 or ui >= W or vi < 0 or vi >= H:
                    continue

                z_scene = depth_dev[vi, ui].item()
                if z_scene < 0.1 or z_scene > 100.0:
                    continue

                penetration = z_smpl_aligned - z_scene - PENETRATION_EPS
                is_penetrating = penetration > 0

                if is_penetrating:
                    n_penetrate += 1

                contact_info_list.append({
                    'frame': frame_global,
                    'fi': fi,
                    'jid': jid,
                    'jname': jname,
                    'conf': conf_val,
                    'u': ui, 'v': vi,
                    'z_smpl_raw': z_smpl_raw,
                    'z_smpl_aligned': z_smpl_aligned,
                    'z_scene': z_scene,
                    'penetration': penetration if is_penetrating else 0.0,
                    'is_penetrating': is_penetrating,
                })

        # ---- 绘制接触点 ----
        for info in contact_info_list:
            u, v = info['u'], info['v']
            if info['is_penetrating']:
                radius = max(5, min(14, int(info['penetration'] * 200)))
                cv2.circle(canvas, (u, v), radius, (0, 0, 255), 2)
                arrow_len = max(12, min(50, int(info['penetration'] * 400)))
                cv2.arrowedLine(canvas, (u, v), (u, v - arrow_len),
                                (0, 0, 255), 2, tipLength=0.3)
                label = f"{info['jname']}@f{info['frame']} pen={info['penetration']:.3f}m"
                cv2.putText(canvas, label, (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)
            else:
                cv2.circle(canvas, (u, v), 3, (0, 200, 0), -1)
                label = f"{info['jname']}@f{info['frame']}"
                cv2.putText(canvas, label, (u + 3, v - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 200, 0), 1)

        # ---- 图例和统计 ----
        cv2.putText(canvas,
                    f"Window center={center} | scale={scale_factor:.3f} | "
                    f"Static: {n_static} | Penetrating: {n_penetrate}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas,
                    "Blue=scale_ref  Green=static(OK)  Red+arrow=penetrating(contact IK)",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        if contact_info_list:
            pen_items = [c for c in contact_info_list if c['is_penetrating']]
            if pen_items:
                y_offset = 50
                cv2.putText(canvas, f"Penetration Summary (scale={scale_factor:.3f}):",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1)
                y_offset += 18
                for item in pen_items[:15]:
                    txt = (f"  f{item['frame']} {item['jname']}: "
                           f"z_raw={item['z_smpl_raw']:.2f} z_align={item['z_smpl_aligned']:.2f} "
                           f"z_scene={item['z_scene']:.2f} pen={item['penetration']:.3f}m")
                    cv2.putText(canvas, txt, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 255), 1)
                    y_offset += 14
                if len(pen_items) > 15:
                    cv2.putText(canvas, f"  ... and {len(pen_items) - 15} more",
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 150, 255), 1)

        out_path = os.path.join(vis_dir, f'{center:05d}.jpg')
        cv2.imwrite(out_path, canvas)

        if n_penetrate > 0:
            self.logger.debug(
                f"[DepthContact] Window {center}: scale={scale_factor:.3f}, "
                f"{n_static} static, {n_penetrate} penetrating"
            )

        return contact_info_list

    def _get_human_mask_for_depth(
        self,
        masks: Optional[torch.Tensor],
        frame_idx: int,
        depth_shape: tuple,
    ) -> Optional[np.ndarray]:
        """取中心帧人体 mask，并 resize 到 depth map 尺寸。"""
        if masks is None:
            return None
        try:
            mask = masks[frame_idx]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.asarray(mask)
            mask_np = np.squeeze(mask_np)
            if mask_np.ndim != 2:
                return None
            target_h, target_w = int(depth_shape[0]), int(depth_shape[1])
            if mask_np.shape != (target_h, target_w):
                mask_img = Image.fromarray((mask_np > 0.5).astype(np.uint8) * 255)
                mask_img = mask_img.resize((target_w, target_h), Image.Resampling.NEAREST)
                mask_np = np.asarray(mask_img) > 0
            else:
                mask_np = mask_np > 0.5
            return mask_np
        except Exception as e:
            self.logger.debug(f"Cannot prepare human mask for depth: {e}")
            return None

    def _align_depth_to_current_human(
        self,
        depth_map: torch.Tensor,
        K: torch.Tensor,
        center: int,
        center_local: int,
        global_orient_w: torch.Tensor,
        win_body_pose: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
    ) -> tuple:
        """Scale Metric3D depth to the current SMPL camera-depth scale using projected FK joints."""
        mode = str(self.config.get('contact_ik_depth_align_mode', 'none')).lower()
        if mode in ('none', 'false', 'off'):
            return depth_map, {'enabled': False, 'mode': mode}
        if mode not in ('joint_scale', 'joints', 'scale_to_smpl'):
            self.logger.warning(f"Unknown contact_ik_depth_align_mode={mode}, skip depth alignment")
            return depth_map, {'enabled': False, 'mode': mode, 'reason': 'unknown_mode'}

        device = self.device
        depth_np = depth_map.detach().cpu().numpy() if isinstance(depth_map, torch.Tensor) else np.asarray(depth_map)
        H, W = depth_np.shape[:2]
        K_dev = K.to(device) if isinstance(K, torch.Tensor) else torch.from_numpy(np.asarray(K, dtype=np.float32)).to(device)
        min_joints = int(self.config.get('contact_ik_depth_align_min_joints', 3))

        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=win_body_pose[center_local:center_local + 1].unsqueeze(0).to(device),
                betas=betas[center:center + 1].unsqueeze(0).to(device),
                global_orient=global_orient_w[center:center + 1].unsqueeze(0).to(device),
                transl=win_transl[center_local:center_local + 1].unsqueeze(0).to(device),
            )[0, 0]
            R_center = R_cw[center].to(device)
            t_center = t_cw[center].to(device)
            joints_cam = torch.einsum('ij,kj->ki', R_center, joints_world) + t_center
            proj = torch.einsum('ij,kj->ki', K_dev, joints_cam)
            uv = (proj[:, :2] / (proj[:, 2:3] + 1e-6)).detach().cpu().numpy()
            z_smpl = joints_cam[:, 2].detach().cpu().numpy()

        z_scene = []
        z_body = []
        for (u, v), z in zip(uv, z_smpl):
            ui, vi = int(round(float(u))), int(round(float(v)))
            if ui < 0 or ui >= W or vi < 0 or vi >= H or not np.isfinite(z) or z <= 0.1:
                continue
            ds = float(depth_np[vi, ui])
            if np.isfinite(ds) and 0.1 < ds < 100.0:
                z_scene.append(ds)
                z_body.append(float(z))
        if len(z_scene) < min_joints:
            return depth_map, {'enabled': False, 'mode': mode, 'reason': 'too_few_joints', 'num_joints': len(z_scene)}

        z_scene = np.asarray(z_scene, dtype=np.float64)
        z_body = np.asarray(z_body, dtype=np.float64)
        ratios = z_scene / np.maximum(z_body, 1e-8)
        scale_scene_to_smpl = float(1.0 / max(float(np.median(ratios)), 1e-8))
        aligned = (depth_np * scale_scene_to_smpl).astype(np.float32)
        aligned[~np.isfinite(aligned)] = 0.0
        stats = {
            'enabled': True,
            'mode': mode,
            'num_joints': int(len(ratios)),
            'smpl_to_scene_scale': float(np.median(ratios)),
            'scene_to_smpl_scale': scale_scene_to_smpl,
            'ratio_min': float(ratios.min()),
            'ratio_max': float(ratios.max()),
        }
        return torch.from_numpy(aligned), stats

    def _fit_ground_plane_from_depth(
        self,
        depth_map: torch.Tensor,
        K: torch.Tensor,
        human_mask: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """从扣除人体后的 Metric3D 点云中 RANSAC 拟合中心帧地面平面。"""
        depth_np = depth_map.detach().cpu().numpy() if isinstance(depth_map, torch.Tensor) else np.asarray(depth_map)
        K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        H, W = depth_np.shape[:2]
        valid = np.isfinite(depth_np) & (depth_np > 0.1) & (depth_np < 100.0)
        valid[:int(H * float(self.config.get('ground_plane_lower_frac', 0.45))), :] = False
        if human_mask is not None and bool(self.config.get('contact_ik_ground_use_human_mask', True)):
            valid &= ~human_mask.astype(bool)

        stride = max(1, int(self.config.get('ground_plane_sample_stride', 6)))
        ys, xs = np.nonzero(valid[::stride, ::stride])
        if len(xs) < 50:
            return None
        xs = xs.astype(np.float32) * stride
        ys = ys.astype(np.float32) * stride
        z = depth_np[ys.astype(np.int64), xs.astype(np.int64)].astype(np.float32)

        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        pts = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 50:
            return None

        max_points = int(self.config.get('ground_plane_max_points', 6000))
        rng = np.random.default_rng(20260621)
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), size=max_points, replace=False)]

        thresh = float(self.config.get('ground_plane_inlier_thresh', 0.05))
        iters = int(self.config.get('ground_plane_ransac_iters', 96))
        best_inliers = None
        best_count = 0
        for _ in range(iters):
            ids = rng.choice(len(pts), size=3, replace=False)
            p0, p1, p2 = pts[ids]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            d = -float(np.dot(n, p0))
            residual = np.abs(pts @ n + d)
            inliers = residual < thresh
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers

        min_inliers = int(self.config.get('ground_plane_min_inliers', 200))
        if best_inliers is None or best_count < min_inliers:
            return None

        inlier_pts = pts[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        n = vh[-1].astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-8)
        d = -float(np.dot(n, centroid))
        # OpenCV camera coordinates: y points down. Use negative-y as the upward normal.
        if n[1] > 0:
            n = -n
            d = -d
        residual = np.abs(inlier_pts @ n + d)
        return {
            'normal': n.astype(np.float32),
            'offset': float(d),
            'inliers': int(best_count),
            'candidates': int(len(pts)),
            'mean_residual': float(residual.mean()),
            'inlier_ratio': float(best_count) / float(max(1, len(pts))),
        }

    def _fit_local_plane_patch(
        self,
        depth_map: torch.Tensor,
        K: torch.Tensor,
        u: float,
        v: float,
        human_mask: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Fit a small support plane around a projected contact point.

        This is a JOSH-style local scene association: use image-nearby static
        scene samples (after removing human mask) rather than one global floor.
        """
        depth_np = depth_map.detach().cpu().numpy() if isinstance(depth_map, torch.Tensor) else np.asarray(depth_map)
        K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        H, W = depth_np.shape[:2]
        radius = int(self.config.get('local_plane_radius', 90))
        stride = max(1, int(self.config.get('local_plane_sample_stride', 3)))
        min_points = int(self.config.get('local_plane_min_points', 40))
        u0, v0 = int(round(float(u))), int(round(float(v)))
        x0, x1 = max(0, u0 - radius), min(W, u0 + radius + 1)
        y0, y1 = max(0, v0 - radius), min(H, v0 + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        patch = depth_np[y0:y1:stride, x0:x1:stride]
        yy, xx = np.mgrid[y0:y1:stride, x0:x1:stride]
        valid = np.isfinite(patch) & (patch > 0.1) & (patch < 100.0)
        # Prefer background samples; if the body mask removes too much, still use the visible annulus.
        if human_mask is not None and bool(self.config.get('contact_ik_ground_use_human_mask', True)):
            valid &= ~human_mask[y0:y1:stride, x0:x1:stride].astype(bool)
        rr = (xx.astype(np.float32) - float(u)) ** 2 + (yy.astype(np.float32) - float(v)) ** 2
        valid &= rr <= float(radius * radius)
        if int(valid.sum()) < min_points:
            return None

        xs = xx[valid].astype(np.float32)
        ys = yy[valid].astype(np.float32)
        z = patch[valid].astype(np.float32)
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        pts = np.stack([(xs - cx) / fx * z, (ys - cy) / fy * z, z], axis=1)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < min_points:
            return None

        rng = np.random.default_rng(20260621 + u0 * 13 + v0)
        thresh = float(self.config.get('local_plane_inlier_thresh', 0.04))
        iters = int(self.config.get('local_plane_ransac_iters', 64))
        best_inliers = None
        best_count = 0
        for _ in range(iters):
            ids = rng.choice(len(pts), size=3, replace=False)
            p0, p1, p2 = pts[ids]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            d = -float(np.dot(n, p0))
            residual = np.abs(pts @ n + d)
            inliers = residual < thresh
            count = int(inliers.sum())
            if count > best_count:
                best_count = count
                best_inliers = inliers
        if best_inliers is None or best_count < min_points:
            return None

        inlier_pts = pts[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        n = vh[-1].astype(np.float32)
        n = n / (np.linalg.norm(n) + 1e-8)
        d = -float(np.dot(n, centroid))
        if n[1] > 0:
            n = -n
            d = -d
        residual = np.abs(inlier_pts @ n + d)
        return {
            'normal': n.astype(np.float32),
            'offset': float(d),
            'inliers': int(best_count),
            'candidates': int(len(pts)),
            'mean_residual': float(residual.mean()),
            'inlier_ratio': float(best_count) / float(max(1, len(pts))),
        }

    def _prepare_local_plane_ik_window(
        self,
        global_orient_w: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        static_conf_logits: Optional[torch.Tensor],
        depth_map: torch.Tensor,
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        K: torch.Tensor,
        center: int,
        start: int,
        end: int,
        human_mask: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """构建单个窗口 Local-Plane Contact IK 的优化前置数据（不做优化）。

        该函数封装了配置解析、点定义、局部平面拟合与 JOSH anchor 分段等所有前处理，
        供单窗口 (`_apply_local_plane_contact_ik`) 与批量 (`_apply_local_plane_contact_ik_batch`)
        两条路径共享，确保二者的 contact 生成逻辑完全一致。

        Returns:
            prep dict；若该窗口无有效 contact / 无可优化关节，返回 None（调用方保持 pose 不变）。
        """
        device = self.device
        win_len = end - start
        opt_steps = int(self.config.get('contact_ik_steps', 30))
        lr = float(self.config.get('contact_ik_lr', 0.003))
        loss_contact_w = float(self.config.get('loss_contact_w', 5.0))
        loss_sigma = float(self.config.get('loss_contact_sigma', 0.05))
        loss_bp_reg_w = float(self.config.get('loss_contact_bp_reg_w', 0.1))
        loss_smooth_w = float(self.config.get('loss_contact_smooth_w', 0.5))
        conf_thr = float(self.config.get('contact_ik_conf_thr', 0.5))
        ground_loss = str(self.config.get('contact_ik_ground_loss', 'l2'))
        scene_model = str(self.config.get('contact_ik_scene_model', 'local_planes'))
        point_type = str(self.config.get('contact_ik_point_type', 'joints')).lower()
        use_vertices = point_type in ('vertices', 'mesh_vertices', 'toeheel_vertices', 'toe_heel_vertices', 'sole_vertices')
        anchor_w = float(self.config.get('josh_anchor_w', 0.0))
        if scene_model == 'josh_planes' and anchor_w <= 0.0:
            anchor_w = 0.5
        anchor_sigma = float(self.config.get('josh_anchor_sigma', 0.08))
        min_inlier_ratio = float(self.config.get('local_plane_min_inlier_ratio', 0.0))
        max_residual = float(self.config.get('local_plane_max_residual', 1e9))
        weighted_conf = bool(self.config.get('contact_ik_weighted_conf', False))
        vel_sigma = max(1e-6, float(self.config.get('contact_ik_vel_sigma', 0.04)))
        height_sigma = max(1e-6, float(self.config.get('contact_ik_height_sigma', 0.06)))
        min_weight = float(self.config.get('contact_ik_min_weight', 0.05))
        optimize_transl = bool(self.config.get('contact_ik_optimize_transl', False))
        transl_reg_w = float(self.config.get('loss_contact_transl_reg_w', 2.0))
        transl_smooth_w = float(self.config.get('loss_contact_transl_smooth_w', 5.0))
        transl_dims = [int(x) for x in self.config.get('contact_ik_transl_dims', [0, 1, 2]) if 0 <= int(x) < 3]

        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits[start:end].to(device))
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        bp_orig = body_pose_aa.to(device).clone().detach()
        win_tr = win_transl.to(device)
        R_center = R_cw[center].to(device)
        t_center = t_cw[center].to(device)
        K_dev = K.to(device)
        use_rich_floor = scene_model in ('rich_floor', 'gt_floor', 'rich_gt_floor')
        rich_floor_plane = None
        if use_rich_floor:
            # RICH/GVHMR world coordinate is ay; the evaluation floor is y=0.
            # Transform n_w=[0,1,0], d_w=0 to the center camera coordinates:
            # x_c = R_cw x_w + t_cw, so n_c = R_cw n_w and d_c = -n_c^T t_cw.
            n_w = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
            n_c = R_center @ n_w
            n_c = n_c / n_c.norm().clamp_min(1e-8)
            d_c = -torch.dot(n_c, t_center)
            n_np = n_c.detach().cpu().numpy().astype(np.float32)
            d_np = float(d_c.detach().cpu())
            if n_np[1] > 0:
                n_np = -n_np
                d_np = -d_np
            rich_floor_plane = {
                'normal': n_np,
                'offset': d_np,
                'inliers': 0,
                'candidates': 0,
                'mean_residual': 0.0,
                'inlier_ratio': 1.0,
            }

        point_defs = []
        selected_vertex_ids = []
        smplx_model = None
        smplx2smpl_sel = None
        with torch.no_grad():
            if use_vertices:
                if point_type == 'sole_vertices':
                    vertex_defs = self._build_sole_vertex_defs(
                        int(self.config.get('contact_ik_sole_max_vertices', 32))
                    )
                else:
                    vertex_defs = self._normalize_contact_vertex_defs(
                        self.config.get('contact_ik_vertex_ids', [3216, 3387, 6617, 6787]),
                        self.config.get('contact_ik_vertex_conf_indices', [1, 0, 3, 2]),
                        self.config.get('contact_ik_vertex_names', ['LToe', 'LHeel', 'RToe', 'RHeel']),
                        self.config.get('contact_ik_vertex_offsets', [0.0, 0.0, 0.0, 0.0]),
                    )
                if not vertex_defs:
                    return None
                selected_vertex_ids = [d['vid'] for d in vertex_defs]
                smplx_model, smplx2smpl = self._get_contact_smplx_assets()
                row_ids = torch.tensor(selected_vertex_ids, dtype=torch.long, device=smplx2smpl.device)
                smplx2smpl_sel = smplx2smpl.index_select(0, row_ids).coalesce() if smplx2smpl.is_sparse else smplx2smpl.index_select(0, row_ids)
                points_world_init = self._forward_selected_smpl_vertices(
                    smplx_model, smplx2smpl_sel, win_orient, bp_orig, win_betas, win_tr
                )
                point_defs = [
                    {'pid': int(d['vid']), 'pidx': i, 'name': d['name'], 'conf_idx': int(d['conf_idx']), 'offset': float(d['offset'])}
                    for i, d in enumerate(vertex_defs)
                ]
            else:
                points_world_init = self.endecoder.fk_v2(
                    body_pose=bp_orig.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=win_tr.unsqueeze(0),
                )[0]
                contact_joint_ids = [int(x) for x in self.config.get('contact_ik_ground_joint_ids', [10, 11])]
                conf_idx_by_jid = {7: 0, 10: 1, 8: 2, 11: 3}
                offsets_cfg = {int(k): float(v) for k, v in self.config.get('contact_ik_ground_joint_offsets', {}).items()}
                point_defs = [
                    {'pid': jid, 'pidx': jid, 'name': f'J{jid}', 'conf_idx': conf_idx_by_jid.get(jid, -1), 'offset': offsets_cfg.get(jid, 0.0)}
                    for jid in contact_joint_ids
                ]

        with torch.no_grad():
            point_vel = torch.zeros(points_world_init.shape[:2], device=device)
            if win_len > 1:
                disp = (points_world_init[1:] - points_world_init[:-1]).norm(dim=-1)
                point_vel[1:] = disp
                point_vel[:-1] = torch.maximum(point_vel[:-1], disp)

        raw_contact_items = []
        rejected_by_quality = 0
        for fi in range(win_len):
            for pdef in point_defs:
                conf_idx = int(pdef['conf_idx'])
                if conf_idx < 0 or conf_idx >= static_conf.shape[1]:
                    continue
                conf = float(static_conf[fi, conf_idx].item())
                if conf < conf_thr:
                    continue
                p_cam = R_center @ points_world_init[fi, int(pdef['pidx'])] + t_center
                if use_rich_floor:
                    plane = rich_floor_plane
                else:
                    proj = K_dev @ p_cam
                    u = float((proj[0] / (proj[2] + 1e-6)).detach().cpu())
                    v = float((proj[1] / (proj[2] + 1e-6)).detach().cpu())
                    plane = self._fit_local_plane_patch(depth_map, K, u, v, human_mask)
                    if plane is None:
                        continue
                    if plane['inlier_ratio'] < min_inlier_ratio or plane['mean_residual'] > max_residual:
                        rejected_by_quality += 1
                        continue
                n_np = plane['normal']
                d_np = float(plane['offset'])
                p_np = p_cam.detach().cpu().numpy().astype(np.float32)
                target_offset = float(pdef['offset'])
                signed = float(np.dot(n_np, p_np) + d_np)
                anchor_np = p_np - signed * n_np
                if weighted_conf:
                    vel = float(point_vel[fi, int(pdef['pidx'])].detach().cpu())
                    vel_conf = float(np.exp(-((vel / vel_sigma) ** 2)))
                    height_conf = float(np.exp(-(((signed - target_offset) / height_sigma) ** 2)))
                    ratio_conf = float(np.clip((plane['inlier_ratio'] - min_inlier_ratio) / max(1e-6, 1.0 - min_inlier_ratio), 0.0, 1.0))
                    residual_conf = float(np.exp(-((plane['mean_residual'] / max(max_residual, 1e-6)) ** 2))) if max_residual < 1e8 else 1.0
                    conf = conf * vel_conf * height_conf * ratio_conf * residual_conf
                    if conf < min_weight:
                        continue
                raw_contact_items.append((fi, int(pdef['pid']), int(pdef['pidx']), pdef['name'], conf, plane, target_offset, anchor_np))

        # JOSH-like persistent contact anchor: same static foot point in consecutive
        # frames should stay fixed on the local scene patch tangent plane.
        contact_items = []
        if anchor_w > 0.0:
            for pid in sorted({item[1] for item in raw_contact_items}):
                items = sorted([item for item in raw_contact_items if item[1] == pid], key=lambda x: x[0])
                seg = []
                last_fi = None

                def _flush_segment(segment):
                    if not segment:
                        return
                    anchors = np.stack([x[7] for x in segment], axis=0)
                    seg_anchor = np.median(anchors, axis=0).astype(np.float32)
                    for fi, spid, pidx, name, conf, plane, target_offset, _ in segment:
                        contact_items.append((fi, spid, pidx, name, conf, plane, target_offset, seg_anchor))
                for item in items:
                    if last_fi is not None and item[0] > last_fi + 1:
                        _flush_segment(seg)
                        seg = []
                    seg.append(item)
                    last_fi = item[0]
                _flush_segment(seg)
        else:
            contact_items = [
                (fi, pid, pidx, name, conf, plane, target_offset, anchor_np)
                for fi, pid, pidx, name, conf, plane, target_offset, anchor_np in raw_contact_items
            ]

        if not contact_items:
            self.logger.debug(
                f"[LocalPlaneContactIK] Window {center}: no local plane contacts, "
                f"point_type={point_type}, rejected_by_quality={rejected_by_quality}, skipping"
            )
            return None

        opt_joints = [int(j) for j in self.config.get('contact_ik_opt_joints', [7, 8, 10, 11])]
        leg_bp_indices = []
        for j in opt_joints:
            if 1 <= j <= 21:
                leg_bp_indices.extend([(j - 1) * 3, (j - 1) * 3 + 1, (j - 1) * 3 + 2])
        if not leg_bp_indices:
            return None

        transl_mask = torch.zeros(3, device=device)
        if transl_dims:
            transl_mask[transl_dims] = 1.0
        plane_tensors = [
            (fi, pid, pidx, name, conf,
             torch.from_numpy(plane['normal']).to(device=device, dtype=torch.float32),
             torch.tensor(float(plane['offset']), device=device),
             float(target_offset),
             torch.from_numpy(anchor_np).to(device=device, dtype=torch.float32))
            for fi, pid, pidx, name, conf, plane, target_offset, anchor_np in contact_items
        ]

        return {
            'win_len': win_len,
            'center': center,
            'bp_orig': bp_orig,
            'win_orient': win_orient,
            'win_betas': win_betas,
            'win_tr': win_tr,
            'R_center': R_center,
            't_center': t_center,
            'use_vertices': use_vertices,
            'smplx_model': smplx_model,
            'smplx2smpl_sel': smplx2smpl_sel,
            'leg_bp_indices': leg_bp_indices,
            'plane_tensors': plane_tensors,
            'contact_items': contact_items,
            'contact_count': len(contact_items),
            'rejected_by_quality': rejected_by_quality,
            'point_type': point_type,
            'transl_mask': transl_mask,
            # scalar hyper-params (shared across windows, carried for convenience)
            'opt_steps': opt_steps,
            'lr': lr,
            'loss_contact_w': loss_contact_w,
            'loss_sigma': loss_sigma,
            'loss_bp_reg_w': loss_bp_reg_w,
            'loss_smooth_w': loss_smooth_w,
            'ground_loss': ground_loss,
            'anchor_w': anchor_w,
            'anchor_sigma': anchor_sigma,
            'optimize_transl': optimize_transl,
            'transl_reg_w': transl_reg_w,
            'transl_smooth_w': transl_smooth_w,
            'weighted_conf': weighted_conf,
        }

    def _apply_local_plane_contact_ik(
        self,
        global_orient_w: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        static_conf_logits: Optional[torch.Tensor],
        depth_map: torch.Tensor,
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        K: torch.Tensor,
        center: int,
        start: int,
        end: int,
        human_mask: Optional[np.ndarray],
    ) -> torch.Tensor:
        device = self.device
        prep = self._prepare_local_plane_ik_window(
            global_orient_w, body_pose_aa, betas, win_transl, static_conf_logits,
            depth_map, R_cw, t_cw, K, center, start, end, human_mask,
        )
        if prep is None:
            return body_pose_aa

        win_len = prep['win_len']
        opt_steps = prep['opt_steps']
        lr = prep['lr']
        loss_contact_w = prep['loss_contact_w']
        loss_sigma = prep['loss_sigma']
        loss_bp_reg_w = prep['loss_bp_reg_w']
        loss_smooth_w = prep['loss_smooth_w']
        ground_loss = prep['ground_loss']
        anchor_w = prep['anchor_w']
        anchor_sigma = prep['anchor_sigma']
        optimize_transl = prep['optimize_transl']
        transl_reg_w = prep['transl_reg_w']
        transl_smooth_w = prep['transl_smooth_w']
        weighted_conf = prep['weighted_conf']
        point_type = prep['point_type']
        bp_orig = prep['bp_orig']
        win_orient = prep['win_orient']
        win_betas = prep['win_betas']
        win_tr = prep['win_tr']
        R_center = prep['R_center']
        t_center = prep['t_center']
        use_vertices = prep['use_vertices']
        smplx_model = prep['smplx_model']
        smplx2smpl_sel = prep['smplx2smpl_sel']
        leg_bp_indices = prep['leg_bp_indices']
        plane_tensors = prep['plane_tensors']
        contact_items = prep['contact_items']
        rejected_by_quality = prep['rejected_by_quality']
        transl_mask = prep['transl_mask']

        delta_bp = torch.zeros(win_len, len(leg_bp_indices), device=device, requires_grad=True)
        delta_tr = torch.zeros(win_len, 3, device=device, requires_grad=optimize_transl)
        params = [delta_bp] + ([delta_tr] if optimize_transl else [])
        optimizer = torch.optim.Adam(params, lr=lr)

        final_loss_contact = torch.tensor(0.0, device=device)
        final_loss_anchor = torch.tensor(0.0, device=device)
        for _ in range(opt_steps):
            optimizer.zero_grad()
            bp_mod = bp_orig.clone()
            bp_mod[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp
            tr_mod = win_tr + (delta_tr * transl_mask[None, :] if optimize_transl else 0.0)
            if use_vertices:
                points_world_opt = self._forward_selected_smpl_vertices(
                    smplx_model, smplx2smpl_sel, win_orient, bp_mod, win_betas, tr_mod
                )
            else:
                points_world_opt = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=tr_mod.unsqueeze(0),
                )[0]
            loss_contact = torch.tensor(0.0, device=device)
            loss_anchor = torch.tensor(0.0, device=device)
            weight_sum = 0.0
            for fi, pid, pidx, name, conf, normal, offset, target_offset, anchor in plane_tensors:
                p_cam = R_center @ points_world_opt[fi, pidx] + t_center
                signed_dist = torch.dot(normal, p_cam) + offset
                diff = signed_dist - target_offset
                if ground_loss == 'penetration':
                    diff = F.relu(-diff)
                loss_contact = loss_contact + float(conf) * gmof(diff, sigma=loss_sigma)
                if anchor_w > 0.0:
                    tangent = p_cam - anchor
                    tangent = tangent - torch.dot(tangent, normal) * normal
                    loss_anchor = loss_anchor + float(conf) * gmof(tangent, sigma=anchor_sigma).sum()
                weight_sum += float(conf)
            loss_contact = loss_contact / max(weight_sum, 1e-6)
            loss_anchor = loss_anchor / max(weight_sum, 1e-6)
            final_loss_contact = loss_contact.detach()
            final_loss_anchor = loss_anchor.detach()
            loss_reg = delta_bp.pow(2).mean()
            loss_smooth = torch.tensor(0.0, device=device)
            if win_len >= 2:
                loss_smooth = (delta_bp[1:] - delta_bp[:-1]).pow(2).mean()
            loss_tr_reg = torch.tensor(0.0, device=device)
            loss_tr_smooth = torch.tensor(0.0, device=device)
            if optimize_transl:
                delta_tr_eff = delta_tr * transl_mask[None, :]
                loss_tr_reg = delta_tr_eff.pow(2).mean()
                if win_len >= 2:
                    loss_tr_smooth = (delta_tr_eff[1:] - delta_tr_eff[:-1]).pow(2).mean()
            total_loss = (
                loss_contact_w * loss_contact
                + anchor_w * loss_anchor
                + loss_bp_reg_w * loss_reg
                + loss_smooth_w * loss_smooth
                + transl_reg_w * loss_tr_reg
                + transl_smooth_w * loss_tr_smooth
            )
            total_loss.backward()
            optimizer.step()

        refined_bp = bp_orig.clone()
        refined_bp[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp.detach()
        refined_tr = win_tr + (delta_tr.detach() * transl_mask[None, :] if optimize_transl else 0.0)
        self.logger.debug(
            f"[LocalPlaneContactIK] Window {center}: point_type={point_type}, contacts={len(contact_items)}, "
            f"rejected_by_quality={rejected_by_quality}, loss_contact={float(final_loss_contact):.6f}, "
            f"loss_anchor={float(final_loss_anchor):.6f}, anchor_w={anchor_w:.3f}, "
            f"weighted_conf={weighted_conf}, optimize_transl={optimize_transl}, "
            f"delta_bp_mean={float(delta_bp.detach().abs().mean()):.6f}, "
            f"delta_tr_mean={float((delta_tr.detach() * transl_mask[None, :]).norm(dim=-1).mean()):.6f}"
        )
        if optimize_transl:
            return refined_bp.cpu(), refined_tr.cpu()
        return refined_bp.cpu()

    def _apply_local_plane_contact_ik_batch(self, preps: List[Dict[str, Any]]):
        """对一组窗口的 Local-Plane Contact IK 做批量 Adam 优化。

        输入 preps 必须来自 `_prepare_local_plane_ik_window`，且 win_len 相同。因为各窗口
        的优化参数(delta_bp / delta_tr)互不相交、损失可分离，批量优化在数学上等价于逐窗口
        独立优化（每个参数元素的 Adam 动量独立），但把 FK 与优化步在 batch 维上并行，
        显著提升 GPU 利用率。

        Returns:
            list，与 preps 一一对应；每项为 refined_bp (win_len,63) CPU tensor，
            或在 optimize_transl 时为 (refined_bp, refined_tr) 元组。
        """
        device = self.device
        B = len(preps)
        p0 = preps[0]
        win_len = p0['win_len']
        leg_bp_indices = p0['leg_bp_indices']
        nleg = len(leg_bp_indices)
        use_vertices = p0['use_vertices']
        smplx_model = p0['smplx_model']
        smplx2smpl_sel = p0['smplx2smpl_sel']
        opt_steps = p0['opt_steps']
        lr = p0['lr']
        loss_contact_w = p0['loss_contact_w']
        loss_sigma = p0['loss_sigma']
        loss_bp_reg_w = p0['loss_bp_reg_w']
        loss_smooth_w = p0['loss_smooth_w']
        ground_loss = p0['ground_loss']
        anchor_w = p0['anchor_w']
        anchor_sigma = p0['anchor_sigma']
        optimize_transl = p0['optimize_transl']
        transl_reg_w = p0['transl_reg_w']
        transl_smooth_w = p0['transl_smooth_w']
        transl_mask = p0['transl_mask']

        bp_orig_b = torch.stack([p['bp_orig'] for p in preps], dim=0)        # (B, win_len, 63)
        win_orient_b = torch.stack([p['win_orient'] for p in preps], dim=0)  # (B, win_len, 3)
        win_betas_b = torch.stack([p['win_betas'] for p in preps], dim=0)    # (B, win_len, 10)
        win_tr_b = torch.stack([p['win_tr'] for p in preps], dim=0)          # (B, win_len, 3)
        R_center_b = torch.stack([p['R_center'] for p in preps], dim=0)      # (B, 3, 3)
        t_center_b = torch.stack([p['t_center'] for p in preps], dim=0)      # (B, 3)

        # 构建 padded contact 张量
        Cmax = max(p['contact_count'] for p in preps)
        fi_idx = torch.zeros(B, Cmax, dtype=torch.long, device=device)
        pidx_idx = torch.zeros(B, Cmax, dtype=torch.long, device=device)
        conf_b = torch.zeros(B, Cmax, device=device)
        normal_b = torch.zeros(B, Cmax, 3, device=device)
        offset_b = torch.zeros(B, Cmax, device=device)
        target_b = torch.zeros(B, Cmax, device=device)
        anchor_b = torch.zeros(B, Cmax, 3, device=device)
        mask_b = torch.zeros(B, Cmax, device=device)
        for bi, p in enumerate(preps):
            for ci, (fi, pid, pidx, name, conf, normal, offset, target_offset, anchor) in enumerate(p['plane_tensors']):
                fi_idx[bi, ci] = int(fi)
                pidx_idx[bi, ci] = int(pidx)
                conf_b[bi, ci] = float(conf)
                normal_b[bi, ci] = normal
                offset_b[bi, ci] = offset
                target_b[bi, ci] = float(target_offset)
                anchor_b[bi, ci] = anchor
                mask_b[bi, ci] = 1.0
        weight_sum_b = (conf_b * mask_b).sum(dim=1).clamp_min(1e-6)  # (B,)
        batch_ar = torch.arange(B, device=device)[:, None].expand(B, Cmax)

        delta_bp = torch.zeros(B, win_len, nleg, device=device, requires_grad=True)
        delta_tr = torch.zeros(B, win_len, 3, device=device, requires_grad=optimize_transl)
        params = [delta_bp] + ([delta_tr] if optimize_transl else [])
        optimizer = torch.optim.Adam(params, lr=lr)

        for _ in range(opt_steps):
            optimizer.zero_grad()
            bp_mod = bp_orig_b.clone()
            bp_mod[:, :, leg_bp_indices] = bp_orig_b[:, :, leg_bp_indices] + delta_bp
            tr_mod = win_tr_b + (delta_tr * transl_mask[None, None, :] if optimize_transl else 0.0)
            if use_vertices:
                pts_flat = self._forward_selected_smpl_vertices(
                    smplx_model, smplx2smpl_sel,
                    win_orient_b.reshape(B * win_len, 3),
                    bp_mod.reshape(B * win_len, 63),
                    win_betas_b.reshape(B * win_len, -1),
                    tr_mod.reshape(B * win_len, 3),
                )
                P = pts_flat.shape[1]
                points_world_opt = pts_flat.reshape(B, win_len, P, 3)
            else:
                points_world_opt = self.endecoder.fk_v2(
                    body_pose=bp_mod,
                    betas=win_betas_b,
                    global_orient=win_orient_b,
                    transl=tr_mod,
                )  # (B, win_len, 22, 3)

            pts_sel = points_world_opt[batch_ar, fi_idx, pidx_idx]  # (B, Cmax, 3)
            p_cam = torch.einsum('bij,bcj->bci', R_center_b, pts_sel) + t_center_b[:, None, :]
            signed_dist = (normal_b * p_cam).sum(dim=-1) + offset_b  # (B, Cmax)
            diff = signed_dist - target_b
            if ground_loss == 'penetration':
                diff = F.relu(-diff)
            contact_elem = conf_b * gmof(diff, sigma=loss_sigma) * mask_b
            loss_contact_b = contact_elem.sum(dim=1) / weight_sum_b  # (B,)

            if anchor_w > 0.0:
                tangent = p_cam - anchor_b
                proj = (tangent * normal_b).sum(dim=-1, keepdim=True)
                tangent = tangent - proj * normal_b
                anchor_elem = conf_b * gmof(tangent, sigma=anchor_sigma).sum(dim=-1) * mask_b
                loss_anchor_b = anchor_elem.sum(dim=1) / weight_sum_b
            else:
                loss_anchor_b = torch.zeros(B, device=device)

            loss_reg_b = delta_bp.pow(2).mean(dim=(1, 2))  # (B,)
            if win_len >= 2:
                loss_smooth_b = (delta_bp[:, 1:] - delta_bp[:, :-1]).pow(2).mean(dim=(1, 2))
            else:
                loss_smooth_b = torch.zeros(B, device=device)

            if optimize_transl:
                delta_tr_eff = delta_tr * transl_mask[None, None, :]
                loss_tr_reg_b = delta_tr_eff.pow(2).mean(dim=(1, 2))
                if win_len >= 2:
                    loss_tr_smooth_b = (delta_tr_eff[:, 1:] - delta_tr_eff[:, :-1]).pow(2).mean(dim=(1, 2))
                else:
                    loss_tr_smooth_b = torch.zeros(B, device=device)
            else:
                loss_tr_reg_b = torch.zeros(B, device=device)
                loss_tr_smooth_b = torch.zeros(B, device=device)

            # 各窗口 total loss 相加：因参数按窗口不相交，梯度与逐窗口独立优化完全一致。
            total_loss = (
                loss_contact_w * loss_contact_b
                + anchor_w * loss_anchor_b
                + loss_bp_reg_w * loss_reg_b
                + loss_smooth_w * loss_smooth_b
                + transl_reg_w * loss_tr_reg_b
                + transl_smooth_w * loss_tr_smooth_b
            ).sum()
            total_loss.backward()
            optimizer.step()

        refined_bp_b = bp_orig_b.clone()
        refined_bp_b[:, :, leg_bp_indices] = bp_orig_b[:, :, leg_bp_indices] + delta_bp.detach()
        refined_tr_b = win_tr_b + (delta_tr.detach() * transl_mask[None, None, :] if optimize_transl else 0.0)

        results = []
        for bi, p in enumerate(preps):
            self.logger.debug(
                f"[LocalPlaneContactIK-batch] Window {p['center']}: point_type={p['point_type']}, "
                f"contacts={p['contact_count']}, rejected_by_quality={p['rejected_by_quality']}, "
                f"anchor_w={anchor_w:.3f}, optimize_transl={optimize_transl}, "
                f"delta_bp_mean={float(delta_bp[bi].detach().abs().mean()):.6f}"
            )
            if optimize_transl:
                results.append((refined_bp_b[bi].detach().cpu(), refined_tr_b[bi].detach().cpu()))
            else:
                results.append(refined_bp_b[bi].detach().cpu())
        return results

    def _apply_ground_plane_contact_ik(
        self,
        global_orient_w: torch.Tensor,
        body_pose_aa: torch.Tensor,
        betas: torch.Tensor,
        win_transl: torch.Tensor,
        static_conf_logits: Optional[torch.Tensor],
        depth_map: torch.Tensor,
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        K: torch.Tensor,
        center: int,
        start: int,
        end: int,
        human_mask: Optional[np.ndarray],
    ) -> torch.Tensor:
        """用扣除人体后补全出的地面平面约束静态脚部接触点。"""
        plane = self._fit_ground_plane_from_depth(depth_map, K, human_mask)
        if plane is None:
            self.logger.debug(f"[GroundContactIK] Window {center}: ground plane fit failed, skipping")
            return body_pose_aa

        device = self.device
        win_len = end - start
        opt_steps = int(self.config.get('contact_ik_steps', 30))
        lr = float(self.config.get('contact_ik_lr', 0.003))
        loss_contact_w = float(self.config.get('loss_contact_w', 5.0))
        loss_sigma = float(self.config.get('loss_contact_sigma', 0.05))
        loss_bp_reg_w = float(self.config.get('loss_contact_bp_reg_w', 0.1))
        loss_smooth_w = float(self.config.get('loss_contact_smooth_w', 0.5))
        conf_thr = float(self.config.get('contact_ik_conf_thr', 0.5))
        ground_loss = str(self.config.get('contact_ik_ground_loss', 'l2'))

        contact_joint_ids = [int(x) for x in self.config.get('contact_ik_ground_joint_ids', [10, 11])]
        conf_idx_by_jid = {7: 0, 10: 1, 8: 2, 11: 3}
        offsets_cfg = self.config.get('contact_ik_ground_joint_offsets', {7: 0.08, 8: 0.08, 10: 0.02, 11: 0.02})
        offsets_cfg = {int(k): float(v) for k, v in offsets_cfg.items()}

        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits[start:end].to(device))
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        contact_items = []
        with torch.no_grad():
            for fi in range(win_len):
                for jid in contact_joint_ids:
                    conf_idx = conf_idx_by_jid.get(jid)
                    if conf_idx is None:
                        continue
                    conf = float(static_conf[fi, conf_idx].item())
                    if conf < conf_thr:
                        continue
                    contact_items.append((fi, jid, conf, offsets_cfg.get(jid, 0.0)))
        if not contact_items:
            self.logger.debug(f"[GroundContactIK] Window {center}: no static foot contacts, skipping")
            return body_pose_aa

        scale_factor = 1.0
        if str(self.config.get('contact_ik_scale_mode', 'none')) == 'joint_median':
            scale_factor = 1.0  # ground-plane mode intentionally avoids human-depth scale fitting
        normal = torch.from_numpy(plane['normal']).to(device=device, dtype=torch.float32)
        offset = torch.tensor(float(plane['offset']) / max(scale_factor, 1e-6), device=device)

        opt_joints = [int(j) for j in self.config.get('contact_ik_opt_joints', [1, 2, 4, 5, 7, 8, 10, 11])]
        leg_bp_indices = []
        for j in opt_joints:
            if 1 <= j <= 21:
                leg_bp_indices.extend([(j - 1) * 3, (j - 1) * 3 + 1, (j - 1) * 3 + 2])
        if not leg_bp_indices:
            self.logger.debug(f"[GroundContactIK] Window {center}: no optimizable joints, skipping")
            return body_pose_aa

        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        bp_orig = body_pose_aa.to(device).clone().detach()
        win_tr = win_transl.to(device)
        R_center = R_cw[center].to(device)
        t_center = t_cw[center].to(device)

        if str(self.config.get('contact_ik_ground_offset_mode', 'fixed')) == 'joint_median':
            with torch.no_grad():
                joints_world_init = self.endecoder.fk_v2(
                    body_pose=bp_orig.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=win_tr.unsqueeze(0),
                )[0]
                dist_by_jid = {}
                for fi, jid, _, extra_offset in contact_items:
                    j_cam = R_center @ joints_world_init[fi, jid] + t_center
                    signed_dist = float((torch.dot(normal, j_cam) + offset).detach().cpu())
                    dist_by_jid.setdefault(jid, []).append(signed_dist)
                target_by_jid = {
                    jid: float(np.median(vals)) + offsets_cfg.get(jid, 0.0)
                    for jid, vals in dist_by_jid.items()
                }
                contact_items = [
                    (fi, jid, conf, target_by_jid.get(jid, extra_offset))
                    for fi, jid, conf, extra_offset in contact_items
                ]

        delta_bp = torch.zeros(win_len, len(leg_bp_indices), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_bp], lr=lr)
        final_loss_contact = torch.tensor(0.0, device=device)

        for step in range(opt_steps):
            optimizer.zero_grad()
            bp_mod = bp_orig.clone()
            bp_mod[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp

            with torch.enable_grad():
                joints_world_opt = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=win_tr.unsqueeze(0),
                )[0]

            loss_contact = torch.tensor(0.0, device=device)
            weight_sum = 0.0
            for fi, jid, conf, target_offset in contact_items:
                j_cam = R_center @ joints_world_opt[fi, jid] + t_center
                signed_dist = torch.dot(normal, j_cam) + offset
                diff = signed_dist - float(target_offset)
                if ground_loss == 'penetration':
                    diff = F.relu(-diff)
                loss_contact = loss_contact + float(conf) * gmof(diff, sigma=loss_sigma)
                weight_sum += float(conf)
            loss_contact = loss_contact / max(weight_sum, 1e-6)
            final_loss_contact = loss_contact.detach()

            loss_reg = delta_bp.pow(2).mean()
            loss_smooth = torch.tensor(0.0, device=device)
            if win_len >= 2:
                loss_smooth = (delta_bp[1:] - delta_bp[:-1]).pow(2).mean()
            total_loss = loss_contact_w * loss_contact + loss_bp_reg_w * loss_reg + loss_smooth_w * loss_smooth
            total_loss.backward()
            optimizer.step()

        refined_bp = bp_orig.clone()
        refined_bp[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp.detach()
        self.logger.debug(
            f"[GroundContactIK] Window {center}: contacts={len(contact_items)}, "
            f"plane_inliers={plane['inliers']}/{plane['candidates']} "
            f"({plane['inlier_ratio']:.2f}), residual={plane['mean_residual']:.4f}, "
            f"loss_contact={float(final_loss_contact):.6f}, "
            f"delta_bp_mean={float(delta_bp.detach().abs().mean()):.6f}"
        )
        return refined_bp.cpu()

    # ------------------------------------------------------------------
    # Depth Contact IK: 基于深度图穿透信号优化 body_pose
    # ------------------------------------------------------------------
    def _apply_depth_contact_ik(
        self,
        global_orient_w: torch.Tensor,   # (F_total, 3) aa
        body_pose_aa: torch.Tensor,      # (win_len, 63) 窗口 body_pose
        betas: torch.Tensor,             # (F_total, 10)
        win_transl: torch.Tensor,        # (win_len, 3) 窗口 transl
        static_conf_logits: Optional[torch.Tensor],  # (F_total, 6)
        depth_map: torch.Tensor,         # (H, W) 中心帧深度图
        R_cw: torch.Tensor,              # (F_total, 3, 3)
        t_cw: torch.Tensor,              # (F_total, 3)
        K: torch.Tensor,                 # (3, 3)
        center: int,
        start: int,
        end: int,
        human_mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """基于深度图的 Contact IK：优化窗口内 body_pose 使穿透接触关节对齐场景表面。

        流程：
        1. 用中心帧所有 22 个关节计算局部 scale_factor = median(z_scene / z_smpl)
        2. 对窗口内每帧的静态脚部关节，计算 scale-aligned 后的目标 3D 位置
           （从深度图反投影得到相机坐标系下目标点，再转回世界坐标系）
        3. Adam 优化 delta_body_pose（只修改腿部运动链），使 FK 后关节到达目标位置
        4. 同时加入时序平滑正则 + body_pose 正则

        只修改腿部运动链：L_Hip(1)→L_Knee(4)→L_Ankle(7)→L_Foot(10)
                          R_Hip(2)→R_Knee(5)→R_Ankle(8)→R_Foot(11)

        Returns:
            refined body_pose: (win_len, 63)
        """
        scene_model = str(self.config.get('contact_ik_scene_model', 'raw_depth'))
        if scene_model == 'ground_plane':
            return self._apply_ground_plane_contact_ik(
                global_orient_w=global_orient_w,
                body_pose_aa=body_pose_aa,
                betas=betas,
                win_transl=win_transl,
                static_conf_logits=static_conf_logits,
                depth_map=depth_map,
                R_cw=R_cw,
                t_cw=t_cw,
                K=K,
                center=center,
                start=start,
                end=end,
                human_mask=human_mask,
            )
        if scene_model in ('local_planes', 'josh_planes', 'rich_floor', 'gt_floor', 'rich_gt_floor'):
            return self._apply_local_plane_contact_ik(
                global_orient_w=global_orient_w,
                body_pose_aa=body_pose_aa,
                betas=betas,
                win_transl=win_transl,
                static_conf_logits=static_conf_logits,
                depth_map=depth_map,
                R_cw=R_cw,
                t_cw=t_cw,
                K=K,
                center=center,
                start=start,
                end=end,
                human_mask=human_mask,
            )

        device = self.device
        win_len = end - start
        center_local = center - start

        # --- 配置 ---
        opt_steps = self.config.get('contact_ik_steps', 30)
        lr = self.config.get('contact_ik_lr', 0.003)
        loss_contact_w = self.config.get('loss_contact_w', 5.0)
        loss_contact_sigma = self.config.get('loss_contact_sigma', 0.05)
        loss_bp_reg_w = self.config.get('loss_contact_bp_reg_w', 0.1)
        loss_smooth_w = self.config.get('loss_contact_smooth_w', 0.5)
        pen_eps = 0.03  # 3cm 穿透阈值

        # --- static_conf ---
        CONTACT_JOINTS = {
            0: 7,   # L_Ankle
            1: 10,  # L_Foot
            2: 8,   # R_Ankle
            3: 11,  # R_Foot
        }
        # 腿部运动链关节 (body_pose 索引)
        LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
        leg_bp_indices = []
        for j in LEG_JOINTS:
            leg_bp_indices.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

        if static_conf_logits is not None:
            static_conf = torch.sigmoid(static_conf_logits[start:end].to(device))
        else:
            static_conf = torch.zeros(win_len, 6, device=device)

        # --- FK 计算当前关节位置 ---
        win_orient = global_orient_w[start:end].to(device)
        win_betas = betas[start:end].to(device)
        win_bp = body_pose_aa.to(device)
        win_tr = win_transl.to(device)

        R_center = R_cw[center].to(device)
        t_center = t_cw[center].to(device)
        K_dev = K.to(device)
        K_inv = torch.inverse(K_dev)
        depth_dev = depth_map.to(device)
        dH, dW = depth_dev.shape

        # --- Phase 1: 计算局部 scale_factor ---
        with torch.no_grad():
            joints_world_init = self.endecoder.fk_v2(
                body_pose=win_bp.unsqueeze(0),
                betas=win_betas.unsqueeze(0),
                global_orient=win_orient.unsqueeze(0),
                transl=win_tr.unsqueeze(0),
            )[0]  # (win_len, 22, 3)

        j_center_world = joints_world_init[center_local]  # (22, 3)
        j_center_cam = (R_center @ j_center_world.T).T + t_center  # (22, 3)
        pj_center = (K_dev @ j_center_cam.T).T
        pj_center_2d = pj_center[:, :2] / (pj_center[:, 2:3] + 1e-6)

        z_smpl_list, z_scene_list = [], []
        for jid in range(22):
            z_s = j_center_cam[jid, 2].item()
            if z_s <= 0.01:
                continue
            u_px = int(round(pj_center_2d[jid, 0].item()))
            v_px = int(round(pj_center_2d[jid, 1].item()))
            if u_px < 0 or u_px >= dW or v_px < 0 or v_px >= dH:
                continue
            z_d = depth_dev[v_px, u_px].item()
            if z_d < 0.1 or z_d > 100.0:
                continue
            z_smpl_list.append(z_s)
            z_scene_list.append(z_d)

        if len(z_smpl_list) < 3:
            self.logger.debug(f"[ContactIK] Window {center}: too few joints for scale, skipping")
            return body_pose_aa

        scale_factor = float(np.median(np.array(z_scene_list) / (np.array(z_smpl_list) + 1e-8)))

        # --- Phase 2: 为每帧每个穿透的静态脚部关节计算目标世界坐标 ---
        # contact_targets[fi][jid] = target_world_pos (3,)
        contact_targets = {}  # {fi: {jid: (3,) tensor}}
        n_targets = 0

        for fi in range(win_len):
            for conf_idx, jid in CONTACT_JOINTS.items():
                conf_val = static_conf[fi, conf_idx].item()
                if conf_val < 0.5:
                    continue

                j_world = joints_world_init[fi, jid]
                j_cam = R_center @ j_world + t_center
                z_smpl_aligned = j_cam[2].item() * scale_factor

                pj = K_dev @ j_cam
                u = pj[0].item() / (pj[2].item() + 1e-6)
                v = pj[1].item() / (pj[2].item() + 1e-6)
                ui, vi = int(round(u)), int(round(v))

                if ui < 0 or ui >= dW or vi < 0 or vi >= dH:
                    continue

                z_scene = depth_dev[vi, ui].item()
                if z_scene < 0.1 or z_scene > 100.0:
                    continue

                penetration = z_smpl_aligned - z_scene - pen_eps
                if penetration <= 0:
                    continue  # 没穿透，不需要修正

                # 用深度图反投影得到目标相机坐标系 3D 位置
                # p_target_cam = z_scene * K^{-1} @ [u, v, 1]^T
                # 但注意 z_scene 是 metric depth，而我们的相机坐标系有 scale 偏差
                # 所以目标深度应该是 z_scene / scale_factor（转回 SMPL 的 scale 空间）
                z_target_smpl_scale = z_scene / scale_factor
                pixel_homo = torch.tensor([u, v, 1.0], device=device)
                p_target_cam = z_target_smpl_scale * (K_inv @ pixel_homo)  # (3,)

                # 转回世界坐标系：p_world = R_center^T @ (p_cam - t_center)
                p_target_world = R_center.T @ (p_target_cam - t_center)

                contact_targets.setdefault(fi, {})[jid] = p_target_world.detach()
                n_targets += 1

        if n_targets == 0:
            self.logger.debug(f"[ContactIK] Window {center}: no penetrating contacts, skipping")
            return body_pose_aa

        self.logger.debug(
            f"[ContactIK] Window {center}: {n_targets} targets, "
            f"scale={scale_factor:.3f}, optimizing {opt_steps} steps"
        )

        # --- Phase 3: Adam 优化 delta_body_pose ---
        bp_orig = win_bp.clone().detach()
        delta_bp = torch.zeros(win_len, len(leg_bp_indices), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta_bp], lr=lr)

        for step in range(opt_steps):
            optimizer.zero_grad()

            bp_mod = bp_orig.clone()
            bp_mod[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp

            # 全局 FK（带 global_orient 和 transl）
            with torch.enable_grad():
                joints_world_opt = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=win_betas.unsqueeze(0),
                    global_orient=win_orient.unsqueeze(0),
                    transl=win_tr.unsqueeze(0),
                )[0]  # (win_len, 22, 3)

            # Loss 1: 接触关节到目标位置的距离
            loss_contact = torch.tensor(0.0, device=device)
            cnt = 0
            for fi, targets in contact_targets.items():
                for jid, target_pos in targets.items():
                    diff = joints_world_opt[fi, jid] - target_pos
                    loss_contact = loss_contact + gmof(diff, sigma=loss_contact_sigma).sum()
                    cnt += 1
            if cnt > 0:
                loss_contact = loss_contact / cnt

            # Loss 2: body_pose 正则化（不偏离原始太多）
            loss_reg = delta_bp.pow(2).mean()

            # Loss 3: 时序平滑（相邻帧的 delta 应该相近）
            loss_smooth = torch.tensor(0.0, device=device)
            if win_len >= 2:
                delta_diff = delta_bp[1:] - delta_bp[:-1]
                loss_smooth = delta_diff.pow(2).mean()

            total_loss = (
                loss_contact_w * loss_contact
                + loss_bp_reg_w * loss_reg
                + loss_smooth_w * loss_smooth
            )

            total_loss.backward()
            optimizer.step()

            if step == 0 or step == opt_steps - 1:
                self.logger.debug(
                    f"[ContactIK step={step}] "
                    f"contact={loss_contact.item():.6f}, "
                    f"reg={loss_reg.item():.6f}, "
                    f"smooth={loss_smooth.item():.6f}, "
                    f"total={total_loss.item():.6f}"
                )

        # --- Phase 4: 应用修正 ---
        with torch.no_grad():
            refined_bp = bp_orig.clone()
            refined_bp[:, leg_bp_indices] = bp_orig[:, leg_bp_indices] + delta_bp.detach()

        delta_norm = delta_bp.detach().abs().mean().item()
        self.logger.debug(
            f"[ContactIK] Window {center}: done, "
            f"delta_bp_mean={delta_norm:.6f}, {n_targets} targets"
        )

        return refined_bp.cpu()

    # ------------------------------------------------------------------
    # 图像模型 3D body pose 纠正
    # ------------------------------------------------------------------
    def _prepare_img_model_joints(
        self,
        phmr_img_smpl: Dict[str, torch.Tensor],
        betas_video: torch.Tensor,  # (F, 10) 视频头的 betas
    ) -> torch.Tensor:
        """从图像模型的 SMPL rotmat 中提取 body-local 关节位置。

        在 body-local space（无 global_orient、无 transl）中做 FK，
        完全消除 global_orient 不一致的影响。
        统一使用视频头的 betas 以确保骨骼长度一致。

        Returns:
            img_j3d_local: (F, 22, 3) body-local root-relative joints
        """
        from lib.utils.rotation_conversions import matrix_to_axis_angle

        rotmat = phmr_img_smpl['rotmat']  # (N, 22, 3, 3)

        # 提取 body_pose 部分（joint 1-21），转为 axis-angle
        body_rotmat = rotmat[:, 1:, :, :]          # (N, 21, 3, 3)
        body_aa = matrix_to_axis_angle(body_rotmat) # (N, 21, 3)
        body_aa_flat = body_aa.reshape(-1, 63)      # (N, 63)

        # Body-local FK: 不传 global_orient 和 transl，消除全局朝向差异
        with torch.no_grad():
            joints_local = self.endecoder.fk_v2(
                body_pose=body_aa_flat.unsqueeze(0).to(self.device),
                betas=betas_video.unsqueeze(0).to(self.device),
                global_orient=None,
                transl=None,
            )[0]  # (N, 22, 3)

        # Root-relative
        img_j3d_local = joints_local - joints_local[:, 0:1, :]
        return img_j3d_local

    def _refine_body_pose_with_img_model(
        self,
        body_pose_aa: torch.Tensor,       # (F, 63) axis-angle
        betas: torch.Tensor,              # (F, 10)
        img_j3d_local: torch.Tensor,      # (F, 22, 3) body-local root-relative
    ) -> torch.Tensor:
        """通过图像模型的单帧 body-local 3D 关节约束来优化 body_pose。

        核心思想：
        - 在 body-local space（无 global_orient / transl）做 FK 和比较
        - 完全消除 global_orient 不一致的影响
        - 优化四肢关节的 body_pose_aa delta，使 FK 的 body-local joints 匹配图像模型

        Returns:
            refined_body_pose: (F, 63) 修正后的 body_pose
        """
        device = self.device
        F_total = body_pose_aa.shape[0]
        opt_steps = self.config['img_pose_opt_steps']
        lr = self.config['img_pose_opt_lr']
        limb_joints = self.config['img_pose_limb_joints']

        # body_pose 索引映射: FK joint i (i>=1) → body_pose[(i-1)*3 : i*3]
        limb_bp_indices = []
        for j in limb_joints:
            if j == 0 or j > 21:
                continue
            limb_bp_indices.extend([(j-1)*3, (j-1)*3+1, (j-1)*3+2])

        self.logger.info(
            f"[ImgPoseRefine] Optimizing {len(limb_joints)} limb joints "
            f"({len(limb_bp_indices)} params) for {F_total} frames, "
            f"{opt_steps} steps, lr={lr}"
        )

        delta_bp = torch.zeros(F_total, len(limb_bp_indices), device=device, requires_grad=True)

        bp_orig = body_pose_aa.to(device)
        beta = betas.to(device)
        img_target = img_j3d_local.to(device)  # (F, 22, 3)

        optimizer = torch.optim.Adam([delta_bp], lr=lr)

        # 要约束的关节索引（FK-22 空间）
        fk_limb_idxs = [j for j in limb_joints if j < 22]

        for step in range(opt_steps):
            optimizer.zero_grad()

            bp_mod = bp_orig.clone()
            bp_mod[:, limb_bp_indices] = bp_orig[:, limb_bp_indices] + delta_bp

            # Body-local FK: 不传 global_orient / transl
            with torch.enable_grad():
                joints_local = self.endecoder.fk_v2(
                    body_pose=bp_mod.unsqueeze(0),
                    betas=beta.unsqueeze(0),
                    global_orient=None,
                    transl=None,
                )[0]  # (F, 22, 3)

            # Root-relative body-local
            fk_root_rel = joints_local - joints_local[:, 0:1, :]

            fk_limbs = fk_root_rel[:, fk_limb_idxs]
            img_limbs = img_target[:, fk_limb_idxs]

            loss_3d = gmof(
                fk_limbs - img_limbs,
                sigma=self.config['loss_img3d_sigma']
            ).mean()

            loss_reg = delta_bp.pow(2).mean()

            total_loss = (
                self.config['loss_img3d_w'] * loss_3d
                + self.config['loss_bodypose_reg_w'] * loss_reg
            )

            total_loss.backward() #这是用反向传播的？不是用逆运动学吗
            optimizer.step()

            if step == 0 or step == opt_steps - 1:
                self.logger.debug(
                    f"[ImgPoseRefine step={step}] "
                    f"loss_3d={loss_3d.item():.6f}, "
                    f"loss_reg={loss_reg.item():.6f}, "
                    f"total={total_loss.item():.6f}, "
                    f"delta_bp_norm={delta_bp.detach().norm().item():.6f}"
                )

        with torch.no_grad():
            refined_bp_dev = bp_orig.clone()
            refined_bp_dev[:, limb_bp_indices] = bp_orig[:, limb_bp_indices] + delta_bp.detach()
            refined_bp = refined_bp_dev.cpu()

        delta_norm = delta_bp.detach().abs().mean().item()
        self.logger.info(
            f"[ImgPoseRefine] Done. "
            f"Final loss_3d={loss_3d.item():.6f}, "
            f"delta_bp_mean={delta_norm:.6f}"
        )

        return refined_bp

    # ------------------------------------------------------------------
    # OAR 诊断可视化
    # ------------------------------------------------------------------
    def _compute_per_frame_projections(
        self,
        joints_world: torch.Tensor,   # (F, J, 3)
        R_cw: torch.Tensor,            # (F, 3, 3)
        t_cw: torch.Tensor,            # (F, 3)
        K: torch.Tensor,               # (3, 3)
    ) -> torch.Tensor:
        """投影所有帧的世界关节到 2D，返回 (F, J, 2)。"""
        pj_2d, _ = self._project_joints(joints_world, R_cw, t_cw, K)
        return pj_2d  # (F, J, 2)

    def _vis_keypoints_2d(
        self,
        kp2d: torch.Tensor,             # (F, 17, 3) COCO-17 x,y,conf
        proj_2d: torch.Tensor,           # (F, J, 2)  FK-22 投影
        image_paths: List[str],
        output_dir: str,
        frame_stride: int = 5,
        tag: str = 'pre',
    ):
        """可视化 2D 关键点：ViTPose 检测点（绿色）+ FK 投影关节（蓝色）。

        COCO-17 skeleton 连线 + 各关键点置信度标注。
        """
        import cv2

        vis_dir = os.path.join(output_dir, f'oar_keypoints_{tag}')
        os.makedirs(vis_dir, exist_ok=True)

        # COCO-17 skeleton
        COCO_SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),         # head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # upper body
            (5, 11), (6, 12), (11, 12),                # torso
            (11, 13), (13, 15), (12, 14), (14, 16),    # lower body
        ]
        # COCO-17 joint names
        COCO_NAMES = [
            'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
            'L_sho', 'R_sho', 'L_elb', 'R_elb', 'L_wri', 'R_wri',
            'L_hip', 'R_hip', 'L_kne', 'R_kne', 'L_ank', 'R_ank',
        ]

        # FK→COCO mapping for drawing correspondence lines
        _FK_TO_COCO = {
            1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16,
            16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10,
        }

        F_total = kp2d.shape[0]
        vis_frames = list(range(0, F_total, frame_stride))

        for fidx in vis_frames:
            if fidx >= len(image_paths):
                continue
            img = cv2.imread(image_paths[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            canvas = img.copy()

            # 画 COCO skeleton (绿色)
            kp = kp2d[fidx]  # (17, 3)
            for i, j in COCO_SKELETON:
                if kp[i, 2] > 0.3 and kp[j, 2] > 0.3:
                    pt1 = (int(kp[i, 0]), int(kp[i, 1]))
                    pt2 = (int(kp[j, 0]), int(kp[j, 1]))
                    cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

            # 画 COCO-17 关键点（绿色圆圈 + 置信度）
            for ki in range(17):
                x, y, c = kp[ki, 0].item(), kp[ki, 1].item(), kp[ki, 2].item()
                if c > 0.3 and 0 <= x < W and 0 <= y < H:
                    cv2.circle(canvas, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(canvas, f"{c:.1f}", (int(x) + 6, int(y) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            # 画 FK-22 投影关节（蓝色小圆）
            n_joints = proj_2d.shape[1]
            for ji in range(n_joints):
                u, v = int(proj_2d[fidx, ji, 0].item()), int(proj_2d[fidx, ji, 1].item())
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(canvas, (u, v), 3, (255, 100, 0), -1)  # 蓝色

            # 画 FK→COCO 对应线（黄色虚线）
            for fk_idx, coco_idx in _FK_TO_COCO.items():
                if fk_idx >= n_joints:
                    continue
                c = kp[coco_idx, 2].item()
                if c <= 0.3:
                    continue
                u_fk = int(proj_2d[fidx, fk_idx, 0].item())
                v_fk = int(proj_2d[fidx, fk_idx, 1].item())
                u_coco = int(kp[coco_idx, 0].item())
                v_coco = int(kp[coco_idx, 1].item())
                if 0 <= u_fk < W and 0 <= v_fk < H and 0 <= u_coco < W and 0 <= v_coco < H:
                    cv2.line(canvas, (u_fk, v_fk), (u_coco, v_coco), (0, 255, 255), 1)

            # 图例
            cv2.putText(canvas, f"Frame {fidx} | 2D KP ({tag})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, "Green=ViTPose  Blue=FK_proj  Yellow=correspondence",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            out_path = os.path.join(vis_dir, f'{fidx:05d}.jpg')
            cv2.imwrite(out_path, canvas)

        self.logger.info(f"Saved 2D keypoint vis ({tag}, {len(vis_frames)} frames) to {vis_dir}")

    def _vis_reprojection_error(
        self,
        kp2d: torch.Tensor,             # (F, 17, 3) COCO-17 x,y,conf
        proj_2d_pre: torch.Tensor,       # (F, J, 2) FK 投影 (OAR 前)
        proj_2d_post: torch.Tensor,      # (F, J, 2) FK 投影 (OAR 后)
        image_paths: List[str],
        output_dir: str,
        frame_stride: int = 5,
    ):
        """可视化重投影误差：OAR 前后对比。

        生成左右拼接图：
          左：OAR 前的重投影误差（红线 = FK投影→ViTPose）
          右：OAR 后的重投影误差
        底部叠加数值统计。
        """
        import cv2

        vis_dir = os.path.join(output_dir, 'oar_reproj_error')
        os.makedirs(vis_dir, exist_ok=True)

        _FK_TO_COCO = {
            1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16,
            16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10,
        }

        F_total = kp2d.shape[0]
        n_joints = proj_2d_pre.shape[1]
        vis_frames = list(range(0, F_total, frame_stride))

        # 全局统计
        all_err_pre, all_err_post = [], []

        for fidx in vis_frames:
            if fidx >= len(image_paths):
                continue
            img = cv2.imread(image_paths[fidx])
            if img is None:
                continue
            H, W = img.shape[:2]
            canvas_pre = img.copy()
            canvas_post = img.copy()

            kp = kp2d[fidx]  # (17, 3)
            frame_err_pre, frame_err_post = [], []

            for fk_idx, coco_idx in _FK_TO_COCO.items():
                if fk_idx >= n_joints:
                    continue
                c = kp[coco_idx, 2].item()
                if c <= 0.5:
                    continue

                gt_x, gt_y = int(kp[coco_idx, 0].item()), int(kp[coco_idx, 1].item())

                # OAR 前
                px_pre = int(proj_2d_pre[fidx, fk_idx, 0].item())
                py_pre = int(proj_2d_pre[fidx, fk_idx, 1].item())
                err_pre = np.sqrt((px_pre - gt_x)**2 + (py_pre - gt_y)**2)
                frame_err_pre.append(err_pre)

                # OAR 后
                px_post = int(proj_2d_post[fidx, fk_idx, 0].item())
                py_post = int(proj_2d_post[fidx, fk_idx, 1].item())
                err_post = np.sqrt((px_post - gt_x)**2 + (py_post - gt_y)**2)
                frame_err_post.append(err_post)

                # 误差颜色映射 (绿→黄→红)
                def err_color(e, max_e=50.0):
                    ratio = min(e / max_e, 1.0)
                    r = int(255 * ratio)
                    g = int(255 * (1 - ratio))
                    return (0, g, r)  # BGR

                # 画 pre
                if 0 <= px_pre < W and 0 <= py_pre < H:
                    cv2.circle(canvas_pre, (gt_x, gt_y), 5, (0, 255, 0), -1)       # GT - 绿
                    cv2.circle(canvas_pre, (px_pre, py_pre), 4, (0, 0, 255), -1)    # FK - 红
                    cv2.line(canvas_pre, (gt_x, gt_y), (px_pre, py_pre),
                             err_color(err_pre), 2)
                    cv2.putText(canvas_pre, f"{err_pre:.0f}", (px_pre + 5, py_pre - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, err_color(err_pre), 1)

                # 画 post
                if 0 <= px_post < W and 0 <= py_post < H:
                    cv2.circle(canvas_post, (gt_x, gt_y), 5, (0, 255, 0), -1)
                    cv2.circle(canvas_post, (px_post, py_post), 4, (0, 0, 255), -1)
                    cv2.line(canvas_post, (gt_x, gt_y), (px_post, py_post),
                             err_color(err_post), 2)
                    cv2.putText(canvas_post, f"{err_post:.0f}", (px_post + 5, py_post - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, err_color(err_post), 1)

            # 帧内统计
            mean_pre = np.mean(frame_err_pre) if frame_err_pre else 0
            mean_post = np.mean(frame_err_post) if frame_err_post else 0
            all_err_pre.extend(frame_err_pre)
            all_err_post.extend(frame_err_post)

            # 标题
            cv2.putText(canvas_pre, f"Frame {fidx} PRE-OAR | mean={mean_pre:.1f}px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas_post, f"Frame {fidx} POST-OAR | mean={mean_post:.1f}px", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 图例
            cv2.putText(canvas_pre, "Green=ViTPose  Red=FK_proj",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas_post, "Green=ViTPose  Red=FK_proj",
                        (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            combined = np.concatenate([canvas_pre, canvas_post], axis=1)
            out_path = os.path.join(vis_dir, f'{fidx:05d}.jpg')
            cv2.imwrite(out_path, combined)
            #print("saving:", out_path) 
        # 全局统计日志
        if all_err_pre:
            self.logger.info(
                f"Reproj error (px): PRE mean={np.mean(all_err_pre):.1f}, "
                f"median={np.median(all_err_pre):.1f}, max={np.max(all_err_pre):.1f} | "
                f"POST mean={np.mean(all_err_post):.1f}, "
                f"median={np.median(all_err_post):.1f}, max={np.max(all_err_post):.1f}"
            )

        self.logger.info(f"Saved reproj error vis ({len(vis_frames)} frames) to {vis_dir}")

    # ------------------------------------------------------------------
    # Visualisation hook helper
    # ------------------------------------------------------------------
    def _save_before_after_vis(self, data: PipelineData, pre_params: dict, output_dir: str):
        """Save side-by-side visualisation: before (left) vs after (right) OAR.

        Uses pytorch3d Renderer (same as adjacent_smpl_renderer gvhmr_world mode)
        to avoid EGL/pyrender dependency. Generates world-space vertices then
        projects to camera space via T_w2c before rendering.
        """
        import sys
        import cv2

        try:
            from lib.gvhmr.renderer import Renderer as GVHMRRenderer
            smplx_model, smplx2smpl, faces_smpl = self._load_smplx_for_render()
        except Exception as e:
            self.logger.warning(f"Cannot load SMPLX for vis: {e}")
            return

        vis_dir = os.path.join(output_dir, 'oar_comparison')
        os.makedirs(vis_dir, exist_ok=True)

        sp = data.smpl_params
        cam = data.camera_params
        device = torch.device(self.device)

        def _to_t(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            return x.float() if x is not None else None

        # Params
        go_w_after = _to_t(sp.global_orient_w).to(device)
        go_w_before = _to_t(pre_params['global_orient_w']).to(device)
        bp_after = _to_t(sp.body_pose_aa).to(device)
        bp_before = _to_t(pre_params['body_pose_aa']).to(device)
        betas = _to_t(sp.betas).to(device)
        global_trans_after = _to_t(sp.global_trans).to(device)
        global_trans_before = _to_t(pre_params['global_trans']).to(device)

        # Use the same c2w/w2c and original-image K convention as adjacent_smpl_renderer.
        img0 = cv2.imread(data.image_paths[0])
        if img0 is None:
            self.logger.warning("Cannot read first image for OAR comparison")
            return
        H, W = img0.shape[:2]
        hpe_K = data.metadata.get('hpe_K', None)
        if hpe_K is not None:
            K_vis = self._inverse_unpad_K(_to_t(hpe_K), W, H).to(device)
        elif cam is not None and cam.intrinsics is not None:
            K_vis = _to_t(cam.intrinsics).to(device)
        else:
            f_est = (H**2 + W**2) ** 0.5
            K_vis = torch.tensor([[f_est, 0, W/2.], [0, f_est, H/2.], [0, 0, 1]], dtype=torch.float, device=device)

        wt = data.metadata.get('world_transform', {})
        R_wc = _to_t(wt.get('R_wc', cam.world_R if cam is not None and cam.world_R is not None else cam.R)).to(device)
        T_wc = _to_t(wt.get('T_wc', cam.world_T if cam is not None and cam.world_T is not None else cam.T)).to(device)
        R_w2c = R_wc.transpose(-1, -2)
        t_w2c = -(R_w2c @ T_wc.unsqueeze(-1)).squeeze(-1)

        go_c = _to_t(sp.global_orient_c).to(device) if sp.global_orient_c is not None else None
        if go_c is not None:
            from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
            go_w_vis = matrix_to_axis_angle(torch.einsum('fij,fjk->fik', R_wc, axis_angle_to_matrix(go_c)))
            go_w_before = go_w_vis
            go_w_after = go_w_vis

        # Generate world-space vertices for before & after
        with torch.no_grad():
            out_before = smplx_model(
                global_orient=go_w_before,
                body_pose=bp_before[:, :63],
                betas=betas,
                transl=global_trans_before,
            )
            verts_world_before = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in out_before.vertices]
            )  # (F, V, 3)

            out_after = smplx_model(
                global_orient=go_w_after,
                body_pose=bp_after[:, :63],
                betas=betas,
                transl=global_trans_after,
            )
            verts_world_after = torch.stack(
                [torch.matmul(smplx2smpl, v) for v in out_after.vertices]
            )  # (F, V, 3)

        # Setup GVHMR renderer with full intrinsics (keeps cx/cy)
        renderer = GVHMRRenderer(W, H, device=device, faces=faces_smpl, K=K_vis)

        # Sample frames for vis
        F_total = go_w_after.shape[0]
        vis_frames = list(range(0, F_total, max(1, F_total // 30)))[:30]

        for fi in vis_frames:
            try:
                img = cv2.imread(data.image_paths[fi])

                # world -> cam_i
                R_i = R_w2c[fi]  # (3, 3)
                t_i = t_w2c[fi]  # (3,)

                verts_cam_b = torch.einsum('ij,vj->vi', R_i, verts_world_before[fi]) + t_i
                verts_cam_a = torch.einsum('ij,vj->vi', R_i, verts_world_after[fi]) + t_i

                img_before = renderer.render_mesh(verts_cam_b, img.copy(), colors=[0.5, 0.8, 0.5])
                img_after = renderer.render_mesh(verts_cam_a, img.copy(), colors=[0.8, 0.5, 0.5])

                # Side by side (BGR, cv2 format)
                combined = np.concatenate([img_before, img_after], axis=1)
                out_path = os.path.join(vis_dir, f'{fi:05d}.jpg')
                cv2.imwrite(out_path, combined)

            except Exception as e:
                self.logger.warning(f"Vis failed for frame {fi}: {e}")
                continue

        # Cleanup
        del smplx_model, smplx2smpl, verts_world_before, verts_world_after
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"Saved OAR comparison visualisation to {vis_dir}")

    # ------------------------------------------------------------------
    # Window-level optimization (single window)
    # ------------------------------------------------------------------
    def _optimize_single_window(
        self,
        center: int,
        start: int,
        end: int,
        joints_world: torch.Tensor,  # (F, J, 3)
        R_cw: torch.Tensor,
        t_cw: torch.Tensor,
        K: torch.Tensor,
        kp2d: Optional[torch.Tensor],
        depth_map: Optional[torch.Tensor],  # (H, W) 中心帧的单张深度图，或 None
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """对单个窗口做 depth + 2D KP refine，返回中心帧的 delta_t (3,)。

        深度约束直接在相机坐标系下比较：
        - SMPL 关节 w2c 投影得到 z_smpl
        - 深度图在关节投影像素位置采样得到 z_scene
        - 只修正 transl（本质是深度 scale 对齐）
        """
        device = self.device
        fps = self.config['fps']
        win_len = end - start
        center_local = center - start

        j_win = joints_world[start:end].to(device)
        R_win = R_cw[start:end].to(device)
        t_win = t_cw[start:end].to(device)

        from lib.utils.rotation_conversions import axis_angle_to_matrix

        delta_t = torch.zeros(3, device=device, requires_grad=True)

        if self.config['enable_orient_refine']:
            delta_orient_aa = torch.zeros(3, device=device, requires_grad=True)
            opt_params = [delta_t, delta_orient_aa]
        else:
            delta_orient_aa = torch.zeros(3, device=device)
            opt_params = [delta_t]

        optimizer = torch.optim.Adam(opt_params, lr=self.config['opt_lr'])

        # FK→COCO mapping
        n_joints = joints_world.shape[1]
        _FK_TO_COCO_PAIRS = [
            (1, 11), (2, 12), (4, 13), (5, 14), (7, 15), (8, 16),
            (16, 5), (17, 6), (18, 7), (19, 8), (20, 9), (21, 10),
        ]
        fk_coco_pairs = [(fk, coco) for fk, coco in _FK_TO_COCO_PAIRS if fk < n_joints]
        fk_pair_idxs = [p[0] for p in fk_coco_pairs]
        coco_pair_idxs = [p[1] for p in fk_coco_pairs]

        # Gaussian weights for smoothness
        half_win = self.config['window_size'] // 2
        sigma = self.config['sigma']
        offsets = torch.arange(-half_win, half_win + 1).float()
        gauss_w = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
        gauss_w = gauss_w / gauss_w.max()

        K_dev = K.to(device)

        # 预处理深度图
        depth_dev = None
        if depth_map is not None and self.config['enable_depth_constraint']:
            depth_dev = depth_map.to(device)

        for step in range(self.config['opt_steps']):
            optimizer.zero_grad()

            delta_R = axis_angle_to_matrix(delta_orient_aa.unsqueeze(0))
            j_center_mod = (delta_R @ j_win[center_local].unsqueeze(-1)).squeeze(-1) + delta_t

            j_cam_center = (R_win[center_local] @ j_center_mod.unsqueeze(-1)).squeeze(-1) + t_win[center_local]
            pj = (K_dev @ j_cam_center.unsqueeze(-1)).squeeze(-1)
            pj_2d = pj[:, :2] / (pj[:, 2:3] + 1e-6)

            total_loss = torch.tensor(0.0, device=device)

            # Loss 1: 2D reprojection
            if kp2d is not None and len(fk_coco_pairs) > 0:
                kp_center = kp2d[center].to(device)
                gt_2d = kp_center[coco_pair_idxs, :2]
                conf = kp_center[coco_pair_idxs, 2]
                proj_2d = pj_2d[fk_pair_idxs]
                reproj_err = gmof(proj_2d - gt_2d, sigma=self.config['reproj_sigma'])
                reproj_err = reproj_err / 1000.0
                conf_mask = conf > 0.5
                loss_reproj = (conf_mask.unsqueeze(-1) * reproj_err).mean()
                total_loss = total_loss + self.config['loss_reproj_w'] * loss_reproj

            # Loss 2: Depth alignment (相机坐标系 z_smpl vs z_scene)
            if depth_dev is not None:
                dH, dW = depth_dev.shape
                u = pj_2d[:, 0].long().clamp(0, dW - 1)
                v = pj_2d[:, 1].long().clamp(0, dH - 1)
                z_smpl = j_cam_center[:, 2]
                z_scene = depth_dev[v, u]
                valid_depth = (z_scene > 0.1) & (z_scene < 100.0) & (z_smpl > 0)
                penetration = F.relu(z_smpl - z_scene - 0.05)
                if valid_depth.any():
                    loss_depth = gmof(penetration[valid_depth], sigma=self.config['depth_sigma']).mean()
                else:
                    loss_depth = torch.tensor(0.0, device=device)
                total_loss = total_loss + self.config['loss_depth_w'] * loss_depth

            # Loss 3: Smoothness
            t_weights = gauss_w[start - center + half_win:end - center + half_win].to(device)
            delta_t_propagated = t_weights.unsqueeze(-1) * delta_t.unsqueeze(0)
            j_mod_transl = j_win[:, 0, :] + delta_t_propagated
            if win_len >= 3:
                vel = (j_mod_transl[1:] - j_mod_transl[:-1]) * fps
                loss_vel = vel.pow(2).mean()
                acc = (j_mod_transl[2:] + j_mod_transl[:-2] - 2 * j_mod_transl[1:-1]) * fps
                loss_acc = acc.norm(dim=-1).mean()
                total_loss = total_loss + self.config['loss_smooth_vel_w'] * loss_vel
                total_loss = total_loss + self.config['loss_smooth_acc_w'] * loss_acc

            # Loss 4: Regularization
            loss_reg = delta_t.norm() + delta_orient_aa.norm()
            total_loss = total_loss + self.config['loss_reg_w'] * loss_reg

            total_loss.backward()
            optimizer.step()

        return delta_t.detach().cpu()

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------
    def execute(self, data: PipelineData) -> PipelineData:
        """窗口级独立 Pipeline：每个窗口独立创建 transl 副本，
        渲染 pre → depth+2DKP refine → IK → 渲染 post。
        同时仍将全局修正（高斯合并）写回 data.smpl_params 供 eval 使用。
        """
        sp = data.smpl_params
        cam = data.camera_params

        def _to_tensor(x):
            if x is None:
                return None
            return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()

        # Save pre-OAR params for visualisation
        pre_params = {
            'global_orient_w': (
                sp.global_orient_w.clone() if isinstance(sp.global_orient_w, torch.Tensor)
                else sp.global_orient_w.copy()
            ),
            'body_pose_aa': (
                sp.body_pose_aa.clone() if isinstance(sp.body_pose_aa, torch.Tensor)
                else sp.body_pose_aa.copy()
            ),
            'global_trans': (
                sp.global_trans.clone() if isinstance(sp.global_trans, torch.Tensor)
                else sp.global_trans.copy()
            ),
        }

        global_orient_w = _to_tensor(sp.global_orient_w)   # (F, 3) aa
        global_orient_c = _to_tensor(sp.global_orient_c)   # (F, 3) aa, camera space
        body_pose_aa = _to_tensor(sp.body_pose_aa)          # (F, 63)
        betas = _to_tensor(sp.betas)                         # (F, 10)
        global_trans = _to_tensor(sp.global_trans)           # (F, 3)
        trans_c = _to_tensor(sp.trans)                       # (F, 3), camera space
        static_conf_logits = _to_tensor(sp.static_conf_logits)

        F_total = global_orient_w.shape[0]
        valid_frame_mask_for_oar = None
        if data.valid_frames_mask is not None:
            mask_np = np.asarray(data.valid_frames_mask).astype(bool)
            if int(mask_np.sum()) == int(F_total):
                valid_frame_mask_for_oar = mask_np
        external_contact_path = self.config.get('external_contact_path', None)
        if external_contact_path:
            contact_np = np.load(external_contact_path)
            contact = torch.from_numpy(contact_np).float()
            vertex_ids = [int(v) for v in self.config.get('external_contact_vertex_ids', [3216, 3387, 6617, 6787])]
            contact_thr = float(self.config.get('external_contact_threshold', 0.5))
            if contact.ndim != 2 or contact.shape[1] <= max(vertex_ids):
                raise ValueError(f"Invalid external contact shape {tuple(contact.shape)} for vertex ids {vertex_ids}")
            n_contact = min(F_total, contact.shape[0])
            external_logits = torch.full((F_total, 6), -10.0)
            vals = contact[:n_contact, vertex_ids]
            logits = torch.where(vals >= contact_thr, torch.full_like(vals, 10.0), torch.full_like(vals, -10.0))
            # static_conf columns: 0=L_Ankle/LHeel, 1=L_Foot/LToe, 2=R_Ankle/RHeel, 3=R_Foot/RToe
            external_logits[:n_contact, 1] = logits[:, 0]
            external_logits[:n_contact, 0] = logits[:, 1]
            external_logits[:n_contact, 3] = logits[:, 2]
            external_logits[:n_contact, 2] = logits[:, 3]
            static_conf_logits = external_logits
            self.logger.info(
                f"Using external JOSH/DECO contact from {external_contact_path}: "
                f"frames={n_contact}/{F_total}, vertex_ids={vertex_ids}, "
                f"positive_ratio={(vals >= contact_thr).float().mean().item():.4f}"
            )

        window_size = self.config['window_size']
        stride = self.config['stride']
        sigma = self.config['sigma']
        half_win = window_size // 2
        self.logger.info(f"OAR: {F_total} frames, window={window_size}, stride={stride}")

        # --- 1. Get camera w2c ---
        wt = data.metadata.get('world_transform', {})
        R_wc = _to_tensor(wt.get('R_wc', cam.world_R if cam.world_R is not None else cam.R))
        T_wc = _to_tensor(wt.get('T_wc', cam.world_T if cam.world_T is not None else cam.T))
        if valid_frame_mask_for_oar is not None:
            mask_t = torch.from_numpy(valid_frame_mask_for_oar).to(R_wc.device)
            if int(R_wc.shape[0]) == len(valid_frame_mask_for_oar):
                self.logger.info(f"OAR filtering camera by valid mask: {R_wc.shape[0]} -> {F_total}")
                R_wc = R_wc[mask_t]
                T_wc = T_wc[mask_t]
            if len(data.image_paths) == len(valid_frame_mask_for_oar):
                data.image_paths = [p for p, keep in zip(data.image_paths, valid_frame_mask_for_oar) if keep]
                self.logger.info(f"OAR filtering image paths by valid mask: {len(valid_frame_mask_for_oar)} -> {len(data.image_paths)}")
        if int(R_wc.shape[0]) != int(F_total):
            keep = min(int(R_wc.shape[0]), int(F_total))
            self.logger.warning(f"OAR camera/SMPL length mismatch: camera={R_wc.shape[0]}, smpl={F_total}; truncating to {keep}")
            R_wc = R_wc[:keep]
            T_wc = T_wc[:keep]
            global_orient_w = global_orient_w[:keep]
            global_orient_c = global_orient_c[:keep] if global_orient_c is not None else None
            body_pose_aa = body_pose_aa[:keep]
            betas = betas[:keep]
            global_trans = global_trans[:keep]
            trans_c = trans_c[:keep] if trans_c is not None else None
            static_conf_logits = static_conf_logits[:keep] if static_conf_logits is not None else None
            F_total = keep
        R_cw, t_cw = self._build_w2c(R_wc, T_wc)

        img0 = np.array(Image.open(data.image_paths[0]))
        img_h, img_w = img0.shape[:2]

        # K selection:
        #   K keeps PromptHMR/video-head padded coordinates for optimization.
        #   K_vis is converted back to original-image coordinates for visualization,
        #   matching adjacent_smpl_renderer.prompthmr_camera.
        K_gt = _to_tensor(cam.intrinsics)
        hpe_K = data.metadata.get('hpe_K', None)
        if hpe_K is not None:
            K = _to_tensor(hpe_K)
            K_vis = self._inverse_unpad_K(K, img_w, img_h)
            if K_gt is not None:
                self.logger.info(
                    f"Using HPE padded K for optimization (fx={float(K[0,0]):.1f}); "
                    f"using inverse-unpadded K for visualization "
                    f"(fx={float(K_vis[0,0]):.1f}, cx={float(K_vis[0,2]):.1f}, cy={float(K_vis[1,2]):.1f})"
                )
            else:
                self.logger.info(
                    f"Using HPE padded K for optimization and inverse-unpadded K for visualization: "
                    f"fx={float(K_vis[0,0]):.1f}, cx={float(K_vis[0,2]):.1f}, cy={float(K_vis[1,2]):.1f}"
                )
        else:
            f_est = (img_h**2 + img_w**2) ** 0.5
            K = torch.tensor([[f_est, 0, img_w/2.], [0, f_est, img_h/2.], [0, 0, 1]], dtype=torch.float)
            K_vis = K_gt if K_gt is not None else K
            if K_gt is not None:
                self.logger.info(
                    f"No HPE estimate_K in metadata; using GT/original K for visualization "
                    f"(fx={float(K_vis[0,0]):.1f}) and computed estimate_K for optimization (fx={f_est:.1f})"
                )
            else:
                self.logger.info(f"Using computed estimate_K: fx={f_est:.1f}")
        K_for_depth = K_gt if K_gt is not None else K_vis

        # --- 2. 加载 Metric3D（窗口循环内按需推理，循环后卸载） ---
        intrinsics_np = K_for_depth.numpy() if isinstance(K_for_depth, torch.Tensor) else K_for_depth
        contact_scene_model = str(self.config.get('contact_ik_scene_model', 'raw_depth'))
        contact_uses_gt_floor = contact_scene_model in ('rich_floor', 'gt_floor', 'rich_gt_floor')
        need_depth_maps = bool(
            self.config.get('enable_depth_constraint', False)
            or (self.config.get('enable_depth_contact_ik', False) and not contact_uses_gt_floor)
        )
        if need_depth_maps:
            gc.collect()
            torch.cuda.empty_cache()
            self._load_metric3d()
            self.logger.info("Metric3D loaded (will infer per-window center frame on demand)")

        # --- 3. FK: compute world joints --- #这是前向传播吗？
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
        if global_orient_c is not None:
            # Keep world orientation in the same c2w frame used by adjacent_smpl_renderer.
            # Otherwise OAR window render can look flipped/reversed because PromptHMR's
            # stored global_orient_w may live in a different world frame.
            root_c = axis_angle_to_matrix(global_orient_c.float())
            global_orient_w = matrix_to_axis_angle(
                torch.einsum('fij,fjk->fik', R_wc.float(), root_c)
            ).cpu()
        with torch.no_grad():
            joints_world = self.endecoder.fk_v2(
                body_pose=body_pose_aa.unsqueeze(0).to(self.device),
                betas=betas.unsqueeze(0).to(self.device),
                global_orient=global_orient_w.unsqueeze(0).to(self.device),
                transl=global_trans.unsqueeze(0).to(self.device),
            )
            joints_world = joints_world[0].cpu()  # (F, 22, 3)

        # --- 4. Prepare 2D keypoints ---
        kp2d = None
        if 'vitpose_kp2d' in data.metadata:
            kp2d = _to_tensor(data.metadata['vitpose_kp2d'])
            if valid_frame_mask_for_oar is not None and int(kp2d.shape[0]) == len(valid_frame_mask_for_oar):
                kp2d = kp2d[torch.from_numpy(valid_frame_mask_for_oar).to(kp2d.device)]
                self.logger.info(f"OAR filtering ViTPose keypoints by valid mask: {len(valid_frame_mask_for_oar)} -> {kp2d.shape[0]}")
            if int(kp2d.shape[0]) != int(F_total):
                keep = min(int(kp2d.shape[0]), int(F_total))
                self.logger.warning(f"OAR kp2d/SMPL length mismatch: kp2d={kp2d.shape[0]}, smpl={F_total}; truncating to {keep}")
                kp2d = kp2d[:keep]
            self.logger.info(f"Using ViTPose COCO-17 keypoints: {kp2d.shape}")
        if kp2d is None:
            self.logger.info("No external 2D keypoints, using projected joints as pseudo-kp2d")
            pj2d, _ = self._project_joints(joints_world[:, :22], R_cw, t_cw, K)
            kp2d = torch.zeros(F_total, 17, 3)
            _fk2coco = {1:11, 2:12, 4:13, 5:14, 7:15, 8:16, 13:5, 14:6, 18:7, 19:8, 20:9, 21:10}
            for fk_idx, coco_idx in _fk2coco.items():
                kp2d[:, coco_idx, :2] = pj2d[:, fk_idx]
                kp2d[:, coco_idx, 2] = 1.0

        kp2d_vis = self._inverse_unpad_kp2d(kp2d, img_w, img_h) if hpe_K is not None else kp2d

        transl_2d_stats = {'enabled': False}
        if self.config.get('enable_transl_2d_refine', False):
            trans_c_refined, transl_2d_stats = self._refine_incam_transl_with_2d(
                global_orient_c=global_orient_c,
                body_pose_aa=body_pose_aa,
                betas=betas,
                transl_c=trans_c,
                kp2d=kp2d,
                K=K,
            )
            if transl_2d_stats.get('enabled', False):
                trans_c = trans_c_refined
                skeleton_offset = _to_tensor(sp.skeleton_offset)
                if skeleton_offset is None:
                    skeleton_offset = torch.zeros(3)
                offset = skeleton_offset[None, :, None].float()
                trans_col = trans_c.unsqueeze(-1).float()
                T_col = T_wc.unsqueeze(-1).float()
                global_trans = (R_wc.float() @ (offset + trans_col) + T_col - offset).squeeze(-1).cpu()
                with torch.no_grad():
                    joints_world = self.endecoder.fk_v2(
                        body_pose=body_pose_aa.unsqueeze(0).to(self.device),
                        betas=betas.unsqueeze(0).to(self.device),
                        global_orient=global_orient_w.unsqueeze(0).to(self.device),
                        transl=global_trans.unsqueeze(0).to(self.device),
                    )[0].cpu()
                self.logger.info(
                    "2D transl refine: "
                    f"mean {transl_2d_stats['init_mean_px']:.2f}px -> {transl_2d_stats['final_mean_px']:.2f}px, "
                    f"median {transl_2d_stats['init_median_px']:.2f}px -> {transl_2d_stats['final_median_px']:.2f}px, "
                    f"delta_mean={transl_2d_stats['delta_mean_m']:.4f}m, "
                    f"delta_max={transl_2d_stats['delta_max_m']:.4f}m"
                )
            else:
                self.logger.info(f"2D transl refine skipped: {transl_2d_stats.get('reason', 'unknown')}")

        # --- 5. Load SMPL-X model for rendering (一次性加载) ---
        import cv2
        smplx_model, smplx2smpl, faces_smpl = None, None, None
        p3d_renderer = None
        vis_output_dir = os.path.join(
            data.metadata.get('output_dir', os.path.join('results', data.sequence_name)),
            data.sequence_name,
        )
        # 清理旧窗口可视化，避免 stride/window_size 改动后残留旧帧（如 00655）误导判断。
        for stale_dir in ['oar_window_compare', 'oar_contact_ik_compare', 'oar_window_pre', 'oar_window_post', 'oar_window_post_pre_dep']:
            stale_path = os.path.join(vis_output_dir, stale_dir)
            if os.path.isdir(stale_path):
                shutil.rmtree(stale_path)
        if self.config.get('enable_window_visualization', True):
            try:
                smplx_model, smplx2smpl, faces_smpl = self._load_smplx_for_render()
                from lib.gvhmr.renderer import Renderer as GVHMRRenderer
                p3d_renderer = GVHMRRenderer(
                    img_w, img_h, device=torch.device(self.device), faces=faces_smpl, K=K_vis.to(self.device)
                )
                self.logger.info("Loaded SMPL-X + GVHMRRenderer for window visualization")
            except Exception as e:
                self.logger.warning(f"Cannot load SMPL-X/Renderer for window vis: {e}")

        # --- 6. Sliding-window pipeline: per-window independent ---
        # 渲染偏移量
        render_offsets = self.config.get('render_offsets', [-10, -5, 0, 5, 10])
        contact_ik_opt_joints = [int(j) for j in self.config.get('contact_ik_opt_joints', [])]
        contact_ik_target_joints = [int(j) for j in self.config.get('contact_ik_ground_joint_ids', [])]
        contact_ik_marker_note = (
            f"ContactIK opt={contact_ik_opt_joints}, target={contact_ik_target_joints}, "
            f"loss={self.config.get('contact_ik_ground_loss', 'raw')}, "
            f"offset_mode={self.config.get('contact_ik_ground_offset_mode', 'fixed')}"
        )

        # gauss_w 在窗口内优化中仍有使用
        gauss_offsets = torch.arange(-half_win, half_win + 1).float()
        gauss_w = torch.exp(-gauss_offsets ** 2 / (2 * sigma ** 2))
        gauss_w = gauss_w / gauss_w.max()

        # 存储每个窗口独立结果
        window_results = []
        window_mesh_observations = []

        window_specs = []
        explicit_centers = self.config.get('window_centers')
        window_centers = list(range(0, F_total, stride)) if explicit_centers is None else list(explicit_centers)
        if explicit_centers is not None and (not window_centers
                or any(not isinstance(c, (int, np.integer)) or c < 0 or c >= F_total for c in window_centers)
                or any(b <= a for a, b in zip(window_centers, window_centers[1:]))):
            raise ValueError('OAR window_centers must be nonempty ordered unique frame indices')
        if (self.config.get('include_last_window_center', False) and F_total > 0
                and window_centers[-1] != F_total - 1):
            window_centers.append(F_total - 1)
        for center in window_centers:
            start = max(0, center - half_win)
            end = min(F_total, center + half_win + 1)
            window_specs.append({
                'center': center,
                'start': start,
                'end': end,
                'win_len': end - start,
                'center_local': center - start,
            })

        debug_centers = self.config.get('debug_window_centers', None)
        if debug_centers is not None:
            wanted = {int(c) for c in debug_centers}
            window_specs = [s for s in window_specs if int(s['center']) in wanted]
            self.logger.info(f"OAR debug_window_centers={sorted(wanted)} -> {len(window_specs)} window(s)")
        else:
            max_debug_windows = int(self.config.get('max_debug_windows', 0) or 0)
            if max_debug_windows > 0:
                window_specs = window_specs[:max_debug_windows]
                self.logger.info(f"OAR max_debug_windows={max_debug_windows} -> {len(window_specs)} window(s)")

        skating_mode = self.config.get('window_skating_mode', 'ik')
        pp_static_joint_windows = {}
        pp_static_joint_batch_stats = {'enabled': False}
        if skating_mode == 'pp_static_joint' and self.config.get('pp_static_joint_use_batch', True):
            import time
            from collections import defaultdict

            t_pp_batch = time.perf_counter()
            grouped_specs = defaultdict(list)
            for spec in window_specs:
                grouped_specs[spec['win_len']].append(spec)

            pp_batch_size = max(1, int(self.config.get('pp_static_joint_batch_size', 64)))
            batch_calls = 0
            for _, specs in grouped_specs.items():
                for chunk_start in range(0, len(specs), pp_batch_size):
                    chunk = specs[chunk_start:chunk_start + pp_batch_size]
                    batch_orient = torch.stack([global_orient_w[s['start']:s['end']] for s in chunk], dim=0)
                    batch_body_pose = torch.stack([body_pose_aa[s['start']:s['end']] for s in chunk], dim=0)
                    batch_betas = torch.stack([betas[s['start']:s['end']] for s in chunk], dim=0)
                    batch_transl = torch.stack([global_trans[s['start']:s['end']] for s in chunk], dim=0)
                    batch_static = (
                        torch.stack([static_conf_logits[s['start']:s['end']] for s in chunk], dim=0)
                        if static_conf_logits is not None else None
                    )
                    center_idxs = torch.tensor([s['center_local'] for s in chunk], dtype=torch.long)
                    corrected_batch = self._apply_pp_static_joint_batch(
                        batch_orient, batch_body_pose, batch_betas, batch_transl, batch_static,
                        center_idxs=center_idxs,
                    )
                    for spec, corrected in zip(chunk, corrected_batch):
                        pp_static_joint_windows[spec['center']] = corrected.clone()
                    batch_calls += 1

            pp_elapsed = time.perf_counter() - t_pp_batch
            pp_static_joint_batch_stats = {
                'enabled': True,
                'num_windows': len(window_specs),
                'num_length_groups': len(grouped_specs),
                'batch_size': pp_batch_size,
                'batch_calls': batch_calls,
                'elapsed_sec': pp_elapsed,
            }
            self.logger.info(
                f"Batch pp_static_joint prepared {len(window_specs)} windows "
                f"in {pp_elapsed:.3f}s ({len(grouped_specs)} length groups, {batch_calls} batch calls)"
            )

        # --- 6.a 批量预推理各窗口中心帧深度图（GPU 利用率优化） ---
        # align_mode=none 时中心帧原始深度与 pose 无关；即使需要 align，也在循环内基于
        # 缓存的原始深度做（align 只是缩放），因此这里一次性批量推理原始深度总是安全的。
        center_depth_cache: Dict[int, torch.Tensor] = {}
        if (need_depth_maps and self.metric3d_model is not None
                and self.config.get('depth_infer_use_batch', True)):
            import time as _time_dep
            t_dep = _time_dep.perf_counter()
            depth_centers = [s['center'] for s in window_specs]
            try:
                center_depth_cache = self._infer_center_depths_batch(
                    depth_centers, data.image_paths, intrinsics_np,
                )
                self.logger.info(
                    f"Batch depth inference prepared {len(center_depth_cache)} center frames "
                    f"in {_time_dep.perf_counter() - t_dep:.3f}s "
                    f"(batch_size={self.config.get('depth_infer_batch_size', 8)})"
                )
            except Exception as e:
                self.logger.warning(
                    f"Batch depth inference failed ({e}); falling back to per-window inference"
                )
                center_depth_cache = {}

        # --- 6.b 批量预计算 Local-Plane Contact IK（Adam 优化按窗口并行） ---
        # 仅在 depth_align_mode=none 且 skating 不改 body_pose（pp_static_joint/none）时安全，
        # 此时各窗口 IK 输入（pose / pp_static transl / 原始中心深度）都在循环前已确定。
        contact_ik_windows: Dict[int, Any] = {}
        contact_ik_batch_stats = {'enabled': False}
        _ik_align_mode = str(self.config.get('contact_ik_depth_align_mode', 'none')).lower()
        ik_batch_ok = (
            bool(self.config.get('contact_ik_use_batch', True))
            and bool(self.config.get('enable_depth_contact_ik', True))
            and contact_scene_model in ('local_planes', 'josh_planes', 'rich_floor', 'gt_floor', 'rich_gt_floor')
            and _ik_align_mode in ('none', 'false', 'off')
            and skating_mode in ('pp_static_joint', 'none')
        )
        if ik_batch_ok:
            import time as _time_ik
            from collections import defaultdict as _dd
            t_ik = _time_ik.perf_counter()
            ik_preps: Dict[int, Dict[str, Any]] = {}
            for spec in window_specs:
                c, s, e = spec['center'], spec['start'], spec['end']
                if skating_mode == 'pp_static_joint':
                    win_tr = pp_static_joint_windows.get(c, global_trans[s:e]).clone()
                else:
                    win_tr = global_trans[s:e].clone()
                dmap = center_depth_cache.get(c, None)
                if not contact_uses_gt_floor and dmap is None:
                    continue  # 深度缺失 → 交给循环内 per-window 路径回退
                hmask = None
                if dmap is not None:
                    hmask = self._get_human_mask_for_depth(data.masks, c, tuple(dmap.shape))
                try:
                    prep = self._prepare_local_plane_ik_window(
                        global_orient_w, body_pose_aa[s:e].clone(), betas, win_tr,
                        static_conf_logits, dmap, R_cw, t_cw, K_for_depth, c, s, e, hmask,
                    )
                except Exception as ex:
                    self.logger.warning(f"Window {c} contact IK prep failed: {ex}")
                    prep = None
                if prep is None:
                    contact_ik_windows[c] = None  # 无有效 contact → 保持 pose 不变
                else:
                    ik_preps[c] = prep

            grouped_ik = _dd(list)
            for c, prep in ik_preps.items():
                grouped_ik[prep['win_len']].append((c, prep))
            ik_batch_size = max(1, int(self.config.get('contact_ik_batch_size', 16)))
            ik_calls = 0
            for _, items in grouped_ik.items():
                for cs in range(0, len(items), ik_batch_size):
                    chunk = items[cs:cs + ik_batch_size]
                    preps = [p for _, p in chunk]
                    try:
                        outs = self._apply_local_plane_contact_ik_batch(preps)
                    except Exception as ex:
                        self.logger.warning(f"Contact IK batch failed ({ex}); those windows fall back to per-window")
                        outs = None
                    if outs is None:
                        continue
                    for (c, _), out in zip(chunk, outs):
                        contact_ik_windows[c] = out
                    ik_calls += 1

            contact_ik_batch_stats = {
                'enabled': True,
                'num_windows': len(ik_preps),
                'batch_calls': ik_calls,
                'elapsed_sec': _time_ik.perf_counter() - t_ik,
            }
            self.logger.info(
                f"Batch contact IK prepared {len(ik_preps)} windows "
                f"in {contact_ik_batch_stats['elapsed_sec']:.3f}s ({ik_calls} batch calls, "
                f"batch_size={ik_batch_size})"
            )

        num_windows = 0
        for spec in window_specs:
            center = spec['center']
            start = spec['start']
            end = spec['end']
            win_len = spec['win_len']

            # ---- Step A: 创建窗口独立 transl 副本 ----
            win_transl = global_trans[start:end].clone()  # (win_len, 3) 独立副本
            win_body_pose = body_pose_aa[start:end].clone()  # (win_len, 63) 独立副本

            window_canvas_pre = None
            window_canvas_post = None
            contact_ik_canvas_pre = None
            contact_ik_canvas_post = None
            metrics_pre = None
            metrics_post = None

            # ---- Step B: 渲染 PRE-refine 的 5 帧 SMPL (debug) ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    window_canvas_pre = self._render_window_smpl(
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,  # 用窗口的 body_pose（此时还是原始的）
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K_vis,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='pre',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                        static_conf_logits=static_conf_logits,
                        save=False,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} pre-render failed: {e}")

            # ---- Step B2: PRE-render 后按配置做 pp_static_joint 纠正滑步 ----
            win_orient = global_orient_w[start:end]
            win_betas = betas[start:end]
            win_static = static_conf_logits[start:end] if static_conf_logits is not None else None
            center_local = center - start
            if smplx_model is not None and smplx2smpl is not None:
                try:
                    metrics_pre = self._compute_window_skating_metrics(
                        win_orient, win_body_pose, win_betas, win_transl, win_static,
                        smplx_model, smplx2smpl,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} pre skating metrics failed: {e}")
            if skating_mode == 'pp_static_joint':
                if center in pp_static_joint_windows:
                    win_transl = pp_static_joint_windows[center].clone()
                else:
                    win_transl = self._apply_pp_static_joint(
                        win_orient, win_body_pose, win_betas, win_transl, win_static,
                        center_idx=center_local,
                    )
                if smplx_model is not None and smplx2smpl is not None:
                    try:
                        metrics_post = self._compute_window_skating_metrics(
                            win_orient, win_body_pose, win_betas, win_transl, win_static,
                            smplx_model, smplx2smpl,
                        )
                    except Exception as e:
                        self.logger.warning(f"Window {center} post skating metrics failed: {e}")
            elif skating_mode == 'none':
                self.logger.debug(f"Window {center}: skip pp_static_joint (window_skating_mode=none)")

            # ---- Step C: 中心帧深度图推理（按需，相机坐标系） ----
            center_depth_map = None
            center_human_mask = None
            if need_depth_maps and self.metric3d_model is not None:
                try:
                    if center in center_depth_cache:
                        center_depth_map = center_depth_cache[center]
                    else:
                        center_depth_map = self._infer_single_depth(
                            data.image_paths[center], intrinsics_np,
                        )
                    align_stats = None
                    if str(self.config.get('contact_ik_depth_align_mode', 'none')).lower() not in ('none', 'false', 'off'):
                        center_depth_map, align_stats = self._align_depth_to_current_human(
                            center_depth_map,
                            K_for_depth,
                            center,
                            center_local,
                            global_orient_w,
                            win_body_pose,
                            betas,
                            win_transl,
                            R_cw,
                            t_cw,
                        )
                        if align_stats and align_stats.get('enabled', False):
                            self.logger.debug(
                                f"[DepthAlign] Window {center}: mode={align_stats.get('mode')} "
                                f"scene_to_smpl={align_stats.get('scene_to_smpl_scale', 1.0):.4f} "
                                f"joints={align_stats.get('num_joints', 0)}"
                            )
                    center_human_mask = self._get_human_mask_for_depth(
                        data.masks, center, tuple(center_depth_map.shape)
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} depth inference failed: {e}")

            # ---- Step C2: Depth Contact Debug 可视化 ----
            if center_depth_map is not None and self.config.get('enable_window_visualization', True):
                try:
                    self._vis_depth_contact_debug(
                        center=center,
                        start=start,
                        end=end,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        static_conf_logits=static_conf_logits,
                        depth_map=center_depth_map,
                        R_cw=R_cw, t_cw=t_cw, K=K_for_depth,
                        image_path=data.image_paths[center],
                        output_dir=vis_output_dir,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} depth contact debug failed: {e}")

            # ---- Step E: 渲染 pp_static_joint 后的窗口，并保存 PRE/POST 左右拼接图 ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    window_canvas_post = self._render_window_smpl( #会在这里进行w2c变换
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K_vis,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='post',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                        static_conf_logits=static_conf_logits,
                        ik_joint_ids=contact_ik_opt_joints,
                        contact_joint_ids=contact_ik_target_joints,
                        marker_note=contact_ik_marker_note,
                        save=False,
                    )
                    if self.config.get('vis_contact_ik_compare', True):
                        contact_ik_canvas_pre = window_canvas_post.copy()
                    self._save_window_concat(
                        center, window_canvas_pre, window_canvas_post, vis_output_dir,
                        metrics_pre=metrics_pre, metrics_post=metrics_post,
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} post-render failed: {e}")

            # ---- Step C3: Depth Contact IK (基于深度穿透信号优化 body_pose) ----
            if self.config.get('enable_depth_contact_ik', True) and (center_depth_map is not None or contact_uses_gt_floor):
                try:
                    # 优先使用循环前批量预计算的 Contact IK 结果（GPU 并行）；
                    # None 表示该窗口无有效 contact，pose 保持不变；未命中则回退 per-window。
                    _use_cached_ik = center in contact_ik_windows
                    contact_refined = contact_ik_windows[center] if _use_cached_ik else self._apply_depth_contact_ik(
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        static_conf_logits=static_conf_logits,
                        depth_map=center_depth_map,
                        R_cw=R_cw, t_cw=t_cw, K=K_for_depth,
                        center=center,
                        start=start,
                        end=end,
                        human_mask=center_human_mask,
                    )
                    if contact_refined is not None:
                        contact_bp_before = win_body_pose.clone()
                        contact_tr_before = win_transl.clone()
                        if isinstance(contact_refined, tuple):
                            refined_bp, refined_tr = contact_refined
                        else:
                            refined_bp, refined_tr = contact_refined, win_transl
                        contact_blend = max(0.0, min(1.0, float(self.config.get('contact_ik_blend', 1.0))))
                        win_body_pose = (1.0 - contact_blend) * contact_bp_before + contact_blend * refined_bp.to(contact_bp_before.device)
                        win_transl = (1.0 - contact_blend) * contact_tr_before + contact_blend * refined_tr.to(contact_tr_before.device)
                except Exception as e:
                    self.logger.warning(f"Window {center} depth contact IK failed: {e}")

            # ---- Step E2: 若 depth/contact IK 改动了 body_pose，再更新同一张 PRE/POST 拼接图 ----
            if smplx_model is not None and p3d_renderer is not None:
                try:
                    window_canvas_post = self._render_window_smpl( #会在这里进行w2c变换
                        center=center,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        win_start=start,
                        win_end=end,
                        R_cw=R_cw, t_cw=t_cw, K=K_vis,
                        image_paths=data.image_paths,
                        output_dir=vis_output_dir,
                        tag='post',
                        render_offsets=render_offsets,
                        smplx_model=smplx_model,
                        smplx2smpl=smplx2smpl,
                        faces_smpl=faces_smpl,
                        renderer=p3d_renderer,
                        static_conf_logits=static_conf_logits,
                        ik_joint_ids=contact_ik_opt_joints,
                        contact_joint_ids=contact_ik_target_joints,
                        marker_note=contact_ik_marker_note,
                        save=False,
                    )
                    if self.config.get('vis_contact_ik_compare', True):
                        contact_ik_canvas_post = window_canvas_post.copy()
                    self._save_window_concat(
                        center, window_canvas_pre, window_canvas_post, vis_output_dir,
                        metrics_pre=metrics_pre, metrics_post=metrics_post,
                    )
                    self._save_window_concat(
                        center, contact_ik_canvas_pre, contact_ik_canvas_post, vis_output_dir,
                        metrics_pre=metrics_post, metrics_post=metrics_post,
                        subdir='oar_contact_ik_compare',
                        left_label='BEFORE Contact IK (after pp_static_joint)',
                        right_label='AFTER Contact IK',
                    )
                except Exception as e:
                    self.logger.warning(f"Window {center} post-render failed: {e}")

            # ---- Step F: 保存窗口结果 ----
            mesh_observation = None
            if self.config.get('save_window_mesh_observations', True):
                try:
                    mesh_observation = self._build_window_mesh_observation(
                        center=center,
                        start=start,
                        end=end,
                        global_orient_w=global_orient_w,
                        body_pose_aa=win_body_pose,
                        betas=betas,
                        win_transl=win_transl,
                        R_cw=R_cw,
                        t_cw=t_cw,
                        static_conf_logits=static_conf_logits,
                    )
                    window_mesh_observations.append(mesh_observation)
                except Exception as e:
                    self.logger.warning(f"Window {center} mesh observation export failed: {e}")

            window_record = {
                'center': center,
                'start': start,
                'end': end,
                'transl': win_transl.clone(),        # (win_len, 3) 独立副本
                'body_pose': win_body_pose.clone(),   # (win_len, 63)
                'skating_metrics_pre': metrics_pre,
                'skating_metrics_post': metrics_post,
            }
            if mesh_observation is not None:
                window_record['mesh_observation'] = mesh_observation
            window_results.append(window_record)

            num_windows += 1
            if num_windows % 10 == 0:
                self.logger.info(f"Processed {num_windows} windows (center={center})")

        self.logger.info(f"Processed {num_windows} windows total")

        # --- 7. 释放 Metric3D + 渲染模型 ---
        self._unload_metric3d()
        if smplx_model is not None:
            del smplx_model, smplx2smpl, p3d_renderer
            torch.cuda.empty_cache()

        # --- 8. 全局修正（高斯加权合并 window_results）写回 data 供 eval ---
        # 用 window_results 中每个窗口的 transl 和 body_pose 进行高斯距离加权合并
        acc_transl = torch.zeros(F_total, 3)
        acc_body_pose = torch.zeros(F_total, 63)
        acc_t_weights = torch.zeros(F_total)
        acc_bp_weights = torch.zeros(F_total)

        for wr in window_results:
            w_center, w_start, w_end = wr['center'], wr['start'], wr['end']
            w_transl = wr['transl']       # (win_len, 3)
            w_bp = wr['body_pose']         # (win_len, 63)
            w_len = w_end - w_start

            # 高斯权重：基于帧到中心帧的距离
            frame_indices = torch.arange(w_start, w_end).float()
            offsets = frame_indices - float(w_center)
            w_gauss = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
            w_gauss = w_gauss / w_gauss.max()  # 中心帧权重=1

            acc_transl[w_start:w_end] += w_gauss.unsqueeze(-1) * w_transl
            acc_t_weights[w_start:w_end] += w_gauss

            acc_body_pose[w_start:w_end] += w_gauss.unsqueeze(-1) * w_bp
            acc_bp_weights[w_start:w_end] += w_gauss

        # 归一化 transl
        t_valid = acc_t_weights > 1e-6
        global_trans_new = global_trans.clone()
        global_trans_new[t_valid] = acc_transl[t_valid] / acc_t_weights[t_valid].unsqueeze(-1)

        # 归一化 body_pose
        bp_valid = acc_bp_weights > 1e-6
        body_pose_new = body_pose_aa.clone()
        body_pose_new[bp_valid] = acc_body_pose[bp_valid] / acc_bp_weights[bp_valid].unsqueeze(-1)

        self.logger.info(
            f"OAR global merge - transl delta: mean={( global_trans_new - global_trans).abs().mean():.4f}, "
            f"max={(global_trans_new - global_trans).abs().max():.4f}, "
            f"body_pose delta: mean={(body_pose_new - body_pose_aa).abs().mean():.4f}, "
            f"max={(body_pose_new - body_pose_aa).abs().max():.4f}"
        )

        # --- 9. Image model 3D body pose refinement (保留但可关闭) ---
        phmr_img_smpl = data.metadata.get('phmr_img_smpl', None)
        if (self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None):
            self.logger.info("Running image-model 3D body pose refinement (body-local space)...")
            img_j3d_local = self._prepare_img_model_joints(phmr_img_smpl, betas)
            body_pose_new = self._refine_body_pose_with_img_model(
                body_pose_new, betas, img_j3d_local,
            )
        elif self.config['enable_img_pose_refine'] and phmr_img_smpl is None:
            self.logger.info("Image-model body pose refinement enabled but no phmr_img_smpl, skipping")

        # --- 11. Update data ---
        data.smpl_params.global_orient_w = global_orient_w
        data.smpl_params.trans = trans_c
        data.smpl_params.global_trans = global_trans_new
        data.smpl_params.transl_w_raw = global_trans_new.clone()
        data.smpl_params.body_pose_aa = body_pose_new

        # Store window results in metadata for downstream (adjacent_render, SLAM)
        data.metadata['oar_window_results'] = window_results
        data.metadata['oar_window_mesh_observations'] = window_mesh_observations

        data.metadata['oar_refine'] = {
            'window_size': window_size,
            'stride': stride,
            'opt_steps': self.config['opt_steps'],
            'delta_transl_mean': (global_trans_new - global_trans).abs().mean().item(),
            'num_windows': num_windows,
            'enable_depth_constraint': self.config['enable_depth_constraint'],
            'enable_ik': self.config['enable_ik'],
            'enable_orient_refine': self.config['enable_orient_refine'],
            'enable_img_pose_refine': self.config['enable_img_pose_refine'],
            'transl_2d_refine': transl_2d_stats,
            'pp_static_joint_batch': pp_static_joint_batch_stats,
            'img_pose_refined': (
                self.config['enable_img_pose_refine']
                and phmr_img_smpl is not None
            ),
        }

        if self.config.get('vis_oar_diagnostics', False):
            # --- 12. Visualisation (before/after comparison) ---
            self._save_before_after_vis(data, pre_params, vis_output_dir)

            # --- 12b. 最终重投影可视化（所有窗口+IK处理完的最终结果） ---
            self.logger.info("Generating final reprojection visualization...")
            try:
                with torch.no_grad():
                    joints_world_final = self.endecoder.fk_v2(
                        body_pose=body_pose_new.unsqueeze(0).to(self.device),
                        betas=betas.unsqueeze(0).to(self.device),
                        global_orient=global_orient_w.unsqueeze(0).to(self.device),
                        transl=global_trans_new.unsqueeze(0).to(self.device),
                    )[0].cpu()

                proj_2d_pre = self._compute_per_frame_projections(joints_world, R_cw, t_cw, K_vis)
                proj_2d_final = self._compute_per_frame_projections(joints_world_final, R_cw, t_cw, K_vis)

                vis_stride = self.config.get('vis_oar_frame_stride', 5)
                if kp2d_vis is not None:
                    self._vis_reprojection_error(
                        kp2d_vis, proj_2d_pre, proj_2d_final,
                        data.image_paths, vis_output_dir,
                        frame_stride=vis_stride,
                    )
                    self._vis_keypoints_2d(
                        kp2d_vis, proj_2d_final, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='final',
                    )
            except Exception as e:
                self.logger.warning(f"Final reprojection visualization failed: {e}")

        # --- 12c. OAR 诊断可视化 ---
        if self.config.get('vis_oar_diagnostics', False):
            self.logger.info("Generating OAR diagnostic visualizations...")
            vis_stride = self.config.get('vis_oar_frame_stride', 5)

            proj_2d_pre = self._compute_per_frame_projections(joints_world, R_cw, t_cw, K_vis)
            _, joints_cam_pre = self._project_joints(joints_world, R_cw, t_cw, K_vis)

            with torch.no_grad():
                joints_world_post = self.endecoder.fk_v2(
                    body_pose=body_pose_new.unsqueeze(0).to(self.device),
                    betas=betas.unsqueeze(0).to(self.device),
                    global_orient=global_orient_w.unsqueeze(0).to(self.device),
                    transl=global_trans_new.unsqueeze(0).to(self.device),
                )
                joints_world_post = joints_world_post[0].cpu()

            proj_2d_post = self._compute_per_frame_projections(
                joints_world_post, R_cw, t_cw, K_vis
            )

            try:
                if kp2d_vis is not None:
                    self._vis_keypoints_2d(
                        kp2d_vis, proj_2d_pre, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='pre',
                    )
                    self._vis_keypoints_2d(
                        kp2d_vis, proj_2d_post, data.image_paths,
                        vis_output_dir, frame_stride=vis_stride, tag='post',
                    )
                    self._vis_reprojection_error(
                        kp2d_vis, proj_2d_pre, proj_2d_post,
                        data.image_paths, vis_output_dir,
                        frame_stride=vis_stride,
                    )
            except Exception as e:
                self.logger.warning(f"OAR diagnostic visualization failed: {e}")

        self.logger.info("OAR refinement complete")
        return data

    def cleanup(self):
        self._unload_metric3d()
        self.endecoder = None
        self._contact_smplx_model = None
        self._contact_smplx2smpl = None
        super().cleanup()

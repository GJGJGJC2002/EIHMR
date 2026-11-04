"""EvaluationComponent - 评估组件"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from lib.pipeline.core.component import Component
from lib.pipeline.core.data import PipelineData, CameraParams, SMPLParams


class EvaluationComponent(Component):
    """
    评估组件
    
    计算各种评估指标：
    - 局部运动: PA-MPJPE, MPJPE, PVE, Accel
    - 全局运动: W-MPJPE, WA-MPJPE, RTE, ERVE, Jitter, Foot-Sliding
    - 相机运动: ATE
    
    Config:
        metrics: 要计算的指标列表
        chunk_length: 全局运动评估的分块长度
        fps: 视频帧率（用于加速度计算）
    """
    
    COMPONENT_TYPE = "evaluation"
    
    ALL_METRICS = [
        'pa_mpjpe', 'mpjpe', 'pve', 'accel',  # 局部运动
        'w_mpjpe', 'wa_mpjpe', 'rte', 'erve', 'jitter', 'fs',  # 全局运动
        'foot_floating_rate', 'ground_penetration',  # 物理合理性 proxy
        'ate', 'ate_s'  # 相机运动
    ]
    
    DEFAULT_CONFIG = {
        'metrics': ALL_METRICS,
        'chunk_length': 100,
        'fps': 30,
        'frame_range': None,  # Optional [start, end) frame range for focused experiments
        'physical_foot_vertex_ids': [3216, 3387, 6617, 6787],
        'physical_ground_axis': 1,
        'physical_ground_quantile': 0.01,
        'physical_ground_mode': 'auto',  # auto uses RICH floor_y when dataset_type=rich, otherwise GT-foot quantile
        'physical_ground_height': 0.0,
        'physical_contact_vel_thr': 0.01,
        'physical_contact_height_thr': 0.05,
        'physical_floating_thr': 0.05,
        'physical_penetration_tol': 0.0,
    }
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)
        
        self.metrics_to_compute = self.config['metrics']
        self.chunk_length = self.config['chunk_length']
        self.fps = self.config['fps']
        
        # SMPL 模型（延迟加载）
        self._smpls = None
    
    def setup(self):
        """初始化 SMPL 模型"""
        from lib.models.smpl import SMPL
        
        self._smpls = {
            gender: SMPL(gender=gender) 
            for gender in ['neutral', 'male', 'female']
        }
        
        # GVHMR SMPL-X 模型和转换矩阵（延迟加载，首次 GVHMR 评估时初始化）
        self._smplx_model = None
        self._smplx2smpl = None
        self._J_regressor = None
        
        # 3DPW 评估用 J_regressor_h36m（延迟加载）
        self._J_regressor_h36m_14 = None
        # RICH 评估用 PromptHMR SMPL-X/SMPL 与 smplx2smpl（延迟加载）
        self._rich_smplx = None
        self._rich_smpl = None
        self._rich_smplx2smpl = None
        
        self._is_setup = True
        self.logger.info("Evaluation component initialized")
    
    def _ensure_smplx_model(self, data: 'PipelineData'):
        """延迟加载 GVHMR 的 SMPL-X 模型和转换矩阵"""
        if self._smplx_model is not None:
            return
        
        from lib.gvhmr import ASSETS
        from lib.gvhmr.body_model import make_smplx

        self._smplx_model = make_smplx("supermotion")
        self._smplx_model.eval()
        self._smplx2smpl = torch.load(ASSETS / "smplx2smpl_sparse.pt", weights_only=True)
        self._J_regressor = torch.load(ASSETS / "smpl_neutral_J_regressor.pt", weights_only=True)
        self.logger.info("Loaded GVHMR SMPL-X model + smplx2smpl + J_regressor for evaluation")
    
    def _ensure_rich_models(self, data: 'PipelineData'):
        """延迟加载 RICH 官方评估所需的 PromptHMR SMPL-X/SMPL 资源。"""
        if self._rich_smplx is not None:
            return
        import sys, os
        import pickle as pkl

        prompthmr_root = data.metadata.get('prompthmr_root', '../PromptHMR')
        prompthmr_abs = os.path.abspath(prompthmr_root)
        if prompthmr_abs in sys.path:
            sys.path.remove(prompthmr_abs)
        sys.path.insert(0, prompthmr_abs)
        # tram 根目录也有 data_config.py，清掉缓存以确保导入 PromptHMR/data_config.py。
        sys.modules.pop('data_config', None)
        old_cwd = os.getcwd()
        os.chdir(prompthmr_abs)
        try:
            from data_config import SMPL_PATH, SMPLX_PATH, SMPLX2SMPL
            from prompt_hmr.smpl_family import SMPL, SMPLX
            self._rich_smplx = {
                gender: SMPLX(SMPLX_PATH, gender=gender)
                for gender in ['neutral', 'male', 'female']
            }
            self._rich_smpl = SMPL(SMPL_PATH, gender='neutral')
            self._rich_smplx2smpl = torch.from_numpy(pkl.load(open(SMPLX2SMPL, 'rb'))['matrix']).float()
        finally:
            os.chdir(old_cwd)
        self.logger.info("Loaded PromptHMR SMPL-X/SMPL resources for RICH evaluation")

    def _ensure_j_regressor_h36m(self):
        """延迟加载 J_regressor_h36m 的 14 关节子集（用于 3DPW 评估）"""
        if self._J_regressor_h36m_14 is not None:
            return
        
        import os
        from lib.core.constants import H36M_TO_J14
        
        j_reg_path = os.path.join('data', 'smpl', 'J_regressor_h36m.npy')
        if not os.path.exists(j_reg_path):
            raise FileNotFoundError(
                f"J_regressor_h36m.npy not found at {j_reg_path}. "
                "Required for 3DPW evaluation."
            )
        
        J_regressor_h36m = np.load(j_reg_path)  # (17, 6890)
        J_regressor_h36m_14 = J_regressor_h36m[H36M_TO_J14]  # (14, 6890)
        self._J_regressor_h36m_14 = torch.from_numpy(J_regressor_h36m_14).float()
        self.logger.info(
            f"Loaded J_regressor_h36m[H36M_TO_J14] for 3DPW evaluation: "
            f"shape={self._J_regressor_h36m_14.shape}"
        )
    
    def validate_input(self, data: PipelineData) -> bool:
        """验证输入：需要预测结果和 GT 标注"""
        has_predictions = data.smpl_params is not None
        has_gt = data.annotations is not None
        return has_predictions and has_gt
    
    def execute(self, data: PipelineData) -> PipelineData:
        """执行评估"""
        self.logger.info(f"Computing metrics: {self.metrics_to_compute}")
        
        dataset_type = data.metadata.get('dataset_type')
        is_3dpw = dataset_type == '3dpw'
        is_rich = dataset_type == 'rich'
        
        # 加载 GT 数据
        if is_3dpw:
            gt_data = self._load_gt_data_3dpw(data)
        elif is_rich:
            gt_data = self._load_gt_data_rich(data)
        else:
            gt_data = self._load_gt_data(data)
        
        # 计算预测的 SMPL 输出
        pred_data = self._compute_pred_smpl(data)
        
        # 3DPW: 将 pred joints/verts 也用 J_regressor_h36m_14 回归
        if is_3dpw:
            self._ensure_j_regressor_h36m()
            j_reg = self._J_regressor_h36m_14
            pred_data['pred_j3d'] = torch.matmul(j_reg, pred_data['pred_vert'])
            pred_data['pred_j3d_w'] = torch.matmul(j_reg, pred_data['pred_vert_w'])
        
        # 应用 valid mask，并可选叠加实验帧范围 [start, end)
        valid_mask = data.valid_frames_mask if data.valid_frames_mask is not None else gt_data.get('valid_mask')
        frame_range = self.config.get('frame_range', None)
        if frame_range is not None:
            n_frames = next(
                (v.shape[0] for v in list(gt_data.values()) + list(pred_data.values())
                 if hasattr(v, 'shape') and len(v.shape) > 0),
                None,
            )
            if n_frames is not None:
                start = max(0, int(frame_range[0]))
                end = min(int(frame_range[1]), int(n_frames))
                range_mask = np.zeros(n_frames, dtype=bool)
                if start < end:
                    range_mask[start:end] = True
                if valid_mask is None:
                    valid_mask = range_mask
                else:
                    valid_mask = np.asarray(valid_mask).astype(bool) & range_mask
                self.logger.info(f"Evaluation frame_range=[{start}, {end})")
        if valid_mask is not None:
            gt_data, pred_data = self._apply_valid_mask(gt_data, pred_data, valid_mask)
        gt_data, pred_data = self._align_lengths_after_mask(gt_data, pred_data)
        
        metrics = {}
        m2mm = 1e3  # 米转毫米
        
        # 3DPW 使用 pelvis_idxs=[2,3]（与 WHAM/GVHMR 一致）；EMDB/RICH 使用 [1,2]
        pelvis_idxs = [2, 3] if is_3dpw else [1, 2]
        
        # === 局部运动评估 ===
        # 会在内部对 pred_j3d/gt_j3d_cam 做 batch_align_by_pelvis 并写回 pred_data/gt_data
        local_aligned = False
        if any(m in self.metrics_to_compute for m in ['pa_mpjpe', 'mpjpe', 'pve']):
            local_metrics = self._evaluate_local_motion(
                gt_data, pred_data, m2mm, pelvis_idxs=pelvis_idxs)
            metrics.update(local_metrics)
            local_aligned = True

        if 'accel' in self.metrics_to_compute:
            # 确保 accel 用 pelvis-aligned 数据（与官方 run_eval.py 一致）
            # 否则 root 在 camera-space 的运动会混入加速度，accel 被严重高估
            if not local_aligned:
                from lib.utils.eval_utils import batch_align_by_pelvis
                pred_j3d_a, gt_j3d_cam_a = batch_align_by_pelvis(
                    [pred_data['pred_j3d'], gt_data['gt_j3d_cam']],
                    pelvis_idxs=pelvis_idxs
                )
            else:
                pred_j3d_a = pred_data['pred_j3d']
                gt_j3d_cam_a = gt_data['gt_j3d_cam']
            accel = self._compute_acceleration_error(
                pred_j3d_a,
                gt_j3d_cam_a
            ) * (self.fps ** 2)
            accel_flat = np.asarray(accel).reshape(-1)
            metrics['accel'] = float(accel_flat.mean())
            # 额外保存 sum / count 供跨 seq 按帧加权聚合
            metrics['_sum_accel'] = float(accel_flat.sum())
            metrics['_cnt_accel'] = int(accel_flat.size)

        # === 全局运动评估 ===
        if any(m in self.metrics_to_compute for m in ['w_mpjpe', 'wa_mpjpe']):
            global_metrics = self._evaluate_global_motion(
                gt_data['gt_j3d'], 
                pred_data['pred_j3d_w'],
                m2mm
            )
            metrics.update(global_metrics)
        
        if 'rte' in self.metrics_to_compute:
            rte = self._compute_rte(
                gt_data['gt_j3d'][:, 0], 
                pred_data['pred_j3d_w'][:, 0]
            ) * 1e2  # 米转厘米
            rte_flat = np.asarray(rte).reshape(-1)
            metrics['rte'] = float(rte_flat.mean())
            metrics['_sum_rte'] = float(rte_flat.sum())
            metrics['_cnt_rte'] = int(rte_flat.size)

        if 'erve' in self.metrics_to_compute:
            erve = self._compute_erve(
                gt_data['gt_ori'],
                gt_data['gt_j3d'],
                pred_data['pred_ori_w'],
                pred_data['pred_j3d_w']
            ) * m2mm
            erve_flat = np.asarray(erve).reshape(-1)
            metrics['erve'] = float(erve_flat.mean())
            metrics['_sum_erve'] = float(erve_flat.sum())
            metrics['_cnt_erve'] = int(erve_flat.size)

        if 'jitter' in self.metrics_to_compute:
            jitter = self._compute_jitter(pred_data['pred_j3d_w'])
            jitter_flat = np.asarray(jitter).reshape(-1)
            metrics['jitter'] = float(jitter_flat.mean()) if jitter_flat.size > 0 else float('nan')
            metrics['_sum_jitter'] = float(jitter_flat.sum())
            metrics['_cnt_jitter'] = int(jitter_flat.size)

        if 'fs' in self.metrics_to_compute:
            foot_sliding = self._compute_foot_sliding(
                gt_data['gt_vert'],
                pred_data['pred_vert_w']
            ) * m2mm
            fs_flat = np.asarray(foot_sliding).reshape(-1)
            metrics['fs'] = float(fs_flat.mean()) if fs_flat.size > 0 else float('nan')
            metrics['_sum_fs'] = float(fs_flat.sum())
            metrics['_cnt_fs'] = int(fs_flat.size)

        if any(m in self.metrics_to_compute for m in ['foot_floating_rate', 'ground_penetration']):
            physical = self._compute_foot_physical_metrics(
                gt_data['gt_vert'],
                pred_data['pred_vert_w'],
                is_rich=is_rich,
            )
            if 'foot_floating_rate' in self.metrics_to_compute:
                floating = np.asarray(physical['foot_floating_rate']).reshape(-1)
                metrics['foot_floating_rate'] = float(floating.mean()) if floating.size > 0 else float('nan')
                metrics['_sum_foot_floating_rate'] = float(floating.sum())
                metrics['_cnt_foot_floating_rate'] = int(floating.size)
            if 'ground_penetration' in self.metrics_to_compute:
                penetration = np.asarray(physical['ground_penetration']).reshape(-1) * m2mm
                metrics['ground_penetration'] = float(penetration.mean()) if penetration.size > 0 else float('nan')
                metrics['_sum_ground_penetration'] = float(penetration.sum())
                metrics['_cnt_ground_penetration'] = int(penetration.size)
        
        # === 相机运动评估 ===
        if 'ate' in self.metrics_to_compute or 'ate_s' in self.metrics_to_compute:
            cam_metrics = self._evaluate_camera_motion(data, gt_data, pred_data)
            metrics.update(cam_metrics)
        
        # 保存结果
        data.metrics = metrics
        
        # self.logger.info("Evaluation results:")
        # for k, v in metrics.items():
        #     self.logger.info(f"  {k}: {v:.4f}")
        
        return data
    
    def _load_gt_data(self, data: PipelineData) -> Dict[str, Any]:
        """加载 GT 数据"""
        from lib.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

        ann = data.annotations

        # 检查是否有帧采样信息
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
            self.logger.info(
                f"Applying frame sampling to GT: {len(ann['smpl']['poses_body'])} -> {len(sampled_indices)} frames"
            )
        else:
            sampled_indices = None

        gender = ann['gender']
        poses_body = ann["smpl"]["poses_body"]
        poses_root = ann["smpl"]["poses_root"]
        betas = np.repeat(
            ann["smpl"]["betas"].reshape((1, -1)),
            repeats=ann["n_frames"],
            axis=0
        )
        trans = ann["smpl"]["trans"]
        ext = ann['camera']['extrinsics']

        # 应用采样
        if sampled_indices is not None:
            poses_body = poses_body[sampled_indices]
            poses_root = poses_root[sampled_indices]
            betas = betas[sampled_indices]
            trans = trans[sampled_indices]
            ext = ext[sampled_indices]

        tt = lambda x: torch.from_numpy(x).float()

        # 判断是否是 GVHMR 路径（Pred 用 SMPL-X + smplx2smpl + 外部 J_regressor）
        smpl = data.smpl_params
        is_gvhmr = (smpl is not None and smpl.rotmat is None and 
                    getattr(smpl, 'global_orient_c', None) is not None)

        # 判断是否是 PromptHMR 图像模型路径
        is_phmr_imgonly = data.metadata.get('hpe_backend') == 'prompthmr_imgonly'

        # 世界坐标系下的 GT
        gt = self._smpls[gender](
            body_pose=tt(poses_body),
            global_orient=tt(poses_root),
            betas=tt(betas),
            transl=tt(trans),
            pose2rot=True,
            default_smpl=True
        )

        # 相机坐标系下的 GT
        poses_root_cam = matrix_to_axis_angle(
            tt(ext[:, :3, :3]) @ axis_angle_to_matrix(tt(poses_root))
        )
        gt_cam = self._smpls[gender](
            body_pose=tt(poses_body),
            global_orient=poses_root_cam,
            betas=tt(betas),
            pose2rot=True,
            default_smpl=True
        )

        if is_phmr_imgonly:
            # PromptHMR 图像模型路径：用 smpl.J_regressor[:24] 回归 joints（与官方 evaluator.py 一致）
            phmr_j_reg = data.metadata.get('phmr_j_regressor')
            if phmr_j_reg is not None:
                gt_j3d = torch.matmul(phmr_j_reg, gt.vertices)
                gt_j3d_cam = torch.matmul(phmr_j_reg, gt_cam.vertices)
                self.logger.info("PromptHMR img-only path: Using smpl.J_regressor[:24] for GT joints")
            else:
                gt_j3d = gt.joints[:, :24]
                gt_j3d_cam = gt_cam.joints[:, :24]
                self.logger.info("PromptHMR img-only path: Using SMPL default joints[:24]")
        elif is_gvhmr:
            # GVHMR 路径：Pred 的 joints 是用外部 smpl_neutral_J_regressor.pt 从 vertices 回归的，
            # GT 的 joints 也必须用相同的 J_regressor，否则会有系统性偏移。
            # 这与 GVHMR 官方评估 (metric_emdb.py) 的做法一致。
            self._ensure_smplx_model(data)
            gt_j3d = torch.matmul(self._J_regressor, gt.vertices)
            gt_j3d_cam = torch.matmul(self._J_regressor, gt_cam.vertices)
            self.logger.info("GVHMR path: Using external J_regressor for GT joints (consistent with Pred)")
        else:
            gt_j3d = gt.joints[:, :24]
            gt_j3d_cam = gt_cam.joints[:, :24]

        valid_mask = ann.get('good_frames_mask')
        if sampled_indices is not None and valid_mask is not None:
            valid_mask = valid_mask[sampled_indices]

        return {
            'gender': gender,
            'gt_j3d': gt_j3d,
            'gt_vert': gt.vertices,
            'gt_ori': axis_angle_to_matrix(tt(poses_root)),
            'gt_j3d_cam': gt_j3d_cam,
            'gt_vert_cam': gt_cam.vertices,
            'ext': ext,
            'valid_mask': valid_mask,
        }
    
    def _load_gt_data_rich(self, data: PipelineData) -> Dict[str, Any]:
        """加载 RICH_TEST GT 数据，匹配 PromptHMR 官方 RICHDataset/Evaluator 口径。"""
        from lib.utils.rotation_conversions import axis_angle_to_matrix

        self._ensure_rich_models(data)
        ann = data.annotations
        rich = ann['smplx']
        rich_orig = ann.get('smplx_original_cam', rich)
        gender = ann['gender']
        if isinstance(gender, (list, tuple, np.ndarray)):
            gender = str(gender[0])
        if gender not in self._rich_smplx:
            gender = 'neutral'

        poses_root = rich['global_orient']
        poses_body = rich['body_pose']
        betas = rich['betas']
        trans = rich['trans']
        poses_root_orig = rich_orig['global_orient']
        poses_body_orig = rich_orig['body_pose']
        betas_orig = rich_orig['betas']
        trans_orig = rich_orig['trans']
        valid_mask = ann.get('good_frames_mask')

        tt = lambda x: torch.from_numpy(np.asarray(x)).float()
        smplx_model = self._rich_smplx[gender]
        with torch.no_grad():
            # Local/camera-coordinate GT follows PromptHMR official RICH evaluator
            # (after principal-point recentering).
            out = smplx_model(
                global_orient=axis_angle_to_matrix(tt(poses_root).reshape(-1, 1, 3)),
                body_pose=axis_angle_to_matrix(tt(poses_body).reshape(-1, 21, 3)),
                betas=tt(betas),
                transl=tt(trans),
            )
            gt_vert_cam = torch.stack([
                torch.matmul(self._rich_smplx2smpl, v) for v in out.vertices
            ])
            gt_j3d_cam = torch.matmul(self._rich_smpl.J_regressor, gt_vert_cam)

            # GVHMR-style global GT: physical camera-coordinate RICH SMPL-X transformed
            # by the known static T_c2ay supplied by RICH/GVHMR resources.
            out_orig = smplx_model(
                global_orient=axis_angle_to_matrix(tt(poses_root_orig).reshape(-1, 1, 3)),
                body_pose=axis_angle_to_matrix(tt(poses_body_orig).reshape(-1, 21, 3)),
                betas=tt(betas_orig),
                transl=tt(trans_orig),
            )
            gt_vert_orig_cam = torch.stack([
                torch.matmul(self._rich_smplx2smpl, v) for v in out_orig.vertices
            ])
            gt_j3d_orig_cam = torch.matmul(self._rich_smpl.J_regressor, gt_vert_orig_cam)
            T_c2ay_np = ann.get('camera', {}).get('T_c2ay')
            if T_c2ay_np is not None:
                T_c2ay = tt(T_c2ay_np)
                R_c2ay = T_c2ay[:3, :3]
                t_c2ay = T_c2ay[:3, 3]
                gt_vert_w = torch.einsum('ij,fvj->fvi', R_c2ay, gt_vert_orig_cam) + t_c2ay[None, None]
                gt_j3d_w = torch.einsum('ij,fkj->fki', R_c2ay, gt_j3d_orig_cam) + t_c2ay[None, None]
                gt_ori_w = torch.einsum(
                    'ij,fjk->fik',
                    R_c2ay,
                    axis_angle_to_matrix(tt(poses_root_orig)),
                )
            else:
                gt_vert_w = gt_vert_cam
                gt_j3d_w = gt_j3d_cam
                gt_ori_w = axis_angle_to_matrix(tt(poses_root))

        self.logger.info(
            f"[RICH] GT loaded: {gt_j3d_cam.shape[0]} frames, "
            f"{gt_j3d_cam.shape[1]} joints, gender={gender}, global_coord=ay"
        )
        return {
            'gender': gender,
            'gt_j3d': gt_j3d_w,
            'gt_vert': gt_vert_w,
            'gt_ori': gt_ori_w,
            'gt_j3d_cam': gt_j3d_cam,
            'gt_vert_cam': gt_vert_cam,
            'valid_mask': valid_mask,
        }

    def _load_gt_data_3dpw(self, data: PipelineData) -> Dict[str, Any]:
        """
        加载 3DPW GT 数据。
        
        与 EMDB 的关键差异：
        - 使用 J_regressor_h36m[H36M_TO_J14] (14 关节) 而非 SMPL joints[:24]
        - pelvis_idxs = [2, 3]（与 WHAM/GVHMR 评估一致）
        - betas 只取前 10 维
        """
        from lib.utils.rotation_conversions import axis_angle_to_matrix
        
        self._ensure_j_regressor_h36m()
        j_reg = self._J_regressor_h36m_14  # (14, 6890)
        
        ann = data.annotations
        
        # 检查帧采样
        sampling_info = data.metadata.get('frame_sampling')
        if sampling_info:
            sampled_indices = np.array(sampling_info['sampled_indices'])
            self.logger.info(
                f"[3DPW] Applying frame sampling to GT: "
                f"{ann['n_frames']} -> {len(sampled_indices)} frames"
            )
        else:
            sampled_indices = None
        
        gender = ann['gender']
        poses_body = ann['smpl']['poses_body']     # (N, 69)
        poses_root = ann['smpl']['poses_root']     # (N, 3)
        betas_raw = ann['smpl']['betas']           # (10,) or (N, 10)
        trans = ann['smpl']['trans']               # (N, 3)
        ext = ann['camera']['extrinsics']          # (N, 4, 4)
        
        n_frames = ann['n_frames']
        if np.asarray(betas_raw).ndim == 1:
            betas = np.repeat(np.asarray(betas_raw).reshape(1, -1), repeats=n_frames, axis=0)  # (N, 10)
        else:
            betas = np.asarray(betas_raw, dtype=np.float32)
        
        # 应用采样
        if sampled_indices is not None:
            poses_body = poses_body[sampled_indices]
            poses_root = poses_root[sampled_indices]
            betas = betas[sampled_indices]
            trans = trans[sampled_indices]
            ext = ext[sampled_indices]
        
        tt = lambda x: torch.from_numpy(x).float()
        
        # 世界坐标系下的 GT SMPL
        gt = self._smpls[gender](
            body_pose=tt(poses_body),
            global_orient=tt(poses_root),
            betas=tt(betas),
            transl=tt(trans),
            pose2rot=True,
            default_smpl=True,
        )
        
        # 相机坐标系下的 GT：直接对世界坐标系顶点做 T_w2c 刚体变换
        # 与 GVHMR 官方 metric_3dpw.py 的做法一致：apply_T_on_points(target_w_verts, T_w2c)
        R_w2c = tt(ext[:, :3, :3])
        t_w2c = tt(ext[:, :3, 3])
        gt_vert_cam = torch.einsum('bij,bnj->bni', R_w2c, gt.vertices) + t_w2c[:, None, :]
        
        # 用 J_regressor_h36m_14 回归 14 关节
        gt_j3d = torch.matmul(j_reg, gt.vertices)       # (N, 14, 3)
        gt_j3d_cam = torch.matmul(j_reg, gt_vert_cam)   # (N, 14, 3)
        
        self.logger.info(
            f"[3DPW] GT loaded: {gt_j3d.shape[0]} frames, "
            f"{gt_j3d.shape[1]} joints (J_regressor_h36m_14), gender={gender}"
        )
        
        valid_mask = ann.get('good_frames_mask')
        if sampled_indices is not None and valid_mask is not None:
            valid_mask = valid_mask[sampled_indices]
        
        return {
            'gender': gender,
            'gt_j3d': gt_j3d,
            'gt_vert': gt.vertices,
            'gt_ori': axis_angle_to_matrix(tt(poses_root)),
            'gt_j3d_cam': gt_j3d_cam,
            'gt_vert_cam': gt_vert_cam,
            'ext': ext,
            'valid_mask': valid_mask,
        }
    
    def _compute_pred_smpl(self, data: PipelineData) -> Dict[str, Any]:
        """计算预测的 SMPL 输出"""
        from lib.vis.traj import traj_filter
        from lib.utils.rotation_conversions import axis_angle_to_matrix
        
        smpl = data.smpl_params
        cam = data.camera_params

        # PromptHMR 图像模型路径：vertices 和 joints 已预计算
        is_phmr_imgonly = data.metadata.get('hpe_backend') == 'prompthmr_imgonly'
        if is_phmr_imgonly:
            pred_vert = smpl.vertices.float() if isinstance(smpl.vertices, torch.Tensor) else torch.from_numpy(smpl.vertices).float()
            pred_j3d = smpl.joints.float() if isinstance(smpl.joints, torch.Tensor) else torch.from_numpy(smpl.joints).float()

            # 图像模型只有 incam，世界坐标系用 identity 填充
            F_len = pred_j3d.shape[0]
            pred_vert_w = pred_vert.clone()
            pred_j3d_w = pred_j3d.clone()
            pred_ori_w = torch.eye(3).unsqueeze(0).expand(F_len, -1, -1)
            pred_camr = torch.eye(3).unsqueeze(0).expand(F_len, -1, -1)
            pred_camt = torch.zeros(F_len, 3)

            return {
                'pred_j3d': pred_j3d,
                'pred_vert': pred_vert,
                'pred_j3d_w': pred_j3d_w,
                'pred_vert_w': pred_vert_w,
                'pred_ori_w': pred_ori_w,
                'pred_camr': pred_camr,
                'pred_camt': pred_camt,
            }

        is_gvhmr = (smpl.rotmat is None and smpl.global_orient_c is not None)
        
        def to_float_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.from_numpy(np.array(x)).float()
        
        pred_shape = to_float_tensor(smpl.betas)
        pred_trans = to_float_tensor(smpl.trans)
        
        if is_gvhmr:
            # GVHMR 使用 SMPL-X (supermotion) 模型，需要用 SMPL-X forward + smplx2smpl 转换
            # 而不是直接用 SMPL forward，否则会因模型差异引入误差
            self._ensure_smplx_model(data)
            
            global_orient_c = to_float_tensor(smpl.global_orient_c)  # (F, 3)
            body_pose_aa = to_float_tensor(smpl.body_pose_aa)  # (F, 63)
            pred_len = int(global_orient_c.shape[0])
            
            # root_rotmat 用于后续 c2w 变换
            root_rotmat = axis_angle_to_matrix(global_orient_c)  # (F, 3, 3)
            
            # SMPL-X forward (camera coordinate system)
            smplx_params_incam = {
                'global_orient': global_orient_c,
                'body_pose': body_pose_aa,
                'betas': pred_shape,
                'transl': pred_trans.squeeze(),
            }
            with torch.no_grad():
                smplx_out = self._smplx_model(**smplx_params_incam)
                # SMPL-X vertices (10475) -> SMPL vertices (6890) via smplx2smpl sparse matrix
                pred_vert = torch.stack(
                    [torch.matmul(self._smplx2smpl, v) for v in smplx_out.vertices]
                )  # (F, 6890, 3)
                # SMPL vertices -> 24 joints via J_regressor
                pred_j3d = torch.matmul(self._J_regressor, pred_vert)  # (F, 24, 3)
            
            # RICH/GVHMR protocol evaluates the model's global branch in ay coordinates,
            # using the dataset-provided static camera only for inputs / local projection.
            is_rich = data.metadata.get('dataset_type') == 'rich'
            cam = data.camera_params
            use_slam_camera = (
                cam is not None 
                and getattr(cam, 'R', None) is not None 
                and getattr(cam, 'T', None) is not None
            )

            if is_rich and smpl.global_orient_w is not None and smpl.global_trans is not None:
                self.logger.info("RICH path: using PromptHMR/GVHMR global branch in ay coordinates for global metrics")
                global_orient_w = to_float_tensor(smpl.global_orient_w)
                transl_w = to_float_tensor(smpl.global_trans)
                smplx_params_world = {
                    'global_orient': global_orient_w,
                    'body_pose': body_pose_aa,
                    'betas': pred_shape,
                    'transl': transl_w.squeeze(),
                }
                with torch.no_grad():
                    smplx_out_w = self._smplx_model(**smplx_params_world)
                    pred_vert_w = torch.stack(
                        [torch.matmul(self._smplx2smpl, v) for v in smplx_out_w.vertices]
                    )
                    pred_j3d_w = torch.matmul(self._J_regressor, pred_vert_w)
                pred_ori_w = axis_angle_to_matrix(global_orient_w)
                if use_slam_camera:
                    pred_camr = to_float_tensor(cam.R)
                    pred_camt = to_float_tensor(cam.T)
                else:
                    F_len = pred_j3d.shape[0]
                    pred_camr = torch.eye(3).unsqueeze(0).expand(F_len, -1, -1)
                    pred_camt = torch.zeros(F_len, 3)
            elif use_slam_camera:
                self.logger.info(
                    "GVHMR path: Using SLAM c2w to transform incam SMPL to world (consistent with TRAM path)"
                )
                pred_camr = to_float_tensor(cam.R)
                pred_camt = to_float_tensor(cam.T)
                pred_camr, pred_camt = self._match_camera_to_prediction_length(
                    pred_camr, pred_camt, pred_len, data.valid_frames_mask
                )
                
                pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
                pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
                pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, root_rotmat)
            else:
                self.logger.info(
                    "GVHMR path: No SLAM camera available, falling back to GVHMR world params"
                )
                global_orient_w = to_float_tensor(smpl.global_orient_w)  # (F, 3)
                transl_w = to_float_tensor(smpl.global_trans)  # (F, 3) post-correction
                
                # Build world-space SMPL-X from GVHMR global params, then convert
                smplx_params_world = {
                    'global_orient': global_orient_w,
                    'body_pose': body_pose_aa,
                    'betas': pred_shape,
                    'transl': transl_w.squeeze(),
                }
                with torch.no_grad():
                    smplx_out_w = self._smplx_model(**smplx_params_world)
                    pred_vert_w = torch.stack(
                        [torch.matmul(self._smplx2smpl, v) for v in smplx_out_w.vertices]
                    )  # (F, 6890, 3)
                    pred_j3d_w = torch.matmul(self._J_regressor, pred_vert_w)  # (F, 24, 3)
                pred_ori_w = axis_angle_to_matrix(global_orient_w)  # (F, 3, 3)
                
                # Derive c2w from GVHMR's w2c for camera metrics (ATE etc.)
                try:
                    from lib.gvhmr.geometry import get_T_w2c_from_wcparams
                    skeleton_offset = to_float_tensor(smpl.skeleton_offset)
                    transl_w_raw = to_float_tensor(smpl.transl_w_raw)
                    transl_c = to_float_tensor(smpl.trans)
                    
                    T_w2c = get_T_w2c_from_wcparams(
                        global_orient_w, transl_w_raw,
                        global_orient_c, transl_c,
                        skeleton_offset,
                    )  # (F, 4, 4)
                    R_w2c = T_w2c[:, :3, :3]
                    t_w2c = T_w2c[:, :3, 3]
                    pred_camr = R_w2c.transpose(1, 2)  # R_c2w
                    pred_camt = -torch.einsum('bij,bj->bi', pred_camr, t_w2c)  # t_c2w
                except Exception as e:
                    self.logger.warning(f"Failed to compute c2w from GVHMR params: {e}, using identity")
                    F_len = pred_j3d.shape[0]
                    pred_camr = torch.eye(3).unsqueeze(0).expand(F_len, -1, -1)
                    pred_camt = torch.zeros(F_len, 3)
            
            # Trajectory filter. GVHMR RICH protocol evaluates raw global branch outputs.
            if not (data.metadata.get('dataset_type') == 'rich'):
                pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
            
        else:
            # VIMO / GT path: original logic
            pred_rotmat = to_float_tensor(smpl.rotmat)
            
            # SMPL forward
            pred = self._smpls['neutral'](
                body_pose=pred_rotmat[:, 1:], 
                global_orient=pred_rotmat[:, [0]], 
                betas=pred_shape, 
                transl=pred_trans.squeeze(),
                pose2rot=False, 
                default_smpl=True
            )
            
            pred_vert = pred.vertices
            pred_j3d = pred.joints[:, :24]
            
            # 转换到世界坐标系
            pred_camr = to_float_tensor(cam.R)
            pred_camt = to_float_tensor(cam.T)

            # If a final world-space skating-removal stage has produced
            # global_orient_w/global_trans, use those fields for world-body
            # metrics. Local camera-space metrics still use pred_vert/pred_j3d
            # above, and camera metrics still use pred_camr/pred_camt.
            if smpl.global_orient_w is not None and smpl.global_trans is not None:
                self.logger.info(
                    "Standard SMPL path: using global_orient_w/global_trans "
                    "for world-space body metrics"
                )
                global_orient_w = to_float_tensor(smpl.global_orient_w)
                global_trans = to_float_tensor(smpl.global_trans)
                body_pose_aa = to_float_tensor(getattr(smpl, 'body_pose_aa', None))
                if body_pose_aa is not None:
                    body21 = axis_angle_to_matrix(body_pose_aa.reshape(body_pose_aa.shape[0], 21, 3))
                    hand_id = torch.eye(3).reshape(1, 1, 3, 3).repeat(body21.shape[0], 2, 1, 1)
                    body23 = torch.cat([body21, hand_id], dim=1)
                else:
                    body23 = pred_rotmat[:, 1:]
                pred_w = self._smpls['neutral'](
                    body_pose=body23,
                    global_orient=axis_angle_to_matrix(global_orient_w).unsqueeze(1),
                    betas=pred_shape,
                    transl=global_trans.squeeze(),
                    pose2rot=False,
                    default_smpl=True,
                )
                pred_vert_w = pred_w.vertices
                pred_j3d_w = pred_w.joints[:, :24]
                pred_ori_w = axis_angle_to_matrix(global_orient_w)
            else:
                pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
                pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]
                pred_ori_w = torch.einsum('bij,bjk->bik', pred_camr, pred_rotmat[:, 0])
            
            # 轨迹滤波
            pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
        
        return {
            'pred_j3d': pred_j3d,
            'pred_vert': pred_vert,
            'pred_j3d_w': pred_j3d_w,
            'pred_vert_w': pred_vert_w,
            'pred_ori_w': pred_ori_w,
            'pred_camr': pred_camr,
            'pred_camt': pred_camt,
        }
    
    def _apply_valid_mask(self, gt_data, pred_data, valid_mask):
        """应用有效帧 mask；已是有效帧长度的预测保持不变。"""
        valid_mask = np.asarray(valid_mask).astype(bool)
        valid_count = int(valid_mask.sum())

        for key in ['gt_j3d', 'gt_vert', 'gt_ori', 'gt_j3d_cam', 'gt_vert_cam', 'ext']:
            if key in gt_data and gt_data[key] is not None:
                gt_data[key] = self._mask_temporal_value(gt_data[key], valid_mask, valid_count, key)
        
        for key in ['pred_j3d', 'pred_vert', 'pred_j3d_w', 'pred_vert_w', 'pred_ori_w', 'pred_camr', 'pred_camt']:
            if key in pred_data and pred_data[key] is not None:
                pred_data[key] = self._mask_temporal_value(pred_data[key], valid_mask, valid_count, key)
        
        return gt_data, pred_data

    def _mask_temporal_value(self, value, valid_mask, valid_count, name):
        """Mask temporal arrays/tensors, allowing values already filtered to valid_count."""
        if not hasattr(value, 'shape') or len(value.shape) == 0:
            return value
        length = int(value.shape[0])
        if length == len(valid_mask):
            return value[valid_mask]
        if length == valid_count:
            return value
        keep = min(length, valid_count)
        self.logger.warning(
            f"Length mismatch for {name}: len={length}, mask_len={len(valid_mask)}, "
            f"valid_count={valid_count}; truncating to {keep}."
        )
        return value[:keep]

    def _align_lengths_after_mask(self, gt_data, pred_data):
        """最后兜底：将 GT / Pred 的时序字段裁到共同长度。"""
        lengths = []
        for container, keys in [
            (gt_data, ['gt_j3d', 'gt_vert', 'gt_ori', 'gt_j3d_cam', 'gt_vert_cam', 'ext']),
            (pred_data, ['pred_j3d', 'pred_vert', 'pred_j3d_w', 'pred_vert_w', 'pred_ori_w', 'pred_camr', 'pred_camt']),
        ]:
            for key in keys:
                value = container.get(key)
                if value is not None and hasattr(value, 'shape') and len(value.shape) > 0:
                    lengths.append(int(value.shape[0]))
        if not lengths:
            return gt_data, pred_data
        target_len = min(lengths)
        if max(lengths) != target_len:
            self.logger.warning(f"Aligning evaluation temporal lengths to {target_len}: {sorted(set(lengths))}")
            for container, keys in [
                (gt_data, ['gt_j3d', 'gt_vert', 'gt_ori', 'gt_j3d_cam', 'gt_vert_cam', 'ext']),
                (pred_data, ['pred_j3d', 'pred_vert', 'pred_j3d_w', 'pred_vert_w', 'pred_ori_w', 'pred_camr', 'pred_camt']),
            ]:
                for key in keys:
                    value = container.get(key)
                    if value is not None and hasattr(value, 'shape') and len(value.shape) > 0 and int(value.shape[0]) > target_len:
                        container[key] = value[:target_len]
        return gt_data, pred_data

    def _match_camera_to_prediction_length(self, cam_r, cam_t, pred_len, valid_mask=None):
        """Match full-frame SLAM camera trajectory to valid-frame SMPL predictions."""
        cam_len = int(cam_r.shape[0])
        if cam_len == pred_len:
            return cam_r, cam_t
        if valid_mask is not None:
            mask = np.asarray(valid_mask).astype(bool)
            if len(mask) == cam_len and int(mask.sum()) == pred_len:
                self.logger.info(f"Filtering camera trajectory by valid mask: {cam_len} -> {pred_len}")
                mask_t = torch.from_numpy(mask).to(cam_r.device)
                return cam_r[mask_t], cam_t[mask_t]
        keep = min(cam_len, pred_len)
        self.logger.warning(f"Camera/SMPL length mismatch: camera={cam_len}, smpl={pred_len}; truncating to {keep}")
        return cam_r[:keep], cam_t[:keep]
    
    def _evaluate_local_motion(self, gt_data, pred_data, m2mm, pelvis_idxs=None):
        """评估局部运动

        注意：会把 pelvis-aligned 后的 pred_j3d/gt_j3d_cam/pred_vert/gt_vert_cam
        写回到 pred_data/gt_data 中，与官方 run_eval.py 的变量复用行为保持一致。
        后续 accel 计算依赖的也是 pelvis-aligned 版本（否则会混入 root 的 camera-space
        运动，使 accel 被严重高估）。
        """
        from lib.utils.eval_utils import (
            batch_align_by_pelvis,
            batch_compute_similarity_transform_torch
        )

        if pelvis_idxs is None:
            pelvis_idxs = [1, 2]

        pred_j3d, gt_j3d_cam, pred_vert, gt_vert_cam = batch_align_by_pelvis(
            [pred_data['pred_j3d'], gt_data['gt_j3d_cam'],
             pred_data['pred_vert'], gt_data['gt_vert_cam']],
            pelvis_idxs=pelvis_idxs
        )

        # 将 pelvis-aligned 版本写回，供后续 accel 计算使用
        # （与 scripts/emdb/run_eval.py 的变量复用一致）
        pred_data['pred_j3d'] = pred_j3d
        pred_data['pred_vert'] = pred_vert
        gt_data['gt_j3d_cam'] = gt_j3d_cam
        gt_data['gt_vert_cam'] = gt_vert_cam

        metrics = {}

        # PA-MPJPE
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d, gt_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['pa_mpjpe'] = float(pa_mpjpe.mean())
        metrics['_sum_pa_mpjpe'] = float(pa_mpjpe.sum())
        metrics['_cnt_pa_mpjpe'] = int(pa_mpjpe.size)

        # MPJPE
        mpjpe = torch.sqrt(((pred_j3d - gt_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['mpjpe'] = float(mpjpe.mean())
        metrics['_sum_mpjpe'] = float(mpjpe.sum())
        metrics['_cnt_mpjpe'] = int(mpjpe.size)

        # PVE
        pve = torch.sqrt(((pred_vert - gt_vert_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
        metrics['pve'] = float(pve.mean())
        metrics['_sum_pve'] = float(pve.sum())
        metrics['_cnt_pve'] = int(pve.size)

        return metrics
    
    def _evaluate_global_motion(self, gt_j3d, pred_j3d_w, m2mm):
        """评估全局运动"""
        from lib.utils.eval_utils import first_align_joints, global_align_joints, compute_jpe
        
        chunk_length = self.chunk_length
        w_mpjpe, wa_mpjpe = [], []
        
        num_frames = len(gt_j3d)
        
        for start in range(0, num_frames, chunk_length):
            end = min(num_frames, start + chunk_length)
            if end - start < 2:
                continue
            
            target_j3d = gt_j3d[start:end].clone().cpu()
            pred_j3d = pred_j3d_w[start:end].clone().cpu()
            
            w_j3d = first_align_joints(target_j3d, pred_j3d)
            wa_j3d = global_align_joints(target_j3d, pred_j3d)
            
            w_jpe = compute_jpe(target_j3d, w_j3d)
            wa_jpe = compute_jpe(target_j3d, wa_j3d)
            
            w_mpjpe.append(w_jpe)
            wa_mpjpe.append(wa_jpe)
        
        if w_mpjpe:
            w_mpjpe = np.concatenate(w_mpjpe) * m2mm
            wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm
            return {
                'w_mpjpe': float(w_mpjpe.mean()),
                'wa_mpjpe': float(wa_mpjpe.mean()),
                '_sum_w_mpjpe': float(w_mpjpe.sum()),
                '_cnt_w_mpjpe': int(w_mpjpe.size),
                '_sum_wa_mpjpe': float(wa_mpjpe.sum()),
                '_cnt_wa_mpjpe': int(wa_mpjpe.size),
            }

        return {'w_mpjpe': 0.0, 'wa_mpjpe': 0.0}
    
    def _compute_acceleration_error(self, pred_j3d, gt_j3d):
        """计算加速度误差"""
        from lib.utils.eval_utils import compute_error_accel
        accel = compute_error_accel(
            joints_pred=pred_j3d.cpu(), 
            joints_gt=gt_j3d.cpu()
        )[1:-1]
        return accel
    
    def _compute_rte(self, gt_root, pred_root):
        """计算根轨迹误差"""
        from lib.utils.eval_utils import compute_rte
        return compute_rte(gt_root, pred_root)
    
    def _compute_erve(self, gt_ori, gt_j3d, pred_ori, pred_j3d):
        """计算自我中心根速度误差"""
        from lib.utils.eval_utils import computer_erve
        return computer_erve(gt_ori, gt_j3d, pred_ori, pred_j3d)

    def _compute_jitter(self, pred_j3d_w):
        """计算 GVHMR 风格的全局关节 jitter"""
        from lib.utils.eval_utils import compute_jitter
        return compute_jitter(pred_j3d_w, fps=self.fps)

    def _compute_foot_sliding(self, gt_vert_w, pred_vert_w):
        """计算 GVHMR 风格的 Foot-Sliding（返回米）"""
        from lib.utils.eval_utils import compute_foot_sliding
        return compute_foot_sliding(gt_vert_w, pred_vert_w)

    def _compute_foot_physical_metrics(self, gt_vert_w, pred_vert_w, is_rich=False):
        """计算足部悬空率和地面穿透深度 proxy / RICH floor-aware 指标。"""
        from lib.utils.eval_utils import compute_foot_physical_metrics
        ground_mode = self.config.get('physical_ground_mode', 'auto')
        ground_height = None
        if ground_mode == 'fixed' or (ground_mode == 'auto' and is_rich) or (ground_mode == 'rich_floor' and is_rich):
            ground_height = self.config.get('physical_ground_height', 0.0)
            if is_rich:
                self.logger.info(
                    f"RICH floor-aware physical metrics: axis={self.config.get('physical_ground_axis', 1)}, "
                    f"ground_height={ground_height}"
                )
        return compute_foot_physical_metrics(
            gt_vert_w,
            pred_vert_w,
            foot_vertex_ids=self.config.get('physical_foot_vertex_ids', [3216, 3387, 6617, 6787]),
            ground_axis=self.config.get('physical_ground_axis', 1),
            ground_quantile=self.config.get('physical_ground_quantile', 0.01),
            ground_height=ground_height,
            contact_vel_thr=self.config.get('physical_contact_vel_thr', 0.01),
            contact_height_thr=self.config.get('physical_contact_height_thr', 0.05),
            floating_thr=self.config.get('physical_floating_thr', 0.05),
            penetration_tol=self.config.get('physical_penetration_tol', 0.0),
        )
    
    def _evaluate_camera_motion(self, data: PipelineData, gt_data, pred_data=None):
        """评估相机运动"""
        from lib.utils.rotation_conversions import matrix_to_quaternion
        from lib.camera.slam_utils import eval_slam
        
        ext = gt_data['ext']
        cam_r = ext[:, :3, :3].transpose(0, 2, 1)
        cam_t = np.einsum('bij, bj->bi', cam_r, -ext[:, :3, -1])
        cam_q = matrix_to_quaternion(torch.from_numpy(cam_r)).numpy()
        
        # Prefer pred_data because it has already been valid-mask aligned.
        pred_camr = None
        pred_camt = None
        if pred_data is not None:
            pred_camr = pred_data.get('pred_camr')
            pred_camt = pred_data.get('pred_camt')
        
        if pred_camr is None and data.camera_params is not None:
            pred_camr = data.camera_params.R
            pred_camt = data.camera_params.T
        
        if pred_camr is None or pred_camt is None:
            self.logger.warning("No camera R/T available for ATE evaluation, skipping")
            return {'ate': 0.0, 'ate_s': 0.0}
        
        if isinstance(pred_camr, torch.Tensor):
            pred_camr = pred_camr
        else:
            pred_camr = torch.from_numpy(pred_camr)
        
        if isinstance(pred_camt, torch.Tensor):
            pred_camt = pred_camt
        else:
            pred_camt = torch.from_numpy(pred_camt)
        
        if pred_camr.shape[0] != cam_t.shape[0]:
            keep = min(int(pred_camr.shape[0]), int(cam_t.shape[0]))
            self.logger.warning(
                f"ATE trajectory length mismatch: pred={pred_camr.shape[0]}, gt={cam_t.shape[0]}; truncating to {keep}"
            )
            pred_camr = pred_camr[:keep]
            pred_camt = pred_camt[:keep]
            cam_t = cam_t[:keep]
            cam_q = cam_q[:keep]

        pred_camq = matrix_to_quaternion(pred_camr)
        pred_traj = torch.concat([pred_camt, pred_camq], dim=-1).numpy()
        
        # 检查是否已经进行了尺度对齐
        scale_aligned = data.metadata.get('scale_aligned_to_first_slam', False)
        if scale_aligned:
            self.logger.warning(
                "Camera trajectory was scale-aligned to reference trajectory. "
                "ATE and ATE_s metrics will be very similar because scale is pre-corrected."
            )
        
        metrics = {}
        
        # ATE with scale correction
        stats_slam, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=True)
        metrics['ate'] = stats_slam['mean']
        
        # ATE without scale correction  
        stats_metric, _, _ = eval_slam(pred_traj.copy(), cam_t, cam_q, correct_scale=False)
        metrics['ate_s'] = stats_metric['mean']
        
        return metrics


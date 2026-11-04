"""
Warm Start DROID-SLAM

Provides modified DROID-SLAM components that support:
1. Forced keyframe selection (bypassing MotionFilter's optical flow threshold)
2. Warm start initialization (injecting pre-computed poses from a previous SLAM run)
3. Disabled keyframe removal in frontend (preserving all forced keyframes)

These components inherit from the original DROID-SLAM classes to avoid modifying
the third-party code directly.
"""

import sys
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')

import torch
import lietorch
import numpy as np
import cv2
import json
import os
import time
import shutil
import zipfile
import http.client
from tqdm import tqdm
from glob import glob
from PIL import Image
from collections import OrderedDict

from droid import Droid
from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller
from factor_graph import FactorGraph
from lietorch import SE3

import geom.projective_ops as pops
from modules.corr import CorrBlock

from .slam_utils import slam_args, parser, image_stream, est_calib, get_dimention, preprocess_masks
from .slam_utils import zoedepth_load_compat, zoedepth_midas_beit_compat
from .est_scale import est_scale_hybrid
from ..utils.rotation_conversions import quaternion_to_matrix


def _cleanup_torch_hub_download_cache():
    """Remove incomplete torch hub zip downloads left by interrupted GitHub downloads."""
    hub_dir = torch.hub.get_dir()
    for zip_path in glob(os.path.join(hub_dir, '*.zip')):
        try:
            print(f"[torch.hub] Removing possibly corrupted zip: {zip_path}")
            os.remove(zip_path)
        except OSError as exc:
            print(f"[torch.hub] Failed to remove {zip_path}: {exc}")


def _is_torch_hub_download_error(exc):
    if isinstance(exc, (zipfile.BadZipFile, http.client.IncompleteRead)):
        return True
    return any(isinstance(arg, (zipfile.BadZipFile, http.client.IncompleteRead)) for arg in getattr(exc, 'args', []))


def _load_zoedepth_model():
    from .zoedepth import load_zoedepth_model
    return load_zoedepth_model()


class WarmstartMotionFilter(MotionFilter):
    """
    Modified MotionFilter that forces keyframe selection based on a pre-determined
    set of frame timestamps, instead of using optical flow magnitude thresholds.
    
    Optionally injects pre-computed SE3 poses for warm start initialization.
    """

    def __init__(self, net, video, forced_keyframes, initial_poses_se3=None,
                 initial_disps=None, thresh=2.5, device="cuda:0"):
        """
        Args:
            net: DroidNet
            video: DepthVideo
            forced_keyframes: set of int timestamps that must be keyframes
            initial_poses_se3: dict mapping tstamp (int) -> numpy array [7] (SE3 format)
            initial_disps: dict mapping tstamp (int) -> numpy array [h//8, w//8]
            thresh: fallback threshold (not used when forced_keyframes is provided)
            device: cuda device
        """
        super().__init__(net, video, thresh=thresh, device=device)
        self.forced_keyframes = set(forced_keyframes)
        self.initial_poses_se3 = initial_poses_se3 or {}
        self.initial_disps = initial_disps or {}

    @torch.amp.autocast('cuda', enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None, mask=None):
        """
        Modified track: only adds frames that are in forced_keyframes.
        Injects pre-computed pose if available.
        """

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features (always needed for keyframes)
        gmap = self._MotionFilter__feature_encoder(inputs)
        if mask is None:
            mask = torch.zeros([gmap.shape[-2], gmap.shape[-1]]).to(gmap)

        # Check if this frame should be a keyframe
        is_forced = int(tstamp) in self.forced_keyframes

        ### always add first frame ###
        if self.video.counter.value == 0:
            if not is_forced:
                # First forced keyframe hasn't arrived yet, skip
                return

            net, inp = self._MotionFilter__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap

            # Use warm start pose if available, otherwise identity
            init_pose = Id
            init_disp = 1.0
            ts = int(tstamp)
            if ts in self.initial_poses_se3:
                init_pose = torch.as_tensor(
                    self.initial_poses_se3[ts], dtype=torch.float, device="cuda"
                )
            if ts in self.initial_disps:
                init_disp = torch.as_tensor(
                    self.initial_disps[ts], dtype=torch.float, device="cuda"
                )

            self.video.append(
                tstamp, image[0], init_pose, init_disp, depth,
                intrinsics / 8.0, gmap, net[0, 0], inp[0, 0], mask
            )

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': True,
                    'reason': 'first_frame_forced',
                    'flow_magnitude': None,
                    'keyframe_idx': 0,
                })

        ### for subsequent frames, only add if forced ###
        elif is_forced:
            self.count = 0
            net, inp = self._MotionFilter__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap

            # Use warm start pose if available
            init_pose = None  # None = use frontend's prediction (poses[t1-1])
            init_disp = None
            ts = int(tstamp)
            if ts in self.initial_poses_se3:
                init_pose = torch.as_tensor(
                    self.initial_poses_se3[ts], dtype=torch.float, device="cuda"
                )
            if ts in self.initial_disps:
                init_disp = torch.as_tensor(
                    self.initial_disps[ts], dtype=torch.float, device="cuda"
                )

            self.video.append(
                tstamp, image[0], init_pose, init_disp, depth,
                intrinsics / 8.0, gmap, net[0], inp[0], mask
            )

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': True,
                    'reason': 'forced_keyframe',
                    'flow_magnitude': None,
                    'keyframe_idx': self.video.counter.value - 1,
                })

        else:
            # Not a forced keyframe, skip entirely
            self.count += 1

            if self._record_filter:
                self.filter_log.append({
                    'tstamp': int(tstamp),
                    'accepted': False,
                    'reason': 'not_forced_keyframe',
                    'flow_magnitude': None,
                })


class WarmstartDroidFrontend(DroidFrontend):
    """
    Modified DroidFrontend that disables keyframe removal.
    
    Since we force specific keyframes, the frontend should not remove any of them
    based on the distance metric.
    """

    def __init__(self, net, video, args):
        super().__init__(net, video, args)
        # Disable keyframe removal by setting threshold to 0
        # (distance is always >= 0, so d < 0 is never True)
        self.keyframe_thresh = 0.0

    # Note: __update uses name mangling (_DroidFrontend__update).
    # By setting keyframe_thresh=0.0 in __init__, the removal branch
    # in the parent's __update will never trigger, so we don't need
    # to override __update or __call__.


class WarmstartDroid(Droid):
    """
    Modified Droid that uses WarmstartMotionFilter and WarmstartDroidFrontend.
    """

    def __init__(self, args, forced_keyframes, initial_poses_se3=None, initial_disps=None):
        """
        Args:
            args: SLAM arguments (same as original Droid)
            forced_keyframes: set/list of int timestamps for forced keyframes
            initial_poses_se3: dict mapping tstamp -> SE3 pose array [7]
            initial_disps: dict mapping tstamp -> disparity array [h//8, w//8]
        """
        # We can't call super().__init__ because it creates MotionFilter/DroidFrontend
        # that we want to replace. So we replicate the init logic.
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # Use warmstart motion filter
        self.filterx = WarmstartMotionFilter(
            self.net, self.video,
            forced_keyframes=forced_keyframes,
            initial_poses_se3=initial_poses_se3,
            initial_disps=initial_disps,
            thresh=args.filter_thresh
        )

        # Use warmstart frontend (no keyframe removal)
        self.frontend = WarmstartDroidFrontend(self.net, self.video, self.args)

        # Standard backend
        self.backend = DroidBackend(self.net, self.video, self.args)

        # Visualizer
        if not self.disable_vis:
            from vis_headless import droid_visualization
            print('Using headless ...')
            from torch.multiprocessing import Process
            self.visualizer = Process(target=droid_visualization, args=(self.video, '.'))
            self.visualizer.start()

        # Post processor
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


def _load_image_for_slam(imfile, calib):
    """Load and preprocess a single image for DROID-SLAM (resize + crop to multiple of 8)."""
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image = cv2.imread(imfile)
    if len(calib) > 4:
        image = cv2.undistort(image, K, calib[4:])

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    image = torch.as_tensor(image).permute(2, 0, 1)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= (w1 / w0)
    intrinsics[1::2] *= (h1 / h0)

    return image[None], intrinsics


def _compute_background_texture_score(image_path, mask, blur_ksize=5):
    """计算图像中非人体区域（背景）的纹理丰富程度。

    使用 Laplacian 方差衡量纹理丰富度。值越高，纹理越丰富。

    Args:
        image_path: 原始图像路径
        mask: 人体 mask (H, W)，>0 表示人体区域
        blur_ksize: 高斯模糊核大小，用于去噪

    Returns:
        float: 背景区域的纹理得分（Laplacian 方差）
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    # 对齐 mask 尺寸
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.shape[0] != img.shape[0] or mask_np.shape[1] != img.shape[1]:
            mask_np = cv2.resize(mask_np.astype(np.float32), (img.shape[1], img.shape[0]))
        bg_mask = mask_np < 0.5  # 背景区域
    else:
        bg_mask = np.ones_like(img, dtype=bool)

    # 只在背景区域计算
    bg_pixels = np.sum(bg_mask)
    if bg_pixels < 100:
        # 背景区域太小，无法可靠评估
        return 0.0

    # Laplacian 边缘检测
    blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # 只取背景区域的 Laplacian 方差
    bg_laplacian = laplacian[bg_mask]
    score = float(np.var(bg_laplacian))

    return score


def _blend_slam_images(original_image, rendered_image, alpha):
    """Blend original and rendered DROID input tensors in pixel space."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = original_image.float() * (1.0 - alpha) + rendered_image.float() * alpha
    blended = blended.clamp(0, 255)
    if original_image.dtype == torch.uint8:
        blended = blended.round().to(torch.uint8)
    else:
        blended = blended.to(original_image.dtype)
    return blended


def run_slam_warmstart(imagedir, forced_keyframes, initial_poses_se3=None,
                       initial_disps=None, masks=None, calib=None,
                       depth=None, record_debug=True, original_imagedir=None,
                       use_rendered_for_keyframes=False,
                       keyframe_render_mode='all',
                       texture_threshold=500.0,
                       render_fusion_mode='replace',
                       render_alpha=1.0,
                       frontend_only=False,
                       debug_filename='droid_frontend_debug.json'):
    """
    Run DROID-SLAM with forced keyframes and warm start initialization.
    
    Args:
        imagedir: directory containing rendered .jpg files (adjacent_smpl/ images)
        forced_keyframes: set/list of int frame indices to force as keyframes
        initial_poses_se3: dict mapping frame_idx -> SE3 pose [7]
        initial_disps: dict mapping frame_idx -> disparity [h//8, w//8]
        masks: optional human masks for masking
        calib: camera intrinsics [fx, fy, cx, cy]
        depth: optional depth input
        record_debug: whether to record debug info
        original_imagedir: directory containing original (unmodified) images.
        use_rendered_for_keyframes: if True, keyframes use rendered images from
            imagedir (with SMPL mesh), non-keyframes use original images + mask.
            If False, all frames use original images + mask (baseline behavior).
        keyframe_render_mode: keyframe rendering strategy
            - 'all': all keyframes use rendered images (original behavior)
            - 'adaptive': per-keyframe decision based on background texture richness.
              Keyframes with rich background texture use mask strategy (to preserve
              background features), while keyframes with poor texture use rendered
              images (SMPL mesh provides visual constraints).
        texture_threshold: threshold for adaptive mode. Keyframes with background
            texture score below this value use rendered images; above use mask.
        render_fusion_mode: 'replace' uses rendered image directly; 'alpha' blends
            original and rendered images for rendered keyframes.
        render_alpha: rendered image weight for alpha fusion.
    
    Returns:
        droid: WarmstartDroid object
        traj: full trajectory [N_total, 7] for all input frames
    """
    droid = None

    # Determine base image dir (for calib, mask sizing, traj_filler)
    base_imagedir = original_imagedir if original_imagedir else imagedir

    if calib is None:
        calib = est_calib(base_imagedir)

    if masks is not None:
        img_msks, conf_msks = preprocess_masks(base_imagedir, masks)

    forced_set = set(int(k) for k in forced_keyframes)

    # Pre-load rendered images by original frame index if needed
    rendered_image_map = None
    if use_rendered_for_keyframes and original_imagedir:
        rendered_image_list = sorted(glob(os.path.join(imagedir, '*.jpg')))
        rendered_image_map = {}
        for path in rendered_image_list:
            try:
                rendered_image_map[int(os.path.splitext(os.path.basename(path))[0])] = path
            except ValueError:
                pass
        print(f"[Warmstart] Keyframes use rendered images from {imagedir}")
        print(f"[Warmstart] Non-keyframes use original images from {original_imagedir}")
        print(f"[Warmstart] Render fusion mode: {render_fusion_mode}, alpha={render_alpha}")

    # Adaptive mode: pre-compute background texture scores for keyframes
    adaptive_use_rendered = {}  # frame_idx -> bool
    if use_rendered_for_keyframes and keyframe_render_mode == 'adaptive' and masks is not None:
        print(f"[Warmstart] Adaptive mode: computing background texture scores (threshold={texture_threshold})...")
        original_images = sorted(glob(os.path.join(base_imagedir, '*.jpg')))
        num_rendered = 0
        num_masked = 0
        for kf_idx in sorted(forced_set):
            if kf_idx < len(original_images) and kf_idx < len(masks):
                score = _compute_background_texture_score(original_images[kf_idx], masks[kf_idx])
                use_render = score < texture_threshold
                adaptive_use_rendered[kf_idx] = use_render
                if use_render:
                    num_rendered += 1
                else:
                    num_masked += 1
                print(f"  Keyframe {kf_idx}: texture_score={score:.1f} -> {'RENDERED (low texture)' if use_render else 'MASKED (rich texture)'}")
        print(f"[Warmstart] Adaptive summary: {num_rendered} keyframes use rendered, {num_masked} use mask")

    for (t, image, intrinsics) in tqdm(image_stream(base_imagedir, calib)):
        if droid is None:
            slam_args.image_size = [image.shape[2], image.shape[3]]
            # DROID's DepthVideo buffer stores retained keyframes, not all frames.
            # Dense forced keyframes (e.g. pre_dis=5 on long EMDB2 sequences) can
            # exceed the default 512 buffer and crash inside DroidFrontend. Grow it
            # based on the forced-keyframe budget while keeping memory bounded.
            default_buffer = int(getattr(slam_args, 'buffer', 512))
            required_buffer = len(forced_set) + 128
            slam_args.buffer = min(max(default_buffer, required_buffer), 4096)
            print(f"[Warmstart] DROID buffer={slam_args.buffer} (forced={len(forced_set)})")
            droid = WarmstartDroid(
                slam_args,
                forced_keyframes=forced_set,
                initial_poses_se3=initial_poses_se3,
                initial_disps=initial_disps
            )
            if record_debug:
                droid.enable_recording(True)

        # Decide which image to use for this frame
        is_keyframe = int(t) in forced_set

        # Determine if this keyframe should use rendered image
        should_use_rendered = False
        if use_rendered_for_keyframes and is_keyframe and rendered_image_map is not None:
            if keyframe_render_mode == 'adaptive':
                should_use_rendered = adaptive_use_rendered.get(int(t), False)
            else:
                # 'all' mode: all keyframes use rendered images
                should_use_rendered = True

        # A keyframe selected for rendering whose rendered image is missing must
        # fall through to the masked-original path below. Tracking it on the raw
        # image would leave human pixels in on that frame alone, which is not what
        # the baseline pass saw and injects spurious motion into the delta.
        rendered_path = None
        if should_use_rendered:
            rendered_path = rendered_image_map.get(int(t))
            if rendered_path is None:
                should_use_rendered = False

        if should_use_rendered:
            # Keyframe: use rendered image (with SMPL mesh from adjacent_smpl/)
            rendered_image, _ = _load_image_for_slam(rendered_path, calib)
            if render_fusion_mode == 'alpha':
                track_image = _blend_slam_images(image, rendered_image, render_alpha)
            else:
                track_image = rendered_image
            # Keyframe still uses mask for correlation weighting:
            # SMPL mesh provides visual context for feature extraction,
            # but mask prevents false matches on non-static human regions
            if masks is not None:
                conf_msk = conf_msks[t]
                droid.track(t, track_image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
            else:
                droid.track(t, track_image, intrinsics=intrinsics, depth=depth, mask=None)
        else:
            # Non-keyframe (or baseline/adaptive-masked keyframe): use original image + mask
            if masks is not None:
                img_msk = img_msks[t]
                conf_msk = conf_msks[t]
                image = image * (img_msk < 0.5)
                droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
            else:
                droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=None)

    if droid is None:
        return None, None

    # save debug info before optional backend termination
    debug_info = None
    if record_debug:
        debug_info = droid.get_debug_info()

    if frontend_only:
        n = droid.video.counter.value
        keyframe_info = {
            'keyframe_tstamps': droid.video.tstamp.cpu().int().numpy()[:n],
            'keyframe_poses_se3': droid.video.poses[:n].cpu().numpy(),
            'keyframe_disps': droid.video.disps[:n].cpu().numpy(),
        }
        if debug_info is not None:
            debug_info['num_keyframes_frontend_only'] = n
            debug_info['keyframe_tstamps_frontend_only'] = keyframe_info['keyframe_tstamps'].tolist()
            debug_info['warmstart'] = True
            debug_info['frontend_only'] = True
            debug_info['forced_keyframes'] = sorted(list(forced_set))
            debug_info['use_rendered_for_keyframes'] = use_rendered_for_keyframes
            debug_info['keyframe_render_mode'] = keyframe_render_mode
            debug_info['render_fusion_mode'] = render_fusion_mode
            debug_info['render_alpha'] = render_alpha
            if keyframe_render_mode == 'adaptive':
                debug_info['texture_threshold'] = texture_threshold
                debug_info['adaptive_decisions'] = {
                    str(k): 'rendered' if v else 'masked' for k, v in adaptive_use_rendered.items()
                }
            debug_dir = os.path.join(imagedir, '..', 'droid_debug')
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, debug_filename)
            with open(debug_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            print(f"[DROID Warmstart Debug] Saved frontend-only debug info to {debug_path}")
        return droid, keyframe_info

    # traj_filler uses original images (unmodified)
    traj = droid.terminate(image_stream(base_imagedir, calib))

    # save debug info
    if debug_info is not None:
        n = droid.video.counter.value
        debug_info['num_keyframes_after_backend'] = n
        debug_info['keyframe_tstamps_after_backend'] = droid.video.tstamp.cpu().numpy()[:n].tolist()
        debug_info['warmstart'] = True
        debug_info['frontend_only'] = False
        debug_info['forced_keyframes'] = sorted(list(forced_set))
        debug_info['use_rendered_for_keyframes'] = use_rendered_for_keyframes
        debug_info['keyframe_render_mode'] = keyframe_render_mode
        debug_info['render_fusion_mode'] = render_fusion_mode
        debug_info['render_alpha'] = render_alpha
        if keyframe_render_mode == 'adaptive':
            debug_info['texture_threshold'] = texture_threshold
            debug_info['adaptive_decisions'] = {
                str(k): 'rendered' if v else 'masked' for k, v in adaptive_use_rendered.items()
            }

        debug_dir = os.path.join(imagedir, '..', 'droid_debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, debug_filename)
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"[DROID Warmstart Debug] Saved debug info to {debug_path}")

    return droid, traj


def run_metric_slam_warmstart(img_folder, forced_keyframes, initial_poses_se3=None,
                              initial_disps=None, masks=None, calib=None,
                              is_static=False, original_image_dir=None,
                              use_rendered_for_keyframes=False,
                              keyframe_render_mode='all',
                              texture_threshold=500.0,
                              render_fusion_mode='replace',
                              render_alpha=1.0,
                              # --- SMPL pose-only refinement (optional) ---
                              smpl_refine_cfg=None,
                              smpl_rotmat=None,
                              smpl_betas=None,
                              smpl_trans=None,
                              ref_cam_R=None,
                              ref_cam_T=None,
                              refine_intrinsics=None,
                              adjacent_projection_refine_cfg=None,
                              adjacent_projection_info=None,
                              return_keyframe_info=False):
    """
    Run metric-scale SLAM with forced keyframes and warm start.
    
    Returns full-frame camera trajectory (no interpolation needed).

    If ``return_keyframe_info=True``, additionally returns a dict with the raw
    DROID keyframe timestamps, SE3 poses and low-res disparities (captured
    before the droid object is released), suitable for warm-starting a later
    frontend-only pass.
    
    Args:
        img_folder: directory containing .jpg files (rendered images with SMPL mesh)
        forced_keyframes: set/list of int frame indices to force as keyframes
        initial_poses_se3: dict mapping frame_idx -> SE3 pose [7]
        initial_disps: dict mapping frame_idx -> disparity [h//8, w//8]
        masks: optional human masks
        calib: camera intrinsics [fx, fy, cx, cy]
        is_static: if True, return identity camera motion
        original_image_dir: directory containing original images
        use_rendered_for_keyframes: if True, keyframes use rendered images,
            non-keyframes use original images + mask
        keyframe_render_mode: 'all' or 'adaptive' (see run_slam_warmstart)
        texture_threshold: threshold for adaptive mode
        smpl_refine_cfg: optional dict with keys
            ``enabled, weight, huber_delta, num_iters, damping,
              pelvis_only, refine_translation_only``.
            If ``None`` or ``enabled=False``, refinement is skipped.
        smpl_rotmat / smpl_betas / smpl_trans: full-frame SMPL params (camera
            frame). Required when refinement is enabled.
        ref_cam_R / ref_cam_T: reference c2w extrinsics used to anchor the
            world-space SMPL joints (typically the first SLAM result or the
            TRAM official cached cameras).
        refine_intrinsics: 4-tuple (fx, fy, cx, cy) used to project SMPL
            joints. If None, falls back to ``calib``.
    
    Returns:
        pred_cam_r: rotation matrices [N, 3, 3]
        pred_cam_t: translation vectors [N, 3]
    """
    # Use original images directory for frame counting and depth estimation
    base_img_folder = original_image_dir if original_image_dir else img_folder
    imgfiles = sorted(glob(f'{base_img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        # Fallback to img_folder
        base_img_folder = img_folder
        imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    if is_static:
        pred_cam_t = torch.zeros([len(imgfiles), 3])
        pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)
        if return_keyframe_info:
            return pred_cam_r, pred_cam_t, None
        return pred_cam_r, pred_cam_t

    # Run warmstart SLAM
    droid, traj = run_slam_warmstart(
        img_folder,
        forced_keyframes=forced_keyframes,
        initial_poses_se3=initial_poses_se3,
        initial_disps=initial_disps,
        masks=masks,
        calib=calib,
        original_imagedir=original_image_dir,
        use_rendered_for_keyframes=use_rendered_for_keyframes,
        keyframe_render_mode=keyframe_render_mode,
        texture_threshold=texture_threshold,
        render_fusion_mode=render_fusion_mode,
        render_alpha=render_alpha,
    )

    if droid is None:
        raise RuntimeError(
            f"Warmstart DROID-SLAM failed. No images found or processed in {img_folder}."
        )

    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]

    # Capture raw keyframe states before releasing droid (for optional warm start)
    keyframe_info = None
    if return_keyframe_info:
        keyframe_info = {
            'keyframe_tstamps': tstamp.copy(),                          # [n] int array
            'keyframe_poses_se3': droid.video.poses[:n].cpu().numpy(),  # [n, 7] SE3
            'keyframe_disps': droid.video.disps[:n].cpu().numpy(),      # [n, h//8, w//8]
        }

    del droid
    torch.cuda.empty_cache()

    # Estimate metric depth using ZoeDepth on original images
    zoe_imgfiles = imgfiles
    print(f"[ZoeDepth] Using images from {base_img_folder}")

    # 兼容新版 timm，见 slam_utils 注释
    with zoedepth_load_compat():
        model_zoe_n = _load_zoedepth_model()
    zoedepth_midas_beit_compat(model_zoe_n)
    _ = model_zoe_n.eval()
    model_zoe_n = model_zoe_n.to('cuda')

    pred_depths = []
    H, W = get_dimention(base_img_folder)
    for t in tqdm(tstamp):
        img = cv2.imread(zoe_imgfiles[t])[:, :, ::-1]
        img = cv2.resize(img, (W, H))
        img_pil = Image.fromarray(img)
        pred_depth = model_zoe_n.infer_pil(img_pil)
        pred_depths.append(pred_depth)

    # Estimate metric scale
    scales_ = []
    for i in tqdm(range(len(tstamp))):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
        slam_depth = 1 / disp

        if masks is None:
            msk = None
        else:
            msk = masks[t].numpy()

        scale = est_scale_hybrid(slam_depth, pred_depth, msk=msk)
        scales_.append(scale)
    scale = np.median(scales_)
    if keyframe_info is not None:
        # Per-keyframe metric/DROID scales, so callers can correct local scale drift.
        keyframe_info['keyframe_metric_scales'] = np.asarray(scales_, dtype=np.float64)
        keyframe_info['metric_scale'] = float(scale)

    del model_zoe_n
    del pred_depths
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to metric-scale camera extrinsics
    pred_cam_t = torch.tensor(traj[:, :3]) * scale
    pred_cam_q = torch.tensor(traj[:, 3:])
    pred_cam_r = quaternion_to_matrix(pred_cam_q[:, [3, 0, 1, 2]])

    # ------------------------------------------------------------------
    # Optional: SMPL joint reprojection pose-only refinement on keyframes
    # ------------------------------------------------------------------
    if smpl_refine_cfg is not None and smpl_refine_cfg.get('enabled', False):
        if (smpl_rotmat is None or smpl_betas is None or smpl_trans is None
                or ref_cam_R is None or ref_cam_T is None):
            print(
                "[SMPL Refine] Skipped: missing required inputs "
                "(smpl_rotmat / smpl_betas / smpl_trans / ref_cam_R / ref_cam_T)."
            )
        else:
            # Determine intrinsics for projection
            if refine_intrinsics is not None:
                fx, fy, cx, cy = [float(v) for v in refine_intrinsics]
            elif calib is not None:
                fx, fy, cx, cy = [float(v) for v in calib[:4]]
            else:
                fx = fy = 1000.0
                cx = pred_cam_t.new_zeros(()).item()  # placeholder, see below
                cy = 0.0
                print("[SMPL Refine] WARNING: no intrinsics, using default fx=fy=1000")

            # Sanity-check tensor lengths against trajectory
            n_traj = pred_cam_t.shape[0]

            def _to_tensor(x):
                if isinstance(x, np.ndarray):
                    return torch.from_numpy(x).float()
                return torch.as_tensor(x).float()

            smpl_rotmat_t = _to_tensor(smpl_rotmat)
            smpl_betas_t = _to_tensor(smpl_betas)
            smpl_trans_t = _to_tensor(smpl_trans)
            if smpl_trans_t.ndim == 3 and smpl_trans_t.shape[1] == 1:
                smpl_trans_t = smpl_trans_t.squeeze(1)
            ref_R_t = _to_tensor(ref_cam_R)
            ref_T_t = _to_tensor(ref_cam_T)

            n_min = min(n_traj, smpl_rotmat_t.shape[0], ref_R_t.shape[0])
            kf_in_range = sorted(set(int(t) for t in forced_keyframes if int(t) < n_min))

            if len(kf_in_range) == 0:
                print("[SMPL Refine] No valid keyframes within range, skipped.")
            else:
                from .smpl_pose_refine import refine_keyframe_poses_with_smpl

                print(
                    f"[SMPL Refine] Running pose-only LM on {len(kf_in_range)} keyframes "
                    f"(weight={smpl_refine_cfg.get('weight', 1.0)}, "
                    f"iters={smpl_refine_cfg.get('num_iters', 5)}, "
                    f"huber_delta={smpl_refine_cfg.get('huber_delta', 4.0)})"
                )
                pred_cam_r, pred_cam_t, dbg = refine_keyframe_poses_with_smpl(
                    cam_R=pred_cam_r,
                    cam_T=pred_cam_t,
                    ref_R=ref_R_t[:n_min],
                    ref_T=ref_T_t[:n_min],
                    smpl_rotmat=smpl_rotmat_t[:n_min],
                    smpl_betas=smpl_betas_t[:n_min],
                    smpl_trans=smpl_trans_t[:n_min],
                    keyframe_indices=kf_in_range,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    weight=float(smpl_refine_cfg.get('weight', 1.0)),
                    huber_delta=float(smpl_refine_cfg.get('huber_delta', 4.0)),
                    num_iters=int(smpl_refine_cfg.get('num_iters', 5)),
                    damping=float(smpl_refine_cfg.get('damping', 1e-3)),
                    pelvis_only=bool(smpl_refine_cfg.get('pelvis_only', False)),
                    refine_translation_only=bool(
                        smpl_refine_cfg.get('refine_translation_only', False)
                    ),
                    init_rmse_gate_px=float(
                        smpl_refine_cfg.get('init_rmse_gate_px', 50.0)
                    ),
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
                print(f"[SMPL Refine] Done. Per-iter RMSE (px): "
                      f"{dbg.get('iter_rmse_pixels')}")
                if not dbg.get('enabled', True):
                    print(f"[SMPL Refine] ABORTED: {dbg.get('reason')}")
                if 'init_rmse_pixels' in dbg:
                    print(f"[SMPL Refine] Initial RMSE (px) under cam_R/T: "
                          f"{dbg['init_rmse_pixels']:.2f}")

                # Persist debug
                debug_dir = os.path.join(img_folder, '..', 'droid_debug')
                try:
                    os.makedirs(debug_dir, exist_ok=True)
                    with open(os.path.join(debug_dir, 'smpl_pose_refine_debug.json'), 'w') as f:
                        json.dump({
                            'config': dict(smpl_refine_cfg),
                            'num_keyframes_used': len(kf_in_range),
                            'intrinsics_fx_fy_cx_cy': [fx, fy, cx, cy],
                            'debug': dbg,
                        }, f, indent=2)
                except Exception as e:
                    print(f"[SMPL Refine] Failed to write debug json: {e}")

    # ------------------------------------------------------------------
    # Optional: Adjacent-SMPL projection refinement.
    # This uses the same local (center <- source) SMPL sequence that was
    # rendered into the keyframe image, simulated by the previous/reference
    # camera trajectory. It is expected to pull the new trajectory towards
    # that reference in a local multi-view reprojection sense.
    # ------------------------------------------------------------------
    if (adjacent_projection_refine_cfg is not None
            and adjacent_projection_refine_cfg.get('enabled', False)):
        if adjacent_projection_info is None or ref_cam_R is None or ref_cam_T is None:
            print(
                "[Adjacent SMPL Refine] Skipped: missing projection_info or ref_cam_R/T."
            )
        else:
            if refine_intrinsics is not None:
                fx, fy, cx, cy = [float(v) for v in refine_intrinsics]
            elif calib is not None:
                fx, fy, cx, cy = [float(v) for v in calib[:4]]
            else:
                fx = fy = 1000.0
                cx = cy = 0.0
                print("[Adjacent SMPL Refine] WARNING: no intrinsics, using defaults")

            from .adjacent_projection_refine import refine_camera_with_adjacent_smpl_projection

            debug_dir = os.path.join(img_folder, '..', 'droid_debug')
            debug_path = os.path.join(debug_dir, 'adjacent_projection_refine_debug.json')
            print(
                f"[Adjacent SMPL Refine] Running on adjacent projection factors "
                f"(weight={adjacent_projection_refine_cfg.get('weight', 1.0)}, "
                f"iters={adjacent_projection_refine_cfg.get('num_iters', 5)})"
            )
            pred_cam_r, pred_cam_t, dbg = refine_camera_with_adjacent_smpl_projection(
                cam_R=pred_cam_r,
                cam_T=pred_cam_t,
                ref_R=ref_cam_R,
                ref_T=ref_cam_T,
                projection_info=adjacent_projection_info,
                intrinsics=(fx, fy, cx, cy),
                weight=float(adjacent_projection_refine_cfg.get('weight', 1.0)),
                pose_prior_weight=float(
                    adjacent_projection_refine_cfg.get('pose_prior_weight', 1.0)
                ),
                smooth_weight=float(adjacent_projection_refine_cfg.get('smooth_weight', 0.0)),
                huber_delta=float(adjacent_projection_refine_cfg.get('huber_delta', 8.0)),
                num_iters=int(adjacent_projection_refine_cfg.get('num_iters', 5)),
                damping=float(adjacent_projection_refine_cfg.get('damping', 1e-3)),
                max_points_per_frame=int(
                    adjacent_projection_refine_cfg.get('max_points_per_frame', 256)
                ),
                init_rmse_gate_px=float(
                    adjacent_projection_refine_cfg.get('init_rmse_gate_px', 300.0)
                ),
                update_clip=float(adjacent_projection_refine_cfg.get('update_clip', 0.2)),
                device='cuda' if torch.cuda.is_available() else 'cpu',
                debug_path=debug_path,
            )
            print(f"[Adjacent SMPL Refine] Done: {dbg}")

    if return_keyframe_info:
        return pred_cam_r, pred_cam_t, keyframe_info
    return pred_cam_r, pred_cam_t

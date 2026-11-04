import sys
sys.path.insert(0, 'thirdparty/DROID-SLAM/droid_slam')
sys.path.insert(0, 'thirdparty/DROID-SLAM')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import json
import os
from PIL import Image
from glob import glob
from torchvision.transforms import Resize

from droid import Droid
from .slam_utils import slam_args, parser
from .slam_utils import get_dimention, est_calib, image_stream, preprocess_masks, list_image_files
from .slam_utils import zoedepth_load_compat, zoedepth_midas_beit_compat
from .est_scale import est_scale_hybrid
from ..utils.rotation_conversions import quaternion_to_matrix

# 设置 multiprocessing start method，如果已经设置过则跳过
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    # Context 已经被设置，使用当前的 context
    pass


def run_metric_slam(img_folder, masks=None, calib=None, is_static=False, return_keyframe_info=False):
    '''
    Input:
        img_folder: directory that contain image files 
        masks: list or array of 2D masks for human. 
               If None, no masking applied during slam.
        calib: camera intrinsics [fx, fy, cx, cy]. 
               If None, will be naively estimated.
        return_keyframe_info: if True, additionally return a dict with keyframe
               timestamps, SE3 poses, and low-res disparities for warm start.
    '''

    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

    ##### If static camera, simply return static camera motion #####
    if is_static:
        pred_cam_t = torch.zeros([len(imgfiles), 3])
        pred_cam_r = torch.eye(3).expand(len(imgfiles), 3, 3)

        if return_keyframe_info:
            return pred_cam_r, pred_cam_t, None
        return pred_cam_r, pred_cam_t

    ##### Masked droid slam #####
    droid, traj = run_slam(img_folder, masks=masks, calib=calib)
    if droid is None:
        raise RuntimeError(
            f"DROID-SLAM failed to initialize. No images found or processed in {img_folder}. "
            f"Check that the directory contains jpg/jpeg/png files."
        )
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]

    # Extract keyframe info for warm start before deleting droid
    keyframe_info = None
    if return_keyframe_info:
        keyframe_info = {
            'keyframe_tstamps': tstamp.copy(),                          # [n] int array
            'keyframe_poses_se3': droid.video.poses[:n].cpu().numpy(),  # [n, 7] SE3
            'keyframe_disps': droid.video.disps[:n].cpu().numpy(),      # [n, h//8, w//8]
        }

    del droid
    torch.cuda.empty_cache()

    ##### Estimate Metric Depth #####
    from .zoedepth import load_zoedepth_model
    model_zoe_n = load_zoedepth_model()
    model_zoe_n = model_zoe_n.to('cuda')

    pred_depths = []
    H, W = get_dimention(img_folder)
    for t in tqdm(tstamp):
        img = cv2.imread(imgfiles[t])[:,:,::-1]
        img = cv2.resize(img, (W, H))
        
        img_pil = Image.fromarray(img)
        pred_depth = model_zoe_n.infer_pil(img_pil)
        pred_depths.append(pred_depth)

    ##### Estimate Metric Scale #####
    scales_ = []
    n = len(tstamp)   # for each keyframe
    for i in tqdm(range(n)):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
        slam_depth = 1/disp
        
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

    # 释放 ZoeDepth 模型
    del model_zoe_n
    del pred_depths
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # convert to metric-scale camera extrinsics: R_wc, T_wc
    pred_cam_t = torch.tensor(traj[:, :3]) * scale
    pred_cam_q = torch.tensor(traj[:, 3:])
    pred_cam_r = quaternion_to_matrix(pred_cam_q[:,[3,0,1,2]])

    if return_keyframe_info:
        return pred_cam_r, pred_cam_t, keyframe_info
    return pred_cam_r, pred_cam_t


def run_slam(imagedir, masks=None, calib=None, depth=None, record_debug=True):
    """ Maksed DROID-SLAM """
    droid = None
    if calib is None:
        calib = est_calib(imagedir)

    if masks is not None:
        img_msks, conf_msks = preprocess_masks(imagedir, masks)

    for (t, image, intrinsics) in tqdm(image_stream(imagedir, calib)):

        if droid is None:
            slam_args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(slam_args)
            if record_debug:
                droid.enable_recording(True)
        
        if masks is not None:
            img_msk = img_msks[t]
            conf_msk = conf_msks[t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=conf_msk)
        else:
            droid.track(t, image, intrinsics=intrinsics, depth=depth, mask=None)  

    if droid is None:
        return None, None

    # save debug info before terminate (terminate deletes frontend)
    debug_info = None
    if record_debug:
        debug_info = droid.get_debug_info()

    traj = droid.terminate(image_stream(imagedir, calib))

    # save debug info to file
    if debug_info is not None:
        # add final keyframe tstamps after backend
        n = droid.video.counter.value
        debug_info['num_keyframes_after_backend'] = n
        debug_info['keyframe_tstamps_after_backend'] = droid.video.tstamp.cpu().numpy()[:n].tolist()

        debug_dir = os.path.join(imagedir, '..', 'droid_debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, 'droid_frontend_debug.json')
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        print(f"[DROID Debug] Saved frontend debug info to {debug_path}")

    return droid, traj


def search_focal_length(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # default estimate
    calib = np.array(est_calib(img_folder))
    best_focal = calib[0]
    best_err = test_slam(img_folder, input_msks, 
                         stride=stride, calib=calib, max_frame=max_frame)
    
    # search based on slam reprojection error
    for focal in range(low, high, step):
        calib[:2] = focal
        err = test_slam(img_folder, input_msks, 
                        stride=stride, calib=calib, max_frame=max_frame)

        if err < best_err:
            best_err = err
            best_focal = focal

    return best_focal


def calibrate_intrinsics(img_folder, masks=None, stride=10, max_frame=50,
                        low=500, high=1500, step=100, is_static=False):
    """ Search for a good focal length by SLAM reprojection error """
    if masks is not None:
        masks = masks[::stride]
        masks = masks[:max_frame]
        img_msks, conf_msks = preprocess_masks(img_folder, masks)
        input_msks = (img_msks, conf_msks)
    else:
        input_msks = None

    # User indicate it's static camera
    if is_static:
        calib = np.array(est_calib(img_folder))
        return calib, is_static
    
    # Estimate camera whether static and intrinsics
    else:
        calib = np.array(est_calib(img_folder))
        best_focal = calib[0]
        best_err, is_static = test_slam(img_folder, input_msks, 
                                        stride=stride, calib=calib, max_frame=max_frame)
        if is_static:
            print('The camera is likely static.')
            return calib, is_static
        
        for focal in range(low, high, step):
            calib[:2] = focal
            err, is_static = test_slam(img_folder, input_msks, 
                                        stride=stride, calib=calib, max_frame=max_frame)
            if err < best_err:
                best_err = err
                best_focal = focal
        calib[0] = calib[1] = best_focal

        return calib, is_static


def test_slam(imagedir, masks, calib, stride=10, max_frame=50):
    """ Shorter SLAM step to test reprojection error """
    args = parser.parse_args([])
    args.stereo = False
    args.upsample = False
    args.disable_vis = True
    args.frontend_window = 10
    args.frontend_thresh = 10
    droid = None

    for (t, image, intrinsics) in image_stream(imagedir, calib, stride, max_frame):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        if masks is not None:
            img_msk = masks[0][t]
            conf_msk = masks[1][t]
            image = image * (img_msk < 0.5)
            droid.track(t, image, intrinsics=intrinsics, mask=conf_msk)  
        else:
            droid.track(t, image, intrinsics=intrinsics, mask=None)  
    
    if droid.video.counter.value <= 1:
        # If less than 2 keyframes, likely static camera
        static_camera = True
        reprojection_error = None
    else:
        static_camera = False
        reprojection_error = droid.compute_error()

    del droid

    return reprojection_error, static_camera



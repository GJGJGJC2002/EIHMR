import numpy as np
import torch
import cv2
from glob import glob
import argparse

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from torchvision.transforms import Resize

# Some default settings for DROID-SLAM
parser = argparse.ArgumentParser()
parser.add_argument("--imagedir", type=str, help="path to image directory")
parser.add_argument("--calib", type=str, help="path to calibration file")
parser.add_argument("--t0", default=0, type=int, help="starting frame")
parser.add_argument("--stride", default=1, type=int, help="frame stride")

parser.add_argument("--weights", default="data/pretrain/droid.pth")
parser.add_argument("--buffer", type=int, default=512)
parser.add_argument("--image_size", default=[240, 320])
parser.add_argument("--disable_vis", action="store_true")

parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

parser.add_argument("--backend_thresh", type=float, default=22.0)
parser.add_argument("--backend_radius", type=int, default=2)
parser.add_argument("--backend_nms", type=int, default=3)
parser.add_argument("--upsample", action="store_true")
parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
slam_args = parser.parse_args([])
slam_args.stereo = False
slam_args.upsample = True
slam_args.disable_vis = True


def list_image_files(imagedir):
    """List image files supported by DROID input helpers."""
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    imgfiles = []
    for pat in patterns:
        imgfiles.extend(glob(f'{imagedir}/{pat}'))
    return sorted(set(imgfiles))


def _read_first_image(imagedir):
    imgfiles = list_image_files(imagedir)
    if not imgfiles:
        raise FileNotFoundError(
            f"No image files found in {imagedir}. Supported extensions: jpg/jpeg/png"
        )
    image = cv2.imread(imgfiles[0])
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {imgfiles[0]}")
    return image


def est_calib(imagedir):
    """ Roughly estimate intrinsics by image dimensions """
    image = _read_first_image(imagedir)

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def get_dimention(imagedir):
    """ Get proper image dimension for DROID """
    image = _read_first_image(imagedir)

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1%8, :w1-w1%8]
    H, W, _ = image.shape
    return H, W


def image_stream(imagedir, calib, stride=1, max_frame=None):
    """ Image generator for DROID """
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = list_image_files(imagedir)
    image_list = image_list[::stride]
    if max_frame is not None:
        image_list = image_list[:max_frame]

    for t, imfile in enumerate(image_list):
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

        yield t, image[None], intrinsics


def preprocess_masks(img_folder, masks):
    """ Resize masks for masked droid """
    H, W = get_dimention(img_folder)
    resize_1 = Resize((H, W), antialias=True)
    resize_2 = Resize((H//8, W//8), antialias=True)
    
    img_msks = []
    for i in range(0, len(masks), 500):
        m = resize_1(masks[i:i+500])
        img_msks.append(m)
    img_msks = torch.cat(img_msks)

    conf_msks = []
    for i in range(0, len(masks), 500):
        m = resize_2(masks[i:i+500])
        conf_msks.append(m)
    conf_msks = torch.cat(conf_msks)

    return img_msks, conf_msks


def eval_slam(traj_est, cam_t, cam_q, return_traj=True, correct_scale=False, align=True, align_origin=False):
    """ Evaluation for SLAM """
    tstamps = np.array([i for i in range(len(traj_est))], dtype=np.float32)

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3], 
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=tstamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=cam_t.copy(),
        orientations_quat_wxyz=cam_q.copy(),
        timestamps=tstamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=align, align_origin=align_origin,
        correct_scale=correct_scale)
    
    stats = result.stats

    if return_traj:
        return stats, traj_ref, traj_est

    return stats


# -------------------- ZoeDepth load 兼容 --------------------
#
# 新版 timm 的 BEiT 去掉了 `attn.relative_position_index` 这个 buffer，
# 但 ZoeDepth 官方预训练权重里仍带 24 层这个 key，
# 导致 model.load_state_dict(..., strict=True) 报 "Unexpected key(s)".
#
# 用 contextmanager 形式 patch torch.nn.Module.load_state_dict：进入块时拦截、
# 退出时恢复，调用方只需：
#   from lib.camera.slam_utils import zoedepth_load_compat, zoedepth_midas_beit_compat
#   with zoedepth_load_compat():
#       model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
#   zoedepth_midas_beit_compat(model_zoe_n)   # 兼容新版 timm 的 BEiT Block 属性名
from contextlib import contextmanager


@contextmanager
def zoedepth_load_compat():
    """临时 patch Module.load_state_dict，过滤掉 *.relative_position_index 等
    由 timm 版本升级导致的 unexpected keys，加载完成后自动还原。"""
    import torch.nn as _nn
    _orig = _nn.Module.load_state_dict

    def _patched(self, state_dict, strict=True, *args, **kwargs):
        if isinstance(state_dict, dict):
            bad = [k for k in list(state_dict.keys())
                   if k.endswith('.attn.relative_position_index')
                   or k.endswith('.relative_position_index')]
            if bad:
                for k in bad:
                    state_dict.pop(k, None)
                strict = False  # 额外保护：允许其他小差异
        return _orig(self, state_dict, strict=strict, *args, **kwargs)

    _nn.Module.load_state_dict = _patched
    try:
        yield
    finally:
        _nn.Module.load_state_dict = _orig


def zoedepth_midas_beit_compat(model):
    """
    兼容新版 timm 的 BEiT Block：
      - 旧 API：block.drop_path, block.gamma_1, block.gamma_2
      - 新 API：block.drop_path1/drop_path2, block.ls1.gamma/block.ls2.gamma
    MiDaS 的 midas/backbones/beit.py::block_forward 用的是旧 API，新 timm 会
    抛 AttributeError: 'Block' object has no attribute 'drop_path'。
    这里遍历整个模型，给每个 Block 动态加"别名"指向新属性。
    """
    import torch.nn as nn

    def _resolve_gamma(block, idx):
        # 新 timm：block.ls1.gamma / block.ls2.gamma （LayerScale）
        ls_name = f'ls{idx}'
        ls = getattr(block, ls_name, None)
        if ls is not None and hasattr(ls, 'gamma'):
            return ls.gamma
        # 兜底：没 LayerScale 就返回 None（由上层决定是否创建标量 1）
        return None

    def _resolve_drop_path(block, idx):
        # 新 timm：block.drop_path1 / block.drop_path2
        dp = getattr(block, f'drop_path{idx}', None)
        if dp is not None:
            return dp
        return nn.Identity()

    patched = 0
    for m in model.modules():
        cls_name = m.__class__.__name__
        # 只处理 timm 的 Block 类（BEiT/ViT 系列），避免误伤
        if cls_name != 'Block':
            continue

        need_patch = False
        if not hasattr(m, 'drop_path'):
            # 直接挂一个 callable 属性——新版 Block 的 forward 不会用到，
            # 只有 MiDaS 的 block_forward 会用；那里两个 drop_path 调用共享
            # 同一个 attr，我们用 drop_path1 作为主要 fallback
            setattr(m, 'drop_path', _resolve_drop_path(m, 1))
            need_patch = True
        if not hasattr(m, 'gamma_1'):
            g1 = _resolve_gamma(m, 1)
            if g1 is not None:
                setattr(m, 'gamma_1', g1)
                need_patch = True
        if not hasattr(m, 'gamma_2'):
            g2 = _resolve_gamma(m, 2)
            if g2 is not None:
                setattr(m, 'gamma_2', g2)
                need_patch = True
        if need_patch:
            patched += 1

    return patched

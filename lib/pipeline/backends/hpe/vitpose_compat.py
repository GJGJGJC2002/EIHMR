"""
Compatibility shim: 用 PromptHMR 自带的 25 关节 ViTPose 替代 GVHMR 硬编码的
17 关节 `vitpose-h-multi-coco.pth`（本地未下载）。

接口保持和 `hmr4d.utils.preproc.vitpose.VitPoseExtractor` 一致：
    extractor = VitPoseExtractorCompat(tqdm_leave=False)
    kp2d = extractor.extract(imgs, bbx_xys)      # 返回 (F, 17, 3) — (x, y, conf)

内部做法：
  1) 基于 `bbx_xys`（中心+size，square crop 的定义）恢复 xyxy 框；
  2) 调 PromptHMR 的 `VitInference.inference_wbbox` 拿到 25 关节 (y, x, conf)；
  3) 把 (y, x) 转 (x, y)，再从 coco_25 映射为 coco_17 返回。

这样既不需要下载 `vitpose-h-multi-coco.pth`（2.4GB 17 关节版），也不需要改
PromptHMR 或 GVHMR 上游代码。

coco_25 vs coco_17 索引映射（coco_25 里的哪些 idx 组成 coco_17）:
    coco_17 idx: 0  1  2  3  4   5  6  7  8  9  10 11 12 13 14 15 16
    对应 coco_25: 0  1  2  3  4   6  7  8  9  10 11 12 13 15 16 17 18
    （跳过 coco_25 的 5=neck, 14=hip, 19..24=foot）
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm


# coco_25 -> coco_17 的索引映射
COCO25_TO_COCO17 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]


def _bbx_xys_to_xyxy(bbx_xys: torch.Tensor) -> np.ndarray:
    """
    GVHMR 约定：bbx_xys = (cx, cy, s)，其中 s 是 crop 正方形的边长。
    转成 xyxy。
    """
    if isinstance(bbx_xys, torch.Tensor):
        bbx_xys = bbx_xys.detach().cpu().numpy()
    cx, cy, s = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    half = s / 2.0
    xyxy = np.stack([cx - half, cy - half, cx + half, cy + half], axis=-1)
    return xyxy


class VitPoseExtractorCompat:
    """
    Drop-in 替代 GVHMR 的 VitPoseExtractor。
    使用 PromptHMR 自带的 25 关节 vitpose-h-coco_25.pth，返回前 17 个 coco 关节。
    """

    # 25 关节 ViTPose checkpoint 相对路径（相对 PromptHMR 项目根目录）
    DEFAULT_CKPT_REL = 'data/pretrain/vitpose-h-coco_25.pth'

    def __init__(self, tqdm_leave: bool = True, prompthmr_root: str = None,
                 ckpt_path: str = None):
        self.tqdm_leave = tqdm_leave

        # 推断 PromptHMR 项目根目录（以便 import pipeline.detector.*）
        if prompthmr_root is None:
            # 尝试环境变量 / 默认相对 tram 的 ../PromptHMR
            # __file__ = .../tram/lib/pipeline/backends/hpe/vitpose_compat.py
            # 向上 5 层到 Projects/，再拼 PromptHMR
            prompthmr_root = os.environ.get(
                'PROMPTHMR_ROOT',
                os.path.abspath(os.path.join(
                    os.path.dirname(__file__),
                    '..', '..', '..', '..', '..', 'PromptHMR'
                ))
            )
        if prompthmr_root not in sys.path:
            sys.path.insert(0, prompthmr_root)
        self.prompthmr_root = prompthmr_root

        # checkpoint 路径
        if ckpt_path is None:
            ckpt_path = os.path.join(prompthmr_root, self.DEFAULT_CKPT_REL)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f'ViTPose checkpoint not found: {ckpt_path}\n'
                f'Expected the 25-keypoint version from PromptHMR '
                f'(data/pretrain/vitpose-h-coco_25.pth).'
            )

        # 延迟 import 避免在无 PromptHMR 环境时直接崩。
        #
        # 注意：不能直接 `from pipeline.detector.vitpose_estimator import load_vit_model`：
        #   1) PromptHMR/pipeline/__init__.py 会链式 import pipeline.camera →
        #      masked_droid_slam → `from droid import Droid`，而 tram 的 EIHMR 环境
        #      里没有装 PromptHMR 专用的 droid_slam，会抛 ModuleNotFoundError。
        #   2) tram 自己也有一个顶级 `lib.pipeline` 包，直接让 PromptHMR 的 pipeline
        #      被 import 进 sys.modules 会污染 tram 后续对 lib.pipeline 的使用。
        #
        # 解决办法：在 sys.modules 里预置空的 `pipeline` / `pipeline.detector` /
        # `pipeline.detector.ViTPose` 命名空间包占位，用 importlib 直接按源文件
        # 加载 vitpose_estimator 与它依赖的 `ViTPose.easy_vitpose.inference`。
        self._model = self._load_vitpose_model_isolated(ckpt_path)

    @staticmethod
    def _load_vitpose_model_isolated(ckpt_path: str):
        """不触发 PromptHMR/pipeline/__init__.py 的前提下加载 load_vit_model。

        返回 VitInference 实例（与 pipeline.detector.vitpose_estimator.load_vit_model
        默认参数 model_name='h', is_video=False, single_pose=True, dataset='coco_25'
        等价）。
        """
        import importlib
        import importlib.util
        import types

        # 当前 vitpose_compat.py 的目录 → 推断 PromptHMR 根目录
        # （self.prompthmr_root 已经处理过，但这里用 ckpt_path 再反推更稳）
        # ckpt_path = <prompthmr_root>/data/pretrain/vitpose-h-coco_25.pth
        prompthmr_root = os.path.abspath(os.path.join(
            os.path.dirname(ckpt_path), '..', '..'
        ))

        detector_dir = os.path.join(prompthmr_root, 'pipeline', 'detector')
        vitpose_src = os.path.join(detector_dir, 'vitpose_estimator.py')
        easy_inference_src = os.path.join(
            detector_dir, 'ViTPose', 'easy_vitpose', 'inference.py'
        )
        if not os.path.isfile(vitpose_src):
            raise FileNotFoundError(f'vitpose_estimator.py not found: {vitpose_src}')
        if not os.path.isfile(easy_inference_src):
            raise FileNotFoundError(f'easy_vitpose/inference.py not found: {easy_inference_src}')

        # ---------- 1. 在 sys.modules 里占位空的 `pipeline` 系列包 ----------
        # 仅当尚未导入，或之前导入的不是 PromptHMR 版本时才占位；否则不改动。
        fake_pkg_names = [
            'pipeline',
            'pipeline.detector',
            'pipeline.detector.ViTPose',
            'pipeline.detector.ViTPose.easy_vitpose',
        ]
        created_fake = {}
        for pkg_name in fake_pkg_names:
            if pkg_name in sys.modules:
                continue
            mod = types.ModuleType(pkg_name)
            mod.__path__ = []  # 标记为 package
            # 为 detector / ViTPose / easy_vitpose 指定真实目录，供相对 import 使用
            parts = pkg_name.split('.')
            if pkg_name == 'pipeline.detector':
                mod.__path__ = [detector_dir]
            elif pkg_name == 'pipeline.detector.ViTPose':
                mod.__path__ = [os.path.join(detector_dir, 'ViTPose')]
            elif pkg_name == 'pipeline.detector.ViTPose.easy_vitpose':
                mod.__path__ = [os.path.join(detector_dir, 'ViTPose', 'easy_vitpose')]
            mod.__package__ = pkg_name
            sys.modules[pkg_name] = mod
            created_fake[pkg_name] = mod

        # 建立父子关系
        if 'pipeline' in created_fake and 'pipeline.detector' in created_fake:
            setattr(sys.modules['pipeline'], 'detector', sys.modules['pipeline.detector'])
        if ('pipeline.detector' in created_fake
                and 'pipeline.detector.ViTPose' in created_fake):
            setattr(sys.modules['pipeline.detector'], 'ViTPose',
                    sys.modules['pipeline.detector.ViTPose'])
        if ('pipeline.detector.ViTPose' in created_fake
                and 'pipeline.detector.ViTPose.easy_vitpose' in created_fake):
            setattr(sys.modules['pipeline.detector.ViTPose'], 'easy_vitpose',
                    sys.modules['pipeline.detector.ViTPose.easy_vitpose'])

        # ---------- 2. 按源文件加载 easy_vitpose.inference ----------
        inference_mod_name = 'pipeline.detector.ViTPose.easy_vitpose.inference'
        if inference_mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                inference_mod_name, easy_inference_src,
                submodule_search_locations=[os.path.dirname(easy_inference_src)],
            )
            inference_mod = importlib.util.module_from_spec(spec)
            sys.modules[inference_mod_name] = inference_mod
            spec.loader.exec_module(inference_mod)
            # 挂到父包
            setattr(sys.modules['pipeline.detector.ViTPose.easy_vitpose'],
                    'inference', inference_mod)

        # ---------- 3. 按源文件加载 vitpose_estimator ----------
        estimator_mod_name = 'pipeline.detector.vitpose_estimator'
        if estimator_mod_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                estimator_mod_name, vitpose_src,
                submodule_search_locations=[os.path.dirname(vitpose_src)],
            )
            estimator_mod = importlib.util.module_from_spec(spec)
            sys.modules[estimator_mod_name] = estimator_mod
            spec.loader.exec_module(estimator_mod)
            setattr(sys.modules['pipeline.detector'],
                    'vitpose_estimator', estimator_mod)
        else:
            estimator_mod = sys.modules[estimator_mod_name]

        load_vit_model = estimator_mod.load_vit_model
        return load_vit_model(
            model_path=ckpt_path,
            model_name='h',
            is_video=False,
            single_pose=True,
        )

    @torch.no_grad()
    def extract(self, imgs, bbx_xys) -> torch.Tensor:
        """
        Args:
            imgs: 和 GVHMR `get_batch` 的输出对应，shape (F, 3, H, W) float tensor
                  （已经做了归一化或尺寸对齐）。此处只需要拿 RGB uint8 的原始图像，
                  所以调用方应传 `get_batch` 后的 bbx_xys（已对应缩放后的 imgs）。
            bbx_xys: (F, 3) — (cx, cy, s) on the imgs coordinate system.

        Returns:
            vitpose: (F, 17, 3) float tensor — (x, y, conf) in imgs 坐标系。
        """
        # imgs 可能是 tensor (F, 3, H, W) 或 ndarray (F, H, W, 3)
        if isinstance(imgs, torch.Tensor):
            if imgs.dim() == 4 and imgs.shape[1] == 3:
                imgs_np = imgs.detach().cpu().permute(0, 2, 3, 1).contiguous().numpy()
            else:
                imgs_np = imgs.detach().cpu().numpy()
        else:
            imgs_np = np.asarray(imgs)

        # 归一化到 uint8 RGB（PromptHMR vitpose_estimator 期望的格式）
        if imgs_np.dtype != np.uint8:
            # 可能是 [0,1] 浮点 → 乘 255；或 [0,255] 浮点 → 直接 cast
            if imgs_np.max() <= 1.5:
                imgs_np = (imgs_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                imgs_np = imgs_np.clip(0, 255).astype(np.uint8)

        xyxy = _bbx_xys_to_xyxy(bbx_xys)  # (F, 4)
        F_, H, W = imgs_np.shape[:3]
        assert xyxy.shape[0] == F_, \
            f'bbx_xys has {xyxy.shape[0]} frames but imgs has {F_}'

        out = np.zeros((F_, 17, 3), dtype=np.float32)
        for i in tqdm(range(F_), desc='ViTPose (25→17 compat)', leave=self.tqdm_leave):
            img = imgs_np[i]
            bb = xyxy[i].copy()

            # clip 到图像范围，并防止无效 bbox 产生空 crop
            bb[0] = max(0.0, bb[0]); bb[1] = max(0.0, bb[1])
            bb[2] = min(float(W - 1), bb[2]); bb[3] = min(float(H - 1), bb[3])
            if bb[2] <= bb[0] + 1 or bb[3] <= bb[1] + 1:
                bb = np.array([0.0, 0.0, float(W - 1), float(H - 1)], dtype=np.float32)

            kp25, _img = self._model.inference_wbbox(img, bb)
            # kp25 shape: (25, 3) — (y, x, conf) 格式（inference_wbbox 的 `keypoints[:, :2]` 是 [y,x]）
            # 转成 (x, y, conf)
            kp25_xy = np.stack([kp25[:, 1], kp25[:, 0], kp25[:, 2]], axis=-1)
            out[i] = kp25_xy[COCO25_TO_COCO17]

        return torch.from_numpy(out)


__all__ = ['VitPoseExtractorCompat']

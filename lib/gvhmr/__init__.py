"""Minimal GVHMR body-model, kinematics and rendering code used by EIHMR (vendored; see README.md, License).

SMPL/SMPL-X model files are licensed separately and are not shipped: point ``EIHMR_BODY_MODELS`` (or
:func:`set_body_model_root`) at a directory containing ``smplx/SMPLX_NEUTRAL.npz`` and ``smpl/SMPL_NEUTRAL.pkl``.
"""
import os
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / 'assets'
_BODY_MODEL_ROOT = None


def set_body_model_root(path):
    global _BODY_MODEL_ROOT
    _BODY_MODEL_ROOT = Path(path)


def body_model_root():
    root = _BODY_MODEL_ROOT or (Path(os.environ['EIHMR_BODY_MODELS']) if 'EIHMR_BODY_MODELS' in os.environ else None)
    if root is None:
        raise RuntimeError('SMPL/SMPL-X model directory not configured: set EIHMR_BODY_MODELS or call '
                           'lib.gvhmr.set_body_model_root()')
    if not (root / 'smplx' / 'SMPLX_NEUTRAL.npz').is_file() or not (root / 'smpl' / 'SMPL_NEUTRAL.pkl').is_file():
        raise FileNotFoundError(f'{root} must contain smplx/SMPLX_NEUTRAL.npz and smpl/SMPL_NEUTRAL.pkl')
    return root

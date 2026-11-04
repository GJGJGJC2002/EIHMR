import copy
import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .core.data import CameraParams


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_npz(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_trusted_pickle(path):
    path = Path(path)
    with path.open('rb') as stream:
        compressed = stream.read(2) == b'\x1f\x8b'
    with patch('torch.storage._load_from_bytes',
               lambda value: torch.load(io.BytesIO(value), map_location='cpu', weights_only=False)):
        with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
            return pickle.load(stream)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def camera_signature(camera):
    value = hashlib.sha256()
    for key, array in sorted(camera.to_dict().items()):
        array = np.ascontiguousarray(array)
        value.update(key.encode())
        value.update(str((array.shape, array.dtype.str)).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def apply_camera(data, R, T):
    camera = copy.copy(data.camera_params)
    camera.R = torch.as_tensor(R).detach().cpu().numpy().astype(np.float32)
    camera.T = torch.as_tensor(T).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(camera.R).all() or not np.isfinite(camera.T).all():
        raise ValueError('Non-finite camera trajectory')
    camera.world_R, camera.world_T = camera.R, camera.T
    data.camera_params = camera
    return data


class CameraCheckpoint:
    def __init__(self, directory, stage, identity):
        self.directory = Path(directory)
        self.stage = stage
        self.identity = signature(identity)
        self.camera_path = self.directory / f'{stage}_camera.npz'
        self.marker_path = self.directory / f'{stage}_complete.json'

    def load(self, data):
        if not self.marker_path.exists():
            return None
        marker = json.loads(self.marker_path.read_text())
        if marker['identity'] != self.identity:
            raise ValueError(f'Stale {self.stage} checkpoint: input/config/source changed')
        if not self.camera_path.is_file() or digest(self.camera_path) != marker['camera_sha256']:
            raise ValueError(f'Corrupt {self.stage} checkpoint')
        camera = CameraParams.from_dict(read_npz(self.camera_path))
        if camera.T.shape != data.camera_params.T.shape:
            raise ValueError('Checkpoint frame count changed')
        data.camera_params = camera
        return marker['debug']

    def save(self, data, debug):
        write_npz(self.camera_path, data.camera_params.to_dict())
        write_json(self.marker_path, {'identity': self.identity,
                   'camera_sha256': digest(self.camera_path), 'debug': debug})

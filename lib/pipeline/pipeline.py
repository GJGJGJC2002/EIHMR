import copy
import importlib
import json
from pathlib import Path

import torch

from .checkpoint import digest, load_trusted_pickle, signature, write_json
from .core.data import CameraParams, PipelineData, SMPLParams


COMPONENTS = {
    'segmentation': ('segmentation', 'SegmentationComponent'),
    'slam': ('slam', 'SLAMComponent'),
    'hpe': ('hpe', 'HPEComponent'),
    'depth_scene_refine': ('depth_scene_refine', 'DepthSceneRefineComponent'),
    'adjacent_smpl_renderer': ('adjacent_smpl_renderer', 'AdjacentSMPLRenderer'),
    'camera_refine': ('camera_refine', 'CameraRefineComponent'),
    'evaluation': ('evaluation', 'EvaluationComponent'),
}


def create_component(config):
    module, name = COMPONENTS[config['type']]
    cls = getattr(importlib.import_module('lib.pipeline.components.' + module), name)
    return cls(config['name'], config)


def video_component_config(config, data, frontend_enabled):
    if data.metadata.get('inference_mode') != 'video' or not frontend_enabled:
        return config
    if config['type'] not in ('depth_scene_refine', 'adjacent_smpl_renderer'):
        return config
    import numpy as np
    route = data.metadata.get('second_pass_route', {})
    info = data.metadata.get('slam_keyframe_info')
    if route.get('mode') == 'uniform_keyframes':
        raw = np.asarray(route['forced_keyframes'])
    else:
        if not info or 'keyframe_tstamps' not in info:
            raise ValueError('Native DROID keyframe information is missing')
        raw = np.asarray(info['keyframe_tstamps'])
    if raw.ndim != 1 or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
        raise ValueError('DROID keyframe timestamps must be integer frame indices')
    centers = raw.astype(np.int64)
    n = len(data.image_paths)
    if np.any(centers < 0) or np.any(centers >= n) or np.any(np.diff(centers) <= 0):
        raise ValueError('DROID keyframe indices must be ordered, unique and within video')
    config = copy.deepcopy(config)
    if config['type'] == 'depth_scene_refine':
        uniform = range(0, n, int(config.get('stride', 5)))
        config['window_centers'] = sorted(set(uniform) | set(centers.tolist()))
    else:
        spacing = route['keyframe_stride'] if route.get('mode') == 'uniform_keyframes' else 1
        config.update(render_frame_indices=centers.tolist(), pre_dis=spacing, post_dis=spacing)
    return config


class Pipeline:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)

    def execute(self, data, start_from='segmentation'):
        started = False
        data.metadata.update(output_dir=str(self.output), pipeline_name='EIHMR')
        for config in self.config['components']:
            kind = config['type']
            if kind == start_from:
                started = True
            if not started:
                continue
            camera_config = next(c for c in self.config['components'] if c['type'] == 'camera_refine')
            frontend = camera_config['frontend']
            routing = camera_config.get('routing', {})
            if (data.metadata.get('inference_mode') == 'video' and frontend.get('enabled', True)
                    and routing.get('enabled', False) and kind not in ('segmentation', 'slam')
                    and 'second_pass_route' not in data.metadata):
                from .video import select_second_pass
                data.metadata['second_pass_route'] = select_second_pass(data.camera_params, len(data.image_paths), routing)
                write_json(self.output / 'second_pass_route.json', data.metadata['second_pass_route'])
            if kind == 'adjacent_smpl_renderer' and not frontend.get('enabled', True):
                continue
            config = video_component_config(config, data, frontend.get('enabled', True))
            if (kind == 'adjacent_smpl_renderer' and data.metadata.get('inference_mode') == 'video'
                    and len(config['render_frame_indices']) < int(frontend.get('min_common_keyframes', 4))):
                continue
            component = create_component(config)
            try:
                if kind in ('camera_refine', 'evaluation'):
                    data = component(data)
                else:
                    directory = self.output / 'checkpoints' / data.sequence_name
                    marker = directory / f'{kind}_complete.json'
                    cache = directory / f'{kind}_data.pkl'
                    identity = signature({'input': data.metadata['input_fingerprint'], 'config': config,
                                          'source': digest(Path(importlib.import_module(component.__module__).__file__))})
                    if marker.exists():
                        stored = json.loads(marker.read_text())
                        if stored['identity'] != identity or digest(cache) != stored['cache_sha256']:
                            raise ValueError(f'Stale/corrupt {kind} checkpoint')
                        if kind == 'adjacent_smpl_renderer' and data.metadata.get('inference_mode') == 'video':
                            for path, expected in stored.get('render_hashes', {}).items():
                                if not Path(path).is_file() or digest(path) != expected:
                                    raise ValueError(f'Missing/changed rendered checkpoint image: {path}')
                        restored = load_trusted_pickle(cache)
                        data.camera_params = CameraParams.from_dict(restored['camera_params']) if restored['camera_params'] else None
                        data.smpl_params = SMPLParams.from_dict(restored['smpl_params']) if restored['smpl_params'] else None
                        data.masks = torch.as_tensor(restored['masks']).cpu() if restored['masks'] is not None else None
                        preserved = {key: data.metadata[key] for key in
                                     ('stop_after', 'input_fingerprint', 'output_dir', 'second_pass_route') if key in data.metadata}
                        data.metadata.update(restored['metadata'])
                        data.metadata.update(preserved)
                    else:
                        data = component(data)
                        directory.mkdir(parents=True, exist_ok=True)
                        temporary = cache.with_name(cache.name + '.tmp')
                        data.save(str(temporary))
                        temporary.replace(cache)
                        record = {'identity': identity, 'cache_sha256': digest(cache)}
                        if kind == 'adjacent_smpl_renderer' and data.metadata.get('inference_mode') == 'video':
                            render = Path(data.metadata['adjacent_render_info']['output_dir'])
                            paths = [render / Path(data.image_paths[t]).name for t in config['render_frame_indices']]
                            if any(not p.is_file() for p in paths):
                                raise ValueError('Renderer did not produce all native keyframe images')
                            record['render_hashes'] = {str(p): digest(p) for p in paths}
                        write_json(marker, record)
            finally:
                component.cleanup()
            if data.should_stop:
                break
        return data

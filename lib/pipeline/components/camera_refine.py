from pathlib import Path

import numpy as np
import torch

from lib.pipeline.checkpoint import CameraCheckpoint, apply_camera, camera_signature, digest
from lib.pipeline.core.component import Component


class CameraRefineComponent(Component):
    COMPONENT_TYPE = 'camera_refine'

    def setup(self):
        self._is_setup = True

    def validate_input(self, data):
        return data.camera_params is not None and bool(data.metadata.get('oar_window_mesh_observations'))

    def execute(self, data):
        from lib.camera.human_mesh_camera_refine import refine_camera_with_window_mesh_consistency

        directory = Path(data.metadata['output_dir']) / 'checkpoints' / data.sequence_name
        source = Path(__file__).resolve()
        identity = {'input': data.metadata['input_fingerprint'],
                    'camera': camera_signature(data.camera_params),
                    'config': self.config, 'component': digest(source),
                    'frontend_source': digest(source.parents[2] / 'camera/frontend_delta.py'),
                    'human_source': digest(source.parents[2] / 'camera/human_mesh_camera_refine.py')}
        frontend_config = dict(self.config.get('frontend', {}))
        frontend_enabled = frontend_config.pop('enabled', True)
        route = data.metadata.get('second_pass_route', {}) if data.metadata.get('inference_mode') == 'video' else {}
        identity['second_pass_route'] = route
        front = CameraCheckpoint(directory, 'frontdelta', identity)
        debug = front.load(data)
        if debug is None:
            if frontend_enabled:
                debug = self._frontdelta(data, route, frontend_config, directory)
            else:
                debug = {'enabled': False, 'reason': 'disabled_by_config'}
            front.save(data, debug)
        data.metadata['frontend_delta_slam'] = debug
        if data.metadata.get('stop_after') == 'frontdelta':
            data.should_stop = True
            return data

        identity['frontdelta_camera'] = camera_signature(data.camera_params)
        human = CameraCheckpoint(directory, 'camera_refine', identity)
        debug = human.load(data)
        if debug is None:
            config = dict(self.config['human'])
            R, T, debug = refine_camera_with_window_mesh_consistency(
                data.camera_params.R, data.camera_params.T,
                data.metadata['oar_window_mesh_observations'], device=self.device,
                debug_path=str(directory / 'camera_refine_debug.json'), **config)
            if not debug.get('enabled'):
                raise RuntimeError(f'Human camera refine failed: {debug.get("reason")}')
            apply_camera(data, R, T)
            human.save(data, debug)
        data.metadata['human_mesh_camera_refine'] = debug
        return data

    def _frontdelta(self, data, route, config, directory):
        """Frontend-only rendered-keyframe delta, anchored to the first-pass camera.

        In video mode (``route['mode'] == 'uniform_keyframes'``) the keyframes are a uniform grid:
        a metric SLAM pass on that grid is pulled back onto the first-pass camera as clamped
        per-keyframe deltas, and its keyframe state warm-starts the rendered frontdelta.  Without a
        route the first-pass DROID keyframes are used.  Any failure keeps the incoming camera.
        """
        from lib.camera.frontend_delta import pull_back_uniform_keyframes, refine_camera_with_frontend_delta

        images = str(Path(data.image_paths[0]).parent)
        masks = torch.as_tensor(data.masks).cpu() if data.masks is not None else None
        K = np.asarray(data.camera_params.intrinsics)
        intrinsics = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        info = data.metadata.get('slam_keyframe_info')
        debug = {}
        if route.get('mode') == 'uniform_keyframes':
            R, T, info, debug['uniform_keyframes'] = pull_back_uniform_keyframes(
                cam_R=data.camera_params.R, cam_T=data.camera_params.T, forced_keyframes=route['forced_keyframes'],
                original_image_dir=images, masks=masks, intrinsics=intrinsics,
                max_translation_delta=config['max_translation_delta'],
                max_rotation_delta_deg=config['max_rotation_delta_deg'],
                local_scale=route.get('local_metric_scale'), debug_path=str(directory / 'uniform_keyframes_debug.json'))
            if info is not None:
                apply_camera(data, R, T)
                data.metadata['slam_keyframe_info'] = info
        render_dir = data.metadata.get('adjacent_render_info', {}).get('output_dir')
        if not info or not render_dir:
            debug.update(enabled=False, reason='missing_keyframes_or_rendered_images', fallback_to_baseline=True)
        else:
            rendered = {int(p.stem) for p in Path(render_dir).iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and p.stem.isdigit()}
            missing = sorted({int(t) for t in info['keyframe_tstamps']} - rendered)
            if missing:
                raise ValueError(f'Rendered images do not cover first-pass keyframes, e.g. {missing[:5]}')
            R, T, result = refine_camera_with_frontend_delta(
                cam_R=data.camera_params.R, cam_T=data.camera_params.T, keyframe_info=info, render_dir=render_dir,
                original_image_dir=images, masks=masks, intrinsics=intrinsics,
                debug_path=str(directory / 'frontdelta_debug.json'), **config)
            debug.update(result)
            if result.get('enabled'):
                apply_camera(data, R, T)
            else:
                debug['fallback_to_baseline'] = True
        debug['second_pass_route'] = route
        return debug

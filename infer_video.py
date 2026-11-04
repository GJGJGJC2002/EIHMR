import argparse
import fcntl
import logging
import os
from pathlib import Path

import numpy as np

from lib.pipeline.checkpoint import digest, signature, write_json
from lib.pipeline.core.data import CameraParams, PipelineData
from lib.pipeline.pipeline import Pipeline
from lib.pipeline.video import asset_fingerprint, decode_video, track_person, video_config
from lib.trajectories import export_prediction
from lib.pipeline.runtime import body_model_root, runtime


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='GT-free single-person video inference using the single EIHMR pipeline')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--droid-root', type=Path, required=True)
    parser.add_argument('--body-model-root', type=Path,
                        help='SMPL/SMPL-X models (smplx/, smpl/); default <prompthmr-root>/data/body_models')
    parser.add_argument('--prompthmr-root', type=Path, required=True)
    parser.add_argument('--metric3d-root', type=Path)
    parser.add_argument('--detector-weights', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pipeline.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interval', type=int, default=5, help='Base OAR window-center spacing, not DROID keyframe spacing')
    parser.add_argument('--frontdelta-max-translation', type=float, help='Per-keyframe delta clamp in meters; default from config')
    parser.add_argument('--frontdelta-max-rotation-deg', type=float, help='Per-keyframe delta clamp in degrees; default from config')
    parser.add_argument('--keyframe-stride', type=int, help='Uniform frontdelta keyframe spacing in frames; default from config')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--intrinsics', nargs=4, type=float, metavar=('FX', 'FY', 'CX', 'CY'))
    parser.add_argument('--track-id', type=int)
    parser.add_argument('--max-track-gap', type=int, default=15)
    parser.add_argument('--no-frontdelta', action='store_true')
    parser.add_argument('--disable-cudnn', action='store_true', help='Workaround for driver/cuDNN convolution faults; slower inference')
    parser.add_argument('--stop-after', choices=['frontdelta'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-vertices', action='store_true')
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.gpu < 0 or args.interval < 1 or args.max_track_gap < 0:
        parser.error('gpu/interval/max-track-gap are outside the supported range')
    for source in [ROOT, args.video.parent, args.asset_root, args.droid_root, args.prompthmr_root]:
        if args.output_root == source or source in args.output_root.parents:
            raise ValueError(f'Output must be separate from input/asset/source directory: {source}')
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / 'video.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import torch
    torch.backends.cudnn.enabled = not args.disable_cudnn
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    paths = {'prompthmr': args.prompthmr_root,
             'metric3d': args.metric3d_root or args.prompthmr_root / 'pipeline/yvanyin_metric3d_main'}
    config = video_config(args.config, paths, args.interval, not args.no_frontdelta)
    camera_config = next(c for c in config['components'] if c['type'] == 'camera_refine')
    routing = camera_config['routing']
    if args.keyframe_stride is not None:
        if args.keyframe_stride < 1:
            parser.error('--keyframe-stride must be positive')
        routing['keyframe_stride'] = args.keyframe_stride
    frontend = camera_config['frontend']
    for option, key in [(args.frontdelta_max_translation, 'max_translation_delta'),
                        (args.frontdelta_max_rotation_deg, 'max_rotation_delta_deg')]:
        if option is not None:
            if not np.isfinite(option) or option <= 0:
                raise ValueError('Frontdelta clamps must be finite and positive')
            frontend[key] = option
    images, timeline, decoded = decode_video(args.video, args.output_root / 'input', args.frame_step, args.max_frames, args.fps)
    intervals = np.diff(timeline['timestamps'])
    if len(intervals) and np.max(np.abs(intervals - np.median(intervals))) > .1 * np.median(intervals):
        raise ValueError('Variable-rate video requires constant-rate resampling before temporal inference')
    effective_fps = 1.0 / np.median(intervals) if len(intervals) else decoded['fps'] / args.frame_step
    for component in config['components']:
        if component['type'] == 'depth_scene_refine':
            component['fps'] = float(effective_fps)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'lib').rglob('*.py')}
    source_hashes['infer_video.py'] = digest(Path(__file__))
    identity = {'video_sha256': digest(args.video), 'config': signature(config),
                'sources': source_hashes,
                'model_assets': asset_fingerprint(args.asset_root, body_model_root(args), args.prompthmr_root),
                'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ['resume', 'stop_after', 'prepare_only', 'gpu']}}
    marker = args.output_root / 'video_manifest.json'
    if marker.exists():
        import json
        if not args.resume or json.loads(marker.read_text())['identity'] != identity:
            raise ValueError('Existing video run: use --resume with identical inputs/config/source')
    else:
        write_json(marker, {'identity': identity, 'mode': 'GT-free single-person video'})
    minimum = max(8, 2 * args.interval + 1)
    if len(images) < minimum:
        raise ValueError(f'Need at least {minimum} processed frames; native DROID still requires sufficient motion to initialize')
    tracking = track_person(images, args.detector_weights, args.output_root / 'input',
                            args.track_id, args.max_track_gap, device='0')
    fx, fy, cx, cy = args.intrinsics or [max(decoded['width'], decoded['height'])] * 2 + [decoded['width'] / 2, decoded['height'] / 2]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError('Invalid camera intrinsics')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    write_json(args.output_root / 'effective_config.json', config)
    write_json(args.output_root / 'input_audit.json', {'annotations': False,
               'intrinsics_source': 'user_calibration' if args.intrinsics else 'max_image_dimension_estimate',
               'K': K.tolist(), 'frames': len(images), 'track_id': int(tracking['track_id']),
               'observed_frames': int(tracking['observed'].sum()), 'timestamp_source': decoded['timestamp_source']})
    if args.prepare_only:
        return
    import torch
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    if not torch.cuda.is_available():
        raise RuntimeError('Full inference requires CUDA and separately installed SLAM extensions')
    _ = torch.ones((8, 8), device='cuda', dtype=torch.float16) @ torch.ones((8, 8), device='cuda', dtype=torch.float16)
    runtime(args)
    data = PipelineData(sequence_name='video', sequence_path=str(args.output_root / 'input'),
                        image_paths=[str(p) for p in images],
                        bboxes=np.concatenate([tracking['bboxes_xyxy'], np.ones((len(images), 1), dtype=np.float32)], axis=1)[:, None],
                        camera_params=CameraParams(intrinsics=K), annotations=None, valid_frames_mask=None,
                        metadata={'inference_mode': 'video', 'dataset_type': 'video',
                                  'output_dir': str(args.output_root), 'stop_after': args.stop_after,
                                  'person_boxes': tracking['person_boxes'],
                                  'input_fingerprint': identity})
    result = Pipeline(config, args.output_root).execute(data)
    if result.should_stop:
        write_json(args.output_root / 'paused.json', {'stage': 'frontdelta', 'success': True})
        return
    exported = export_prediction(result, args.output_root / 'trajectories', timeline, valid=tracking['observed'], device='cuda', vertices=args.export_vertices)
    write_json(args.output_root / 'video_success.json', {'success': True, 'trajectory_files': exported['files'],
               'scope': 'fresh GT-free video pipeline and final-camera-consistent body export'})
    print(f'Trajectories: {args.output_root / "trajectories"}')


if __name__ == '__main__':
    main()

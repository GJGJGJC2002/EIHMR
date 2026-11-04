import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from .checkpoint import digest, read_npz, signature, write_json, write_npz


def video_config(path, paths, interval=5, frontend=True, device='cuda'):
    config = yaml.safe_load(Path(path).read_text())
    config['device'] = device
    config['protocol'] = 'GT-free single-person video'
    config['components'] = [c for c in config['components'] if c['type'] != 'evaluation']
    for component in config['components']:
        kind = component['type']
        component['device'] = device
        if kind == 'slam':
            for key in ('forced_keyframe_stride', 'forced_keyframes', 'use_rendered_for_keyframes',
                        'use_warmstart_poses', 'enable_scale_alignment'):
                component.pop(key, None)
            component.update(backend='droid', align_to_world=False, return_keyframe_info=True)
        if kind == 'hpe':
            component.update(prompthmr_root=str(paths['prompthmr']), use_gt_video_k=True,
                             substep_cache_use=False, frame_stride=1, max_frames=None)
        if kind == 'depth_scene_refine':
            component.update(stride=interval, metric3d_root=str(paths['metric3d']),
                             prompthmr_root=str(paths['prompthmr']))
            if component.get('contact_ik_scene_model') in ('rich_floor', 'gt_floor', 'rich_gt_floor'):
                raise ValueError('GT floor is unavailable in video inference')
        if kind == 'adjacent_smpl_renderer':
            component.update(pre_dis=interval, num_adjacent_frames=6)
        if kind == 'camera_refine':
            component['frontend']['enabled'] = frontend
            component['routing'] = {'enabled': True, **component.get('routing', {})}
    return config


def select_second_pass(camera, num_frames, policy):
    """Frontdelta keyframes for a video: a uniform grid with ``policy['keyframe_stride']``, for every video.

    The predicted first-pass camera travel (no annotations) is recorded for reference only.
    """
    positions = np.asarray(camera.T, dtype=np.float64)
    if positions.shape != (num_frames, 3) or num_frames < 2 or not np.isfinite(positions).all():
        raise ValueError('Second pass requires finite full-frame metric pass1 camera positions')
    stride = policy.get('keyframe_stride', 5)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError('Uniform keyframe stride must be a positive integer')
    return {'pass1_travel_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum()),
            'source': 'pass1_predicted_metric_c2w_not_GT', 'mode': 'uniform_keyframes',
            'keyframe_stride': stride, 'forced_keyframes': list(range(0, num_frames, stride)),
            'local_metric_scale': dict(policy.get('local_metric_scale', {}))}


def asset_fingerprint(asset_root, body_model_root, prompthmr_root):
    roots = [Path(asset_root) / 'pretrain', Path(asset_root) / 'smpl', Path(body_model_root),
             Path(prompthmr_root) / 'data/pretrain']
    files = {}
    for index, root in enumerate(roots):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.pt', '.pth', '.ckpt', '.pkl', '.npz', '.npy', '.yaml'):
                files[f'{index}/{path.relative_to(root)}'] = digest(path)
    if not files:
        raise ValueError('No model assets found for inference fingerprint')
    return files


def decode_video(video, directory, frame_step=1, max_frames=None, fps_override=None):
    video, directory = Path(video), Path(directory)
    identity = {'video_sha256': digest(video), 'frame_step': frame_step,
                'max_frames': max_frames, 'fps_override': fps_override}
    marker = directory / 'video_complete.json'
    if marker.exists():
        stored = json.loads(marker.read_text())
        if stored['identity'] != identity:
            raise ValueError('Decoded video cache belongs to different input/options')
        images = [directory / 'images' / name for name in stored['images']]
        if any(not p.is_file() or digest(p) != stored['image_hashes'][p.name] for p in images):
            raise ValueError('Decoded frame cache is missing or changed')
        return images, read_npz(directory / 'timeline.npz'), stored
    if frame_step < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError('frame_step and max_frames must be positive')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode video: {video}')
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video FPS is unavailable; provide --fps')
    image_dir = directory / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    if any(image_dir.iterdir()):
        cap.release()
        raise ValueError('Incomplete decoded cache; choose a fresh output directory')
    images, frame_ids, timestamps = [], [], []
    shape, frame = None, 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame % frame_step == 0:
                if shape is not None and image.shape != shape:
                    raise ValueError('Video resolution changes are unsupported')
                shape = image.shape
                path = image_dir / f'{len(images):05d}.jpg'
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f'Cannot write decoded frame: {path}')
                position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                images.append(path)
                frame_ids.append(frame)
                timestamps.append(position)
                if max_frames is not None and len(images) >= max_frames:
                    break
            frame += 1
    finally:
        cap.release()
    if not images:
        raise ValueError('Video contains no decodable frames')
    timestamps = np.asarray(timestamps, dtype=np.float64)
    timestamp_source = 'decoder_presentation_time'
    if fps_override is not None:
        timestamps = np.asarray(frame_ids, dtype=np.float64) / fps
        timestamp_source = 'user_fps_override'
    elif not np.isfinite(timestamps).all() or (len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0)):
        timestamps = np.asarray(frame_ids, dtype=np.float64) / fps
        timestamp_source = 'nominal_fps_fallback'
    timeline = {'frame_ids': np.asarray(frame_ids, dtype=np.int64), 'timestamps': timestamps}
    write_npz(directory / 'timeline.npz', timeline)
    stored = {'identity': identity, 'fps': fps, 'width': shape[1], 'height': shape[0],
              'timestamp_source': timestamp_source, 'images': [p.name for p in images],
              'image_hashes': {p.name: digest(p) for p in images}}
    write_json(marker, stored)
    return images, timeline, stored


def complete_track(frame_ids, boxes, num_frames, width, height, max_gap=15):
    ids, boxes = np.asarray(frame_ids, dtype=np.int64), np.asarray(boxes, dtype=np.float32)
    if len(ids) == 0 or boxes.shape != (len(ids), 4) or not np.isfinite(boxes).all():
        raise ValueError('No usable person detections')
    if np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= num_frames:
        raise ValueError('Track indices must be unique, sorted and inside video')
    observed = np.zeros(num_frames, dtype=bool)
    observed[ids] = True
    missing = np.flatnonzero(~observed)
    runs = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1) if len(missing) else []
    if any(len(run) > max_gap for run in runs):
        raise ValueError('Selected person has long missing spans; trim video or provide a continuous track')
    full = np.stack([np.interp(np.arange(num_frames), ids, boxes[:, i]) for i in range(4)], axis=1).astype(np.float32)
    full[:, [0, 2]] = full[:, [0, 2]].clip(0, width)
    full[:, [1, 3]] = full[:, [1, 3]].clip(0, height)
    if np.any(full[:, 2:] - full[:, :2] < 2):
        raise ValueError('Person boxes must have width and height >= 2 pixels')
    return full, observed


def _yolo(detector_path):
    """Load YOLO without letting it rebind the process to physical GPU 0.

    ultralytics' select_device('0') overwrites CUDA_VISIBLE_DEVICES; initializing CUDA first fixes the
    device mapping of this process, and the variable is restored afterwards.
    """
    import os

    import torch
    from ultralytics import YOLO
    if torch.cuda.is_available():
        torch.cuda.init()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    return YOLO(str(detector_path)), visible


def _restore_visible(visible):
    import os
    if visible is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible


def track_person(images, detector_path, output, track_id=None, max_gap=15, device='0'):
    detector_path = Path(detector_path)
    if not detector_path.is_file():
        raise FileNotFoundError('Provide a locally obtained YOLO person detector checkpoint')
    output = Path(output)
    identity = {'frames': signature([(p.name, digest(p)) for p in images]),
                'weights': digest(detector_path), 'track_id': track_id, 'max_gap': max_gap, 'person_boxes': True}
    marker = output / 'tracking_complete.json'
    if marker.exists():
        stored = json.loads(marker.read_text())
        if stored['identity'] != identity or digest(output / 'tracking.npz') != stored['sha256']:
            raise ValueError('Stale or corrupt person tracking cache')
        return read_npz(output / 'tracking.npz')
    detector, visible = _yolo(detector_path)
    tracks = defaultdict(list)
    people = []  # every detected person per frame, so SLAM can mask bystanders too
    for index, image in enumerate(images):
        results = detector.track(str(image), persist=True, classes=[0], conf=.5,
                                 verbose=False, device=device, tracker='bytetrack.yaml')
        boxes = results[0].boxes
        people.append(np.zeros((0, 5), np.float32) if boxes is None else np.concatenate(
            [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None]], axis=1).astype(np.float32))
        if boxes is not None and boxes.id is not None:
            for ident, box in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().numpy()):
                tracks[ident].append((index, box))
    _restore_visible(visible)
    if not tracks:
        raise ValueError('No person track detected')
    if track_id is None:
        track_id = max(tracks, key=lambda ident: (len(tracks[ident]), sum(
            float((box[2] - box[0]) * (box[3] - box[1])) for _, box in tracks[ident]), -ident))
    if track_id not in tracks:
        raise ValueError(f'Track {track_id} absent; available IDs: {sorted(tracks)}')
    image = cv2.imread(str(images[0]))
    boxes, observed = complete_track([i for i, _ in tracks[track_id]],
                                     [b for _, b in tracks[track_id]], len(images),
                                     image.shape[1], image.shape[0], max_gap)
    person_boxes = np.zeros((len(images), max(1, max(len(p) for p in people)), 5), np.float32)
    for index, frame in enumerate(people):
        person_boxes[index, :len(frame)] = frame
    value = {'bboxes_xyxy': boxes, 'observed': observed, 'track_id': np.array(track_id, dtype=np.int64),
             'person_boxes': person_boxes}
    write_npz(output / 'tracking.npz', value)
    write_json(marker, {'identity': identity, 'sha256': digest(output / 'tracking.npz')})
    return value


def detect_people(images, detector_path, output, device='0'):
    """Every detected person per frame ([N, K, 5] xyxy + score), used to mask bystanders for SLAM.

    For benchmarks that provide the target box: no tracking, only per-frame person detection.
    """
    detector_path = Path(detector_path)
    if not detector_path.is_file():
        raise FileNotFoundError('Provide a locally obtained YOLO person detector checkpoint')
    output = Path(output)
    identity = {'frames': signature([(Path(p).name, digest(p)) for p in images]), 'weights': digest(detector_path)}
    marker = output / 'people_complete.json'
    if marker.exists():
        stored = json.loads(marker.read_text())
        if stored['identity'] != identity or digest(output / 'people.npz') != stored['sha256']:
            raise ValueError('Stale or corrupt person detection cache')
        return read_npz(output / 'people.npz')['person_boxes']
    detector, visible = _yolo(detector_path)
    people = []
    for image in images:
        boxes = detector.predict(str(image), classes=[0], conf=.5, verbose=False, device=device)[0].boxes
        people.append(np.zeros((0, 5), np.float32) if boxes is None else np.concatenate(
            [boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None]], axis=1).astype(np.float32))
    _restore_visible(visible)
    person_boxes = np.zeros((len(images), max(1, max(len(p) for p in people)), 5), np.float32)
    for index, frame in enumerate(people):
        person_boxes[index, :len(frame)] = frame
    output.mkdir(parents=True, exist_ok=True)
    write_npz(output / 'people.npz', {'person_boxes': person_boxes})
    write_json(marker, {'identity': identity, 'sha256': digest(output / 'people.npz')})
    return person_boxes

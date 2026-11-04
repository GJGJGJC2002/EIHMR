import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'eihmr.trajectory.v1'
JOINT_SCHEMA = 'smpl_neutral_regressed24'


def array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def world_points(points, R, T):
    return np.einsum('fij,fkj->fki', R, points) + T[:, None]


def world_translation(transl, offset, R, T):
    return np.einsum('fij,fj->fi', R, transl + offset) + T - offset


def poses(R, T):
    result = np.broadcast_to(np.eye(4, dtype=np.float64), (len(T), 4, 4)).copy()
    result[:, :3, :3], result[:, :3, 3] = R, T
    return result


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_bundle(directory, camera, human, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in [('camera', camera), ('human', human)]:
        if not value:
            continue
        for key, item in value.items():
            if np.asarray(item).dtype.kind == 'O':
                raise ValueError(f'Object/pickle field forbidden: {name}.{key}')
        temporary = directory / f'{name}.npz.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **value)
        temporary.replace(directory / f'{name}.npz')
    manifest = dict(metadata, schema_version=SCHEMA, units='m',
                    camera_convention='c2w_column_vector', camera_axes='OpenCV: x right, y down, z forward',
                    joint_schema=JOINT_SCHEMA,
                    files={f'{name}.npz': _hash(directory / f'{name}.npz')
                           for name, value in [('camera', camera), ('human', human)] if value})
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    temporary.replace(directory / 'manifest.json')
    load_bundle(directory)
    return manifest


def validate_timeline(value):
    ids, times = value['frame_ids'], value['timestamps']
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or times.shape != ids.shape or not len(ids):
        raise ValueError('Require nonempty integer frame_ids and aligned timestamps')
    if not np.isfinite(times).all() or np.any(np.diff(ids) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError('Frame IDs and timestamps must strictly increase')
    valid = value['valid']
    if valid.shape != ids.shape or valid.dtype.kind != 'b':
        raise ValueError('Require boolean valid array on the exact timeline')
    return len(ids)


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema_version'] != SCHEMA or manifest['units'] != 'm' or manifest['camera_convention'] != 'c2w_column_vector':
        raise ValueError('Unsupported trajectory schema, units or coordinate convention')
    values = {}
    for name in ['camera', 'human']:
        path = directory / f'{name}.npz'
        if path.name not in manifest['files']:
            continue
        if _hash(path) != manifest['files'][path.name]:
            raise ValueError(f'Trajectory checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as archive:
            value = {key: archive[key] for key in archive.files}
        n = validate_timeline(value)
        if name == 'camera':
            c2w = value['c2w']
            if c2w.shape != (n, 4, 4) or not np.isfinite(c2w).all():
                raise ValueError('Invalid c2w trajectory')
            if not np.allclose(c2w[:, 3], [0, 0, 0, 1], atol=1e-6):
                raise ValueError('Invalid homogeneous pose row')
            R = c2w[:, :3, :3]
            if not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=2e-4) or not np.allclose(np.linalg.det(R), 1, atol=2e-4):
                raise ValueError('Camera rotations must belong to SO(3)')
            if 'K' in value and (value['K'].shape not in [(3, 3), (n, 3, 3)] or not np.isfinite(value['K']).all()):
                raise ValueError('Invalid camera intrinsics')
        else:
            joints = value['eval_joints24_world']
            if joints.shape != (n, 24, 3) or not np.isfinite(joints).all():
                raise ValueError('Require finite eval_joints24_world with shape F x 24 x 3')
        values[name] = value
    if not values:
        raise ValueError('No trajectory arrays found')
    return manifest, values


def export_prediction(data, directory, timeline, valid=None, batch_size=128, device='cpu', vertices=False):
    import smplx
    import torch
    from scipy.ndimage import gaussian_filter1d
    from lib.gvhmr import ASSETS, body_model_root
    from lib.utils.rotation_conversions import axis_angle_to_matrix

    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    params = data.smpl_params.to_dict()
    n = len(timeline['frame_ids'])
    R, T = array(data.camera_params.R), array(data.camera_params.T)
    if R.shape != (n, 3, 3) or T.shape != (n, 3):
        raise ValueError('Camera and timeline lengths differ')
    full_valid = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if full_valid.shape != (n,):
        raise ValueError('Valid mask does not match camera timeline')
    camera = dict(timeline, valid=full_valid, c2w=poses(R, T))
    if data.camera_params.intrinsics is not None:
        camera['K'] = array(data.camera_params.intrinsics)
    required = ['global_orient_c', 'body_pose_aa', 'trans', 'betas']
    body_n = params['global_orient_c'].shape[0]
    if body_n != n:
        if body_n != int(full_valid.sum()):
            raise ValueError('Body timeline is neither full camera timeline nor exact valid-frame subset')
        R, T = R[full_valid], T[full_valid]
        timeline = {key: value[full_valid] for key, value in timeline.items()}
        body_valid = np.ones(body_n, dtype=bool)
    else:
        body_valid = full_valid
    n = body_n
    for name in required:
        if name not in params or params[name].shape[0] != n:
            raise ValueError(f'SMPL-X field {name} does not match body timeline')
    model = smplx.create(str(body_model_root()), model_type='smplx', gender='neutral', num_betas=10,
                         num_pca_comps=12, flat_hand_mean=False).to(device).eval()
    mapper_path = ASSETS / 'smplx2smpl_sparse.pt'
    regressor_path = ASSETS / 'smpl_neutral_J_regressor.pt'
    mapper = torch.load(mapper_path, map_location=device, weights_only=True)
    regressor = torch.load(regressor_path, map_location=device, weights_only=True)
    joints22, joints24, mesh, offsets = [], [], [], []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            tensor = lambda key: torch.as_tensor(params[key][start:end], device=device, dtype=torch.float32)
            count = end - start
            zeros3 = torch.zeros(count, 3, device=device)
            output = model(global_orient=tensor('global_orient_c'), body_pose=tensor('body_pose_aa'),
                           betas=tensor('betas'), transl=tensor('trans'),
                           left_hand_pose=torch.zeros(count, 12, device=device),
                           right_hand_pose=torch.zeros(count, 12, device=device),
                           jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                           expression=torch.zeros(count, 10, device=device))
            converted = torch.stack([torch.matmul(mapper, v) for v in output.vertices])
            joints24.append(array(torch.matmul(regressor, converted)))
            joints22.append(array(output.joints[:, :22]))
            shape = model.v_template + torch.einsum('bl,vcl->bvc', tensor('betas'), model.shapedirs)
            offsets.append(array(torch.einsum('v,bvc->bc', model.J_regressor[0], shape)))
            if vertices:
                mesh.append(array(converted))
    cam24, cam22, offset = np.concatenate(joints24), np.concatenate(joints22), np.concatenate(offsets)
    raw24, raw22 = world_points(cam24, R, T), world_points(cam22, R, T)
    filtered24 = raw24 + (gaussian_filter1d(raw24[:, 0], 3, axis=0) - raw24[:, 0])[:, None]
    orient_cam = array(axis_angle_to_matrix(torch.as_tensor(params['global_orient_c']).float()))
    common = dict(timeline, valid=body_valid)
    human = dict(common, joints22_cam=cam22, joints22_world=raw22,
                 eval_joints24_cam=cam24, eval_joints24_world=filtered24,
                 eval_joints24_world_raw=raw24, root_position_world=raw22[:, 0],
                 root_position_cam=cam22[:, 0], root_orient_cam=orient_cam,
                 root_orient_world=R @ orient_cam, transl_cam=params['trans'],
                 transl_world=world_translation(params['trans'], offset, R, T),
                 skeleton_offset=offset, body_pose_aa=params['body_pose_aa'], betas=params['betas'])
    if 'global_trans' in params:
        human['oar_transl_world'] = params['global_trans']
    if 'global_orient_w' in params:
        human['oar_global_orient_world_aa'] = params['global_orient_w']
    if vertices:
        cam_mesh = np.concatenate(mesh)
        human['eval_vertices6890_cam'] = cam_mesh
        human['eval_vertices6890_world_raw'] = world_points(cam_mesh, R, T)
    metadata = {'sequence': data.sequence_name, 'kind': 'prediction', 'model_type': 'SMPL-X supermotion',
                'world_frame': 'final_slam_arbitrary_gauge', 'world_source': 'final_c2w_times_incam_body',
                'postprocess': {'joints22_world': 'none', 'transl_world': 'beta_root_offset_aware',
                                'eval_joints24_world': 'historical root Gaussian sigma3 frames',
                                'oar_fields': 'separate branch; not paired with final camera'},
                'native_joint_schema': 'smplx_first22',
                'timeline_policy': 'full camera; body exact valid subset when historical body is compressed',
                'assets': {'smplx2smpl_sha256': _hash(mapper_path), 'joint_regressor_sha256': _hash(regressor_path)},
                'inference_audit': {'second_pass_route': data.metadata.get('second_pass_route'),
                                    'frontdelta': data.metadata.get('frontend_delta_slam'),
                                    'camera_refine': data.metadata.get('human_mesh_camera_refine'),
                                    'tracking_valid_is_observed': data.metadata.get('inference_mode') == 'video'}}
    return save_bundle(directory, camera, human, metadata)

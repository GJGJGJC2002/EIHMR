import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eval_trajectories import align, evaluate, match_frames, transform
from lib.pipeline.core.data import CameraParams, PipelineData
from lib.pipeline.components.slam import SLAMComponent
from lib.pipeline.video import complete_track, decode_video, select_second_pass, video_config
from lib.pipeline.pipeline import video_component_config
from lib.trajectories import load_bundle, poses, save_bundle, world_points, world_translation


ROOT = Path(__file__).resolve().parents[1]


def fixture(n=105):
    ids = np.arange(n, dtype=np.int64)
    common = {'frame_ids': ids, 'timestamps': ids / 30, 'valid': np.ones(n, dtype=bool)}
    t = ids / 20
    root = np.stack([t, np.sin(t), np.cos(t) * .2], axis=1)
    offsets = np.random.default_rng(7).normal(0, .1, (24, 3))
    joints = root[:, None] + offsets[None]
    return dict(common, c2w=poses(np.broadcast_to(np.eye(3), (n, 3, 3)), root)), dict(common, eval_joints24_world=joints)


class TrajectoryTest(unittest.TestCase):
    def test_world_transform_and_beta_offset_translation(self):
        R = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
        T = np.array([[1, 2, 3]])
        transl = np.array([[.3, .2, 1.]])
        offset = np.array([[.1, .2, 0.]])
        root = world_points((transl + offset)[:, None], R, T)[:, 0]
        np.testing.assert_allclose(world_translation(transl, offset, R, T) + offset, root)

    def test_perfect_trajectory_metrics_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, human = fixture()
            pred, gt = Path(directory) / 'pred', Path(directory) / 'gt'
            save_bundle(pred, camera, human, {})
            save_bundle(gt, camera, human, {})
            result = evaluate(pred, gt)
            for key in ['ate', 'ate_s', 'w_mpjpe', 'wa_mpjpe', 'rte']:
                self.assertLess(abs(result['metrics'][key]), 1e-3)
            self.assertEqual(result['metrics']['_cnt_w_mpjpe'], 105)
            self.assertEqual(result['metrics']['_cnt_rte'], 105)
            self.assertEqual(result['unavailable'], {})

    def test_sim3_camera_alignment_recovers_scale(self):
        camera, _ = fixture()
        target = camera['c2w'][:, :3, 3]
        source = target * 2 + np.array([5, 3, -1])
        np.testing.assert_allclose(transform(source, align(source, target)), target, atol=1e-10)
        self.assertGreater(np.linalg.norm(transform(source, align(source, target, False)) - target), .1)

    def test_corrupt_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, human = fixture(10)
            save_bundle(directory, camera, human, {})
            (Path(directory) / 'camera.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                load_bundle(directory)

    def test_no_object_fields_and_bad_units(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, human = fixture(10)
            camera['unsafe'] = np.array([{}], dtype=object)
            with self.assertRaises(ValueError):
                save_bundle(directory, camera, human, {})
            camera.pop('unsafe')
            save_bundle(directory, camera, human, {})
            path = Path(directory) / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['units'] = 'mm'
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_bundle(directory)

    def test_missing_frame_ids_not_silently_truncated(self):
        camera, _ = fixture(10)
        shorter = {key: value[1:] for key, value in camera.items()}
        with self.assertRaises(ValueError):
            match_frames(camera, shorter)
        ids, _, _ = match_frames(camera, shorter, allow_partial=True)
        self.assertEqual(len(ids), 9)
        shorter['timestamps'] = shorter['timestamps'] + 1
        with self.assertRaises(ValueError):
            match_frames(camera, shorter, allow_partial=True)

    def test_stationary_gt_rte_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            camera, human = fixture(10)
            human['eval_joints24_world'][:] = human['eval_joints24_world'][0]
            pred, gt = Path(directory) / 'pred', Path(directory) / 'gt'
            save_bundle(pred, camera, human, {})
            save_bundle(gt, camera, human, {})
            result = evaluate(pred, gt)
            self.assertNotIn('rte', result['metrics'])
            self.assertIn('rte', result['unavailable'])

    def test_eval_does_not_import_torch_or_smpl(self):
        code = 'import sys; import eval_trajectories; assert "torch" not in sys.modules; assert "smplx" not in sys.modules'
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_person_short_gap_and_long_gap_rejection(self):
        boxes, observed = complete_track([0, 2], [[1, 1, 10, 20], [3, 3, 12, 22]], 3, 100, 100, 1)
        np.testing.assert_allclose(boxes[1], [2, 2, 11, 21])
        np.testing.assert_array_equal(observed, [True, False, True])
        with self.assertRaises(ValueError):
            complete_track([0, 4], [[1, 1, 10, 20], [3, 3, 12, 22]], 5, 100, 100, 1)
        with self.assertRaises(ValueError):
            complete_track([], [], 5, 100, 100)

    def test_video_decode_original_frame_ids_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / 'input.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 24))
            self.assertTrue(writer.isOpened())
            for i in range(12):
                writer.write(np.full((24, 32, 3), i * 10, dtype=np.uint8))
            writer.release()
            output = Path(directory) / 'frames'
            images, timeline, _ = decode_video(video, output, frame_step=2, max_frames=4)
            np.testing.assert_array_equal(timeline['frame_ids'], [0, 2, 4, 6])
            self.assertEqual(len(images), 4)
            again, _, _ = decode_video(video, output, frame_step=2, max_frames=4)
            self.assertEqual(images, again)
            with self.assertRaises(ValueError):
                decode_video(video, output, frame_step=1)
            _, adjusted, metadata = decode_video(video, Path(directory) / 'override', frame_step=2, max_frames=4, fps_override=15)
            np.testing.assert_allclose(adjusted['timestamps'], [0, 2 / 15, 4 / 15, 6 / 15])
            self.assertEqual(metadata['timestamp_source'], 'user_fps_override')

    def test_video_profile_is_gt_free_and_corrected(self):
        config = video_config(ROOT / 'configs/pipeline.yaml',
                              {key: Path('/assets') for key in ['metric3d', 'prompthmr']})
        components = {c['type']: c for c in config['components']}
        self.assertNotIn('evaluation', components)
        self.assertFalse(components['slam']['align_to_world'])
        self.assertEqual(components['slam']['backend'], 'droid')
        self.assertTrue(components['slam']['return_keyframe_info'])
        self.assertNotIn('forced_keyframe_stride', components['slam'])
        self.assertNotIn('forced_keyframes', components['slam'])
        self.assertEqual(components['camera_refine']['frontend']['delta_coordinate_mode'], 'baseline_world')
        self.assertTrue(components['camera_refine']['frontend']['enabled'])
        self.assertNotIn('reject_on_excess_delta', components['camera_refine']['frontend'])
        self.assertTrue(components['camera_refine']['routing']['enabled'])
        self.assertEqual(components['camera_refine']['routing']['keyframe_stride'], 5)

    def test_every_video_uses_uniform_keyframes(self):
        policy = {'keyframe_stride': 5, 'local_metric_scale': {'strength': .75}}
        for length in [5., 19., 20., 60., 200.]:
            camera = CameraParams(T=np.stack([np.linspace(0, length, 41), np.zeros(41), np.zeros(41)], axis=1))
            route = select_second_pass(camera, 41, policy)
            self.assertAlmostEqual(route['pass1_travel_m'], length)
            self.assertEqual(route['mode'], 'uniform_keyframes')
            self.assertEqual(route['forced_keyframes'], list(range(0, 41, 5)))
            self.assertEqual(route['local_metric_scale'], {'strength': .75})
        with self.assertRaises(ValueError):
            select_second_pass(camera, 41, dict(policy, keyframe_stride=0))
        current = PipelineData(image_paths=[f'{i}.jpg' for i in range(41)], camera_params=CameraParams(T=np.zeros((41, 3))),
                               metadata={'inference_mode': 'video'})
        current.metadata['second_pass_route'] = select_second_pass(current.camera_params, 41, policy)
        render = video_component_config({'type': 'adjacent_smpl_renderer', 'num_adjacent_frames': 6}, current, True)
        self.assertEqual(render['render_frame_indices'], list(range(0, 41, 5)))
        self.assertEqual(render['pre_dis'], 5)
        self.assertEqual(render['num_adjacent_frames'], 6)
        oar = video_component_config({'type': 'depth_scene_refine', 'stride': 10}, current, True)
        self.assertEqual(oar['window_centers'], list(range(0, 41, 5)))

    def test_native_keyframes_added_to_oar_and_renderer(self):
        current = PipelineData(image_paths=[f'{i}.jpg' for i in range(23)],
                               metadata={'inference_mode': 'video', 'slam_keyframe_info': {'keyframe_tstamps': [0, 3, 11, 22]}})
        oar = {'type': 'depth_scene_refine', 'stride': 5}
        result = video_component_config(oar, current, True)
        self.assertEqual(result['window_centers'], [0, 3, 5, 10, 11, 15, 20, 22])
        self.assertNotIn('window_centers', oar)
        render = video_component_config({'type': 'adjacent_smpl_renderer', 'pre_dis': 5}, current, True)
        self.assertEqual(render['render_frame_indices'], [0, 3, 11, 22])
        self.assertEqual(render['pre_dis'], 1)
        self.assertIs(video_component_config(oar, current, False), oar)
        current.metadata['slam_keyframe_info']['keyframe_tstamps'] = [0, 3, 3]
        with self.assertRaises(ValueError):
            video_component_config(oar, current, True)

    def test_slam_reads_non_gt_input_intrinsics(self):
        K = np.array([[500, 0, 160], [0, 510, 120], [0, 0, 1]])
        data = PipelineData(camera_params=CameraParams(intrinsics=K), annotations=None)
        component = SLAMComponent('slam', {'backend': 'droid_warmstart', 'device': 'cpu'})
        self.assertEqual(component._get_intrinsics(data), [500, 510, 160, 120])


if __name__ == '__main__':
    unittest.main()

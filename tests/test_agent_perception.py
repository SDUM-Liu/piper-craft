from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from astra_vla.agent.contracts import ImageEvidence, number, validate
from astra_vla.agent.experiment import ImagePlaneSimulator
from astra_vla.agent.perception import (ColorRegions, LocalVisualServo, grasp_relation, inspect_grasp,
                                       observe_scene, visual_error)


def make_scene(tmp_path):
    return ImagePlaneSimulator(5).observe(tmp_path/'source')


def test_original_sources_and_no_invented_depth(tmp_path):
    scene = make_scene(tmp_path)
    assert set(scene.views) == {'top', 'left_wrist', 'right_wrist'}
    obj = scene.object('top', 'target')
    assert obj['evidence_sha256'] == scene.views['top'].sha256
    assert obj['timestamp'] == scene.views['top'].timestamp
    assert np.linalg.norm(obj['axis_2d']) == pytest.approx(1)
    assert obj['actionable_regions'] == [] and obj['occlusion'] == 'unknown'
    assert scene.uncertainty == ['metric_depth_unknown', 'contact_force_unknown']
    assert visual_error(scene, 'top', 'target', [110, 90])['norm_px'] > 20
    assert grasp_relation(scene, 'top', 'target')['status'] == 'needs_observation'


def test_ambiguous_object_never_selected_by_list_order(tmp_path):
    scene = make_scene(tmp_path)
    scene.objects.append(dict(scene.objects[0]))
    assert scene.object('top', 'target') is None
    assert visual_error(scene, 'top', 'target', [100, 80])['status'] == 'needs_observation'


def test_partial_mask_axis_and_occlusion_not_overclaimed():
    image = np.zeros((100, 100, 3), np.uint8)
    image[30:40, :50] = [0, 0, 255]
    provider = ColorRegions({'red': [[0, 180, 180], [8, 255, 255]]})
    row, = provider.detect(image)
    assert row['visibility'] == 'partial' and row['axis_2d'] is None
    assert row['occlusion'] == 'unknown'


def test_source_modified_during_inference_rejected(tmp_path):
    scene = make_scene(tmp_path)
    metadata = {camera: dict(available=True, lossless=True, resized=False, stamp=v.timestamp,
                            sequence=v.sequence, serial=v.serial, path=v.path, width=v.width, height=v.height)
                for camera, v in scene.views.items()}
    class TamperingProvider(ColorRegions):
        def detect(self, image):
            Path(scene.views['top'].path).write_bytes(b'changed')
            return []
    with pytest.raises(ValueError, match='source image changed'):
        observe_scene(metadata, {}, TamperingProvider({}))


def test_live_stale_or_future_rgb_rejected(tmp_path):
    scene = make_scene(tmp_path)
    metadata = {camera: dict(available=True, lossless=True, resized=False, stamp=v.timestamp,
                            sequence=v.sequence, serial=v.serial, path=v.path, width=v.width, height=v.height)
                for camera, v in scene.views.items()}
    for now in (100, 2000):
        with pytest.raises(ValueError, match='stale/future RGB'):
            observe_scene(metadata, {}, ColorRegions({}), mode='live_readonly', now=now)
    metadata['top']['lossless'] = False
    with pytest.raises(ValueError, match='lossless'):
        ImageEvidence.from_metadata('top', 'obs', metadata['top'])


def add_gripper(scene, target_x=50):
    original = dict(scene.object('top', 'target'))
    scene.objects = [dict(original, label=name, center_px=center, object_id=name) for name, center in
                     [('target', [target_x, 50]), ('finger_left', [30, 50]),
                      ('finger_right', [70, 50]), ('palm', [50, 30])]]
    return scene


def test_projection_and_apparent_slip_preserve_unknown_physics(tmp_path):
    sim = ImagePlaneSimulator(5)
    before = add_gripper(sim.observe(tmp_path/'before'))
    after = add_gripper(sim.observe(tmp_path/'after'), target_x=60)
    relation = grasp_relation(before, 'top', 'target')
    assert relation['projected_between_fingers']
    assert relation['bilateral_contact'] is None and relation['entry_depth'] is None
    result = inspect_grasp(before, after, 'top', 'target')
    assert result['status'] == 'slippage_detected'
    assert result['physical_slippage'] is None and result['physical_grasp_verified'] is None
    assert inspect_grasp(before, before, 'top', 'target')['status'] == 'needs_observation'
    after.views['top'] = replace(after.views['top'], serial='another_camera')
    assert inspect_grasp(before, after, 'top', 'target')['status'] == 'needs_observation'


def test_servo_uses_measured_feedback_and_rejects_singular_or_noisy_fit():
    servo = LocalVisualServo(gain=.5, max_step=.1)
    assert servo.propose([20, 0])['status'] == 'needs_observation'
    servo.update([.1, 0], [10, 0], evidence=['actual1'])
    servo.update([0, .1], [0, 20], evidence=['actual2'])
    assert servo.propose([20, 0])['delta'] == pytest.approx([.1, 0])
    assert servo.propose([20, 0])['motion_authorized'] is False
    servo.update([.1, 0], [-40, 0], evidence=['actual3'])
    assert servo.propose([20, 0])['status'] == 'needs_observation'
    with pytest.raises(ValueError, match='no measured'):
        servo.update([0, 0], [1, 1], evidence=['target_is_not_feedback'])


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '1'])
def test_nonfinite_and_boolean_numbers_rejected(value):
    with pytest.raises(ValueError):
        number(value)


def test_schema_rejects_extra_fields_and_invalid_bounds():
    schema = {'type': 'object', 'properties': {'gain': {'type': 'number', 'maximum': 1}}, 'required': ['gain']}
    for value in ({}, {'gain': 2}, {'gain': .5, 'execute': True}, {'gain': float('nan')}):
        with pytest.raises(ValueError):
            validate(value, schema)

"""Pure logic tests for TaskB perception/control guards."""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from demo.solution import AlgSolution
from demo.tool.yolo_targets import TrashTarget, servo_command_from_target


def make_target(*, err_x: float, err_y: float, distance: float, source_image: str = "live_000010_head_rgb.png"):
    w, h = 640, 480
    cx = 0.5 * w + err_x * 0.5 * w
    cy = 0.5 * h + err_y * 0.5 * h
    bw, bh = 80, 70
    return TrashTarget(
        target_id="head_00",
        camera="head",
        label="trash",
        confidence=0.9,
        bbox_xyxy=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
        image_size=(w, h),
        image_error=(err_x, err_y),
        depth_m=distance,
        has_valid_depth=True,
        distance_hint_m=distance,
        world_xy=None,
        scan_angle_deg=None,
        source_image=source_image,
        point_camera=None,
        point_body=None,
        frame_kind="live",
        state="precise",
    )


class TaskBLogicTest(unittest.TestCase):
    def test_hold_for_grasp_limits_but_does_not_stop_when_not_ready(self):
        servo = servo_command_from_target(make_target(err_x=0.5, err_y=0.55, distance=0.8))
        self.assertTrue(servo["hold_for_grasp"])
        self.assertFalse(servo["ready_to_grasp"])
        self.assertLessEqual(abs(servo["lin_x"]), 0.05)
        self.assertLessEqual(abs(servo["lin_y"]), 0.08)
        self.assertLessEqual(abs(servo["yaw_rate"]), 0.02)
        self.assertNotEqual((servo["lin_x"], servo["lin_y"], servo["yaw_rate"]), (0.0, 0.0, 0.0))

    def test_ready_for_grasp_stops_with_tight_lateral_window(self):
        servo = servo_command_from_target(make_target(err_x=0.1, err_y=0.55, distance=0.8))
        self.assertTrue(servo["hold_for_grasp"])
        self.assertTrue(servo["ready_to_grasp"])
        self.assertEqual((servo["lin_x"], servo["lin_y"], servo["yaw_rate"]), (0.0, 0.0, 0.0))

    def test_bottom_ready_branch_does_not_relax_lateral_error(self):
        servo = servo_command_from_target(make_target(err_x=0.6, err_y=0.9, distance=0.8))
        self.assertTrue(servo["hold_for_grasp"])
        self.assertFalse(servo["ready_to_grasp"])

    def test_ready_target_match_rejects_target_switch(self):
        a = AlgSolution._ready_target_key(make_target(err_x=0.05, err_y=0.55, distance=0.8))
        b = AlgSolution._ready_target_key(make_target(err_x=0.45, err_y=0.55, distance=0.8))
        self.assertTrue(AlgSolution._ready_target_matches(None, a))
        self.assertFalse(AlgSolution._ready_target_matches(a, b))

    def test_stale_restart_uses_threshold_and_backoff(self):
        obj = object.__new__(AlgSolution)
        obj.last_yolo_diag = {}
        obj.yolo_stale_count = 0
        obj.yolo_stale_restart_threshold = 2
        obj.yolo_restart_backoff_s = 10.0
        obj.last_yolo_restart_time = -1.0
        obj.yolo_started = True
        obj.yolo_process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        obj.yolo_stop_event = None
        obj.yolo_thread = None
        obj.yolo_log_handle = None
        obj.trash_targets = ["old"]
        obj.current_trash_target = "old"
        obj.locked_trash_target = "old"
        obj.locked_trash_miss_steps = 1
        obj.ready_to_grasp_steps = 1
        obj.ready_to_grasp_last_image = "live_000001_head_rgb.png"
        obj.ready_to_grasp_last_target = {"camera": "head"}
        obj.current_trash_servo = {"ready_to_grasp": True}

        obj._clear_yolo_targets("stale_yolo_result")
        self.assertEqual(obj.trash_targets, [])
        self.assertIsNone(obj.current_trash_target)

        self.assertTrue(obj._request_yolo_restart("stale_yolo_result"))
        first_restart = obj.last_yolo_restart_time
        obj.yolo_started = True
        obj.yolo_process = SimpleNamespace(poll=lambda: None, terminate=lambda: None)
        self.assertFalse(obj._request_yolo_restart("stale_yolo_result"))
        self.assertEqual(obj.last_yolo_restart_time, first_restart)
        obj.last_yolo_restart_time = time.time() - 11.0
        self.assertTrue(obj._request_yolo_restart("stale_yolo_result"))


if __name__ == "__main__":
    unittest.main()

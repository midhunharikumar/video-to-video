# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for visualizer/camera_editor.py — presets, look-at, and undo/redo logic."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from visualizer.camera_editor import (
    PRESET_NAMES,
    _look_at_c2w,
    generate_preset,
)
from visualizer.scene import get_source_camera


class TestLookAt:
    def test_shape_and_dtype(self):
        c2w = _look_at_c2w(
            eye=np.array([0, 0, -5]),
            target=np.array([0, 0, 0]),
        )
        assert c2w.shape == (4, 4)
        assert c2w.dtype == np.float32

    def test_rotation_is_valid(self):
        c2w = _look_at_c2w(
            eye=np.array([3, -2, -5]),
            target=np.array([0, 0, 0]),
        )
        R = c2w[:3, :3].astype(np.float64)
        det = np.linalg.det(R)
        assert det == pytest.approx(1.0, abs=1e-5)
        RtR = R.T @ R
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-5)

    def test_forward_points_toward_target(self):
        eye = np.array([0, 0, -10], dtype=np.float64)
        target = np.array([0, 0, 0], dtype=np.float64)
        c2w = _look_at_c2w(eye, target)
        forward = c2w[:3, 2]
        expected = (target - eye) / np.linalg.norm(target - eye)
        np.testing.assert_allclose(forward, expected, atol=1e-5)

    def test_eye_position(self):
        eye = np.array([1, 2, 3], dtype=np.float64)
        c2w = _look_at_c2w(eye, np.array([0, 0, 0]))
        np.testing.assert_allclose(c2w[:3, 3], eye, atol=1e-5)


class TestGeneratePreset:
    def _source_and_center(self, depth=5.0):
        _, c2w = get_source_camera(depth, start_elevation=5.0)
        center = np.array([0, 0, 0], dtype=np.float32)
        return c2w, center

    def test_all_presets_return_valid_keyframes(self):
        c2w, center = self._source_and_center()
        for name in PRESET_NAMES:
            kfs = generate_preset(name, c2w, center)
            assert len(kfs) >= 2, f"{name}: need >= 2 keyframes"
            for i, kf in enumerate(kfs):
                assert kf.shape == (4, 4), f"{name}[{i}]: shape={kf.shape}"
                R = kf[:3, :3].astype(np.float64)
                det = np.linalg.det(R)
                assert det == pytest.approx(1.0, abs=1e-4), f"{name}[{i}]: det={det}"

    def test_push_in_moves_closer(self):
        c2w, center = self._source_and_center()
        kfs = generate_preset("Push-in", c2w, center)
        d_start = np.linalg.norm(kfs[0][:3, 3] - center)
        d_end = np.linalg.norm(kfs[-1][:3, 3] - center)
        assert d_end < d_start

    def test_pull_out_moves_farther(self):
        c2w, center = self._source_and_center()
        kfs = generate_preset("Pull-out", c2w, center)
        d_start = np.linalg.norm(kfs[0][:3, 3] - center)
        d_end = np.linalg.norm(kfs[-1][:3, 3] - center)
        assert d_end > d_start

    def test_orbit_has_three_keyframes(self):
        c2w, center = self._source_and_center()
        kfs = generate_preset("Orbit Left", c2w, center)
        assert len(kfs) == 3

    def test_crane_up_raises_camera(self):
        c2w, center = self._source_and_center()
        kfs = generate_preset("Crane Up", c2w, center)
        y_start = kfs[0][1, 3]
        y_end = kfs[-1][1, 3]
        assert y_end < y_start  # OpenCV Y-down: smaller Y = higher

    def test_first_keyframe_matches_source(self):
        c2w, center = self._source_and_center()
        for name in PRESET_NAMES:
            kfs = generate_preset(name, c2w, center)
            np.testing.assert_allclose(kfs[0], c2w, atol=1e-5, err_msg=f"{name}")

    def test_unknown_preset_raises(self):
        c2w, center = self._source_and_center()
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_preset("Nonexistent", c2w, center)


class TestViewfinderOverlay:
    def test_border_only(self):
        from visualizer.app import _make_viewfinder_overlay
        rgba = _make_viewfinder_overlay(16 / 9, mode="Border only")
        assert rgba is not None
        assert rgba.shape[2] == 4
        assert rgba[100, 200, 3] == 0  # interior (away from crosshair) is transparent

    def test_rule_of_thirds(self):
        from visualizer.app import _make_viewfinder_overlay
        rgba = _make_viewfinder_overlay(16 / 9, mode="Rule of thirds")
        assert rgba is not None
        h, w = rgba.shape[:2]
        y_third = h // 3
        assert rgba[y_third, w // 2, 3] > 0  # grid line is visible

    def test_off_returns_none(self):
        from visualizer.app import _make_viewfinder_overlay
        assert _make_viewfinder_overlay(16 / 9, mode="Off") is None

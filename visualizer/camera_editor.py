# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""
camera_editor.py — Camera keyframe state management for the Viser viewer.

Manages a list of user-placed keyframe cameras (c2w matrices in OpenCV convention)
and their corresponding Viser scene handles (frustums + transform gizmos).
Also maintains the interpolated path visualisation.

Coordinate conventions:
  - Internally: OpenCV (X right, Y down, Z forward)
  - Viser display: Y-up world space (X right, Y up, Z backward)
  - Conversion: apply diag(1, -1, -1) flip on world positions; apply
    _OPENGL_TO_OPENCV to c2w matrices (same flip but for 4x4 matrices)
"""

from __future__ import annotations

import threading

import numpy as np
import viser

from .interpolation import (
    _OPENGL_TO_OPENCV,
    interpolate_camera_path,
    EASING_MODES,
)


# ── Colour palette ────────────────────────────────────────────────────────────
_KF_COLOUR    = (255, 200, 0)    # yellow — keyframe frustums
_PATH_COLOUR  = (255, 140, 0)    # orange — interpolated path spline
_OUTFR_COLOUR = (255, 140, 0)    # orange — output-frame frustums
_SRC_COLOUR   = (0, 200, 255)    # cyan   — source/reference camera


class CameraEditor:
    """
    Manages interactive keyframe cameras inside a Viser scene.

    All c2w / w2c matrices stored internally use OpenCV convention.
    Conversion to Viser (Y-up world) is done on the way in/out.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        fov_deg: float,
        aspect: float,
        frustum_scale: float = 0.15,
        display_offset_viser: np.ndarray | None = None,
    ) -> None:
        self._server = server
        self._fov_deg = fov_deg
        self._aspect = aspect
        self._frustum_scale = frustum_scale
        self._lock = threading.Lock()
        self._display_offset_viser = np.zeros(3, dtype=np.float32)
        if display_offset_viser is not None:
            arr = np.asarray(display_offset_viser, dtype=np.float32).reshape(-1)
            if arr.size == 3:
                self._display_offset_viser = arr.copy()

        # OpenCV c2w matrices — one per keyframe
        self._c2w_keyframes: list[np.ndarray] = []
        # Viser transform-gizmo handles for each keyframe
        self._gizmo_handles: list[viser.TransformControlsHandle] = []
        # Viser frustum handles for each keyframe
        self._frustum_handles: list[viser.CameraFrustumHandle] = []

        # Easing mode for path interpolation
        self._easing_mode: str = "Linear"

        # Undo/redo stacks (shallow copies of keyframe lists)
        self._undo_stack: list[list[np.ndarray]] = []
        self._redo_stack: list[list[np.ndarray]] = []

        # Path visualisation handles (replaced on every update)
        self._path_handle: viser.SplineCatmullRomHandle | None = None
        self._output_frustum_handles: list[viser.CameraFrustumHandle] = []
        self._source_handle: viser.CameraFrustumHandle | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def n_keyframes(self) -> int:
        with self._lock:
            return len(self._c2w_keyframes)

    def get_keyframe_c2ws(self) -> list[np.ndarray]:
        with self._lock:
            return list(self._c2w_keyframes)

    def _save_undo(self) -> None:
        """Snapshot current keyframes onto the undo stack. Clears redo."""
        with self._lock:
            snapshot = [kf.copy() for kf in self._c2w_keyframes]
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def add_keyframe(self, c2w_opencv: np.ndarray) -> None:
        """
        Add a new keyframe at the given c2w pose (OpenCV convention).
        Adds a draggable gizmo + frustum to the scene.
        """
        self._save_undo()
        with self._lock:
            idx = len(self._c2w_keyframes)
            self._c2w_keyframes.append(c2w_opencv.copy())

        self._add_scene_handles(idx, c2w_opencv)
        self._refresh_path()

    def remove_last_keyframe(self) -> bool:
        """Remove the most recently added keyframe. Returns False if nothing to remove."""
        with self._lock:
            if not self._c2w_keyframes:
                return False
        self._save_undo()
        with self._lock:
            self._c2w_keyframes.pop()
            gizmo   = self._gizmo_handles.pop() if self._gizmo_handles else None
            frustum = self._frustum_handles.pop() if self._frustum_handles else None

        if gizmo is not None:
            gizmo.remove()
        if frustum is not None:
            frustum.remove()

        self._refresh_path()
        return True

    def remove_keyframe(self, idx: int) -> bool:
        """Remove keyframe at *idx*. Returns False if index out of range."""
        with self._lock:
            if idx < 0 or idx >= len(self._c2w_keyframes):
                return False
        self._save_undo()
        self._remove_all_scene_handles()
        with self._lock:
            self._c2w_keyframes.pop(idx)
            kfs = list(self._c2w_keyframes)
        self._rebuild_scene_handles(kfs)
        self._refresh_path()
        return True

    def refresh_path(self, n_frames: int | None = None) -> None:
        """
        Force-redraw the interpolated path visualisation.

        If n_frames is given, the output frustums are sampled from a path
        of exactly that many frames (useful for the Preview button).
        """
        self._refresh_path(n_output=n_frames)

    def clear_all(self) -> None:
        """Remove all keyframes and their scene handles."""
        self._save_undo()
        self._remove_all_scene_handles()
        with self._lock:
            self._c2w_keyframes.clear()
        self._clear_path()

    def undo(self) -> bool:
        """Restore keyframes from the undo stack. Returns False if nothing to undo."""
        if not self._undo_stack:
            return False
        with self._lock:
            current = [kf.copy() for kf in self._c2w_keyframes]
        self._redo_stack.append(current)
        snapshot = self._undo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def redo(self) -> bool:
        """Re-apply the last undone change. Returns False if nothing to redo."""
        if not self._redo_stack:
            return False
        with self._lock:
            current = [kf.copy() for kf in self._c2w_keyframes]
        self._undo_stack.append(current)
        snapshot = self._redo_stack.pop()
        self._restore_snapshot(snapshot)
        return True

    def set_easing_mode(self, mode: str) -> None:
        """
        Set the easing/transition mode used for path interpolation.
        Must be one of EASING_MODES.  Triggers a path refresh.
        """
        if mode not in EASING_MODES:
            return
        self._easing_mode = mode
        self._refresh_path()

    def get_interpolated_path(self, n_frames: int) -> np.ndarray | None:
        """
        Return interpolated c2w matrices (OpenCV) of shape [n_frames, 4, 4].
        Returns None if fewer than 2 keyframes have been placed.
        """
        with self._lock:
            kfs = list(self._c2w_keyframes)

        if len(kfs) < 2:
            return None
        return interpolate_camera_path(kfs, n_frames, self._easing_mode)

    def add_source_camera(self, c2w_opencv: np.ndarray) -> None:
        """
        Display the source (reference) camera in the scene as a fixed cyan frustum.
        """
        F = _OPENGL_TO_OPENCV[:3, :3]
        R_viser = (F @ c2w_opencv[:3, :3].astype(np.float32)).astype(np.float32)
        wxyz = _rotation_to_wxyz(R_viser)
        pos  = tuple(self._opencv_pos_to_display(c2w_opencv[:3, 3]).tolist())
        self._source_handle = self._server.scene.add_camera_frustum(
            name="source_camera",
            fov=float(np.radians(self._fov_deg)),
            aspect=float(self._aspect),
            scale=self._frustum_scale * 1.2,
            color=_SRC_COLOUR,
            wxyz=wxyz,
            position=pos,
        )

    def set_gizmos_visible(self, visible: bool) -> None:
        """Toggle visibility of keyframe transform gizmos."""
        with self._lock:
            gizmos = list(self._gizmo_handles)
        for g in gizmos:
            g.visible = bool(visible)

    def set_scene_overlays_visible(self, visible: bool) -> None:
        """Toggle non-pointcloud overlays (source/keyframe/path frustums and spline)."""
        vis = bool(visible)
        with self._lock:
            frustums = list(self._frustum_handles)
            out_frustums = list(self._output_frustum_handles)
            path_handle = self._path_handle
            source_handle = self._source_handle
        if source_handle is not None:
            source_handle.visible = vis
        for h in frustums:
            h.visible = vis
        for h in out_frustums:
            h.visible = vis
        if path_handle is not None:
            path_handle.visible = vis

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _remove_all_scene_handles(self) -> None:
        with self._lock:
            gizmos = list(self._gizmo_handles)
            frustums = list(self._frustum_handles)
            self._gizmo_handles.clear()
            self._frustum_handles.clear()
        for g in gizmos:
            g.remove()
        for f in frustums:
            f.remove()

    def _rebuild_scene_handles(self, kfs: list[np.ndarray]) -> None:
        for i, c2w in enumerate(kfs):
            self._add_scene_handles(i, c2w)

    def _restore_snapshot(self, snapshot: list[np.ndarray]) -> None:
        self._remove_all_scene_handles()
        with self._lock:
            self._c2w_keyframes = snapshot
        self._rebuild_scene_handles(snapshot)
        self._refresh_path()

    def _opencv_pos_to_display(self, pos_opencv: np.ndarray) -> np.ndarray:
        """Convert OpenCV world position to Viser display position (Y-up, recentered)."""
        p = pos_opencv.astype(np.float32)
        return np.array([p[0], -p[1], -p[2]], dtype=np.float32) - self._display_offset_viser

    def _display_pos_to_opencv(self, pos_display: np.ndarray) -> np.ndarray:
        """Convert Viser display position to OpenCV world position."""
        p = np.asarray(pos_display, dtype=np.float32) + self._display_offset_viser
        return np.array([p[0], -p[1], -p[2]], dtype=np.float32)

    def _add_scene_handles(self, idx: int, c2w_opencv: np.ndarray) -> None:
        """Add gizmo + frustum for keyframe at index idx."""
        F = _OPENGL_TO_OPENCV[:3, :3]
        R_viser = (F @ c2w_opencv[:3, :3].astype(np.float32)).astype(np.float32)
        wxyz = _rotation_to_wxyz(R_viser)
        pos  = tuple(self._opencv_pos_to_display(c2w_opencv[:3, 3]).tolist())

        # Transform gizmo — user can drag to reposition/reorient
        gizmo = self._server.scene.add_transform_controls(
            name=f"keyframe_gizmo/{idx}",
            wxyz=wxyz,
            position=pos,
            scale=self._frustum_scale * 1.5,
        )

        # Camera frustum to make the pose visually obvious
        frustum = self._server.scene.add_camera_frustum(
            name=f"keyframe_frustum/{idx}",
            fov=float(np.radians(self._fov_deg)),
            aspect=float(self._aspect),
            scale=self._frustum_scale,
            color=_KF_COLOUR,
            wxyz=wxyz,
            position=pos,
        )

        with self._lock:
            self._gizmo_handles.append(gizmo)
            self._frustum_handles.append(frustum)

        # In viser 1.x the on_update callback receives a TransformControlsEvent.
        def _on_gizmo_update(event: viser.TransformControlsEvent, kf_idx: int = idx) -> None:
            upd_wxyz = event.target.wxyz
            upd_pos  = event.target.position

            with self._lock:
                if kf_idx < len(self._frustum_handles):
                    fh = self._frustum_handles[kf_idx]
                    fh.wxyz     = upd_wxyz
                    fh.position = upd_pos

                if kf_idx < len(self._c2w_keyframes):
                    R_viser = _wxyz_to_rotation(upd_wxyz)
                    # Rotation: world-frame flip only (Viser already uses
                    # col2=forward, col1=down — same local axes as OpenCV)
                    F = _OPENGL_TO_OPENCV[:3, :3]
                    R_opencv = (F @ R_viser).astype(np.float32)
                    # Position: display → OpenCV world
                    pos_opencv = self._display_pos_to_opencv(upd_pos)
                    c2w_ocv = np.eye(4, dtype=np.float32)
                    c2w_ocv[:3, :3] = R_opencv
                    c2w_ocv[:3, 3] = pos_opencv
                    self._c2w_keyframes[kf_idx] = c2w_ocv

            self._refresh_path()

        gizmo.on_update(_on_gizmo_update)

    def _refresh_path(self, n_output: int | None = None) -> None:
        """
        Recompute and redraw the interpolated path spline + output frustums.

        Args:
            n_output: If given, show one frustum per output frame (up to 24 shown
                      for readability). If None, show ~12 evenly-spaced frustums.
        """
        with self._lock:
            kfs = list(self._c2w_keyframes)

        self._clear_path()
        if len(kfs) < 2:
            return

        # Sample the spline at enough points for a smooth curve
        n_vis = max(120, len(kfs) * 20)
        c2ws_vis = interpolate_camera_path(kfs, n_vis, self._easing_mode)

        # Control points in Viser display space for the spline curve
        ctrl_pts = np.array(
            [self._opencv_pos_to_display(c[:3, 3]) for c in c2ws_vis],
            dtype=np.float32,
        )
        self._path_handle = self._server.scene.add_spline_catmull_rom(
            name="camera_path_spline",
            points=ctrl_pts,
            tension=0.5,
            line_width=2.0,
            color=_PATH_COLOUR,
        )

        # Frustums: when n_output is given, sample the exact output frames;
        # otherwise use ~12 evenly-spaced preview frustums.
        if n_output is not None and n_output >= 2:
            c2ws_out = interpolate_camera_path(kfs, n_output, self._easing_mode)
            max_shown = 24
            step = max(1, n_output // max_shown)
            frustum_c2ws = c2ws_out[::step]
        else:
            step = max(1, n_vis // 12)
            frustum_c2ws = c2ws_vis[::step]

        F = _OPENGL_TO_OPENCV[:3, :3]
        for j, c2w_ocv in enumerate(frustum_c2ws):
            R_viser = (F @ c2w_ocv[:3, :3].astype(np.float32)).astype(np.float32)
            wxyz = _rotation_to_wxyz(R_viser)
            pos  = tuple(self._opencv_pos_to_display(c2w_ocv[:3, 3]).tolist())
            h = self._server.scene.add_camera_frustum(
                name=f"path_frustum/{j}",
                fov=float(np.radians(self._fov_deg)),
                aspect=float(self._aspect),
                scale=self._frustum_scale * 0.5,
                color=_OUTFR_COLOUR,
                wxyz=wxyz,
                position=pos,
            )
            self._output_frustum_handles.append(h)

    def _clear_path(self) -> None:
        if self._path_handle is not None:
            self._path_handle.remove()
            self._path_handle = None
        for h in self._output_frustum_handles:
            h.remove()
        self._output_frustum_handles.clear()


# ── Quaternion helpers ────────────────────────────────────────────────────────

def _rotation_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation as _Rot
    q = _Rot.from_matrix(R.astype(np.float64)).as_quat()   # (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _wxyz_to_rotation(wxyz) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation as _Rot
    wxyz = np.asarray(wxyz, dtype=np.float64)
    q_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return _Rot.from_quat(q_xyzw).as_matrix().astype(np.float32)


# ── Camera trajectory presets ────────────────────────────────────────────────

PRESET_NAMES: tuple[str, ...] = (
    "Push-in",
    "Pull-out",
    "Orbit Left",
    "Orbit Right",
    "Crane Up",
)


def _look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Build a c2w matrix looking from *eye* toward *target* (OpenCV convention)."""
    if up is None:
        up = np.array([0, -1, 0], dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def generate_preset(
    name: str,
    c2w_source: np.ndarray,
    scene_center_opencv: np.ndarray,
) -> list[np.ndarray]:
    """Generate 2-3 keyframe c2w matrices (OpenCV) for a preset camera motion.

    Args:
        name: one of PRESET_NAMES
        c2w_source: [4,4] source camera c2w (OpenCV convention)
        scene_center_opencv: [3] scene center in OpenCV world coordinates

    Returns:
        List of c2w matrices forming the preset trajectory.
    """
    eye = c2w_source[:3, 3].astype(np.float64)
    center = scene_center_opencv.astype(np.float64)
    to_cam = eye - center
    dist = np.linalg.norm(to_cam)
    direction = to_cam / (dist + 1e-12)

    if name == "Push-in":
        end_eye = center + direction * dist * 0.65
        return [
            c2w_source.copy(),
            _look_at_c2w(end_eye, center),
        ]

    elif name == "Pull-out":
        end_eye = center + direction * dist * 1.4
        return [
            c2w_source.copy(),
            _look_at_c2w(end_eye, center),
        ]

    elif name in ("Orbit Left", "Orbit Right"):
        angle = np.radians(15.0 if name == "Orbit Left" else -15.0)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_y = np.array([
            [cos_a, 0, sin_a],
            [0,     1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=np.float64)
        mid_dir = R_y @ direction * 0.5
        mid_eye = center + mid_dir * dist
        end_dir = R_y @ R_y @ direction
        end_eye = center + end_dir * dist
        return [
            c2w_source.copy(),
            _look_at_c2w(mid_eye, center),
            _look_at_c2w(end_eye, center),
        ]

    elif name == "Crane Up":
        up_offset = np.array([0, -1, 0], dtype=np.float64) * dist * 0.3
        end_eye = eye + up_offset
        return [
            c2w_source.copy(),
            _look_at_c2w(end_eye, center),
        ]

    else:
        raise ValueError(f"Unknown preset: {name!r}. Choose from {PRESET_NAMES}")

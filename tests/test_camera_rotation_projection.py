import numpy as np
import pytest
from VideoAnimatorEngine import project_to_keyframe

def test_projection_in_bounds():
    """Ensure camera projections stay within normalized bounds for common rotations."""
    width, height = 1080, 1920
    fov_deg = 45
    cam_pos = (-5, 0, 2)
    xg, yg, zg = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5), [0])
    points = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)
    tests = [
        (0, 0, 0),
        (30, 0, 0),
        (0, -90, 0),
        (0, 0, 30),
        (30, 30, 0),
        (0, 30, 30),
        (30, 0, 30),
        (30, 30, 30),
    ]
    for yaw, pitch, roll in tests:
        proj = np.array([
            project_to_keyframe(x, y, z, width=width, height=height, fov_deg=fov_deg,
                                cam_pos=cam_pos, cam_euler=(yaw, pitch, roll))
            for x, y, z in points
        ])
        assert np.all((proj >= 0) & (proj <= 1)), f"Projection out of bounds for yaw={yaw}, pitch={pitch}, roll={roll}"

def test_projection_identity():
    """Validate the origin projects below center when the camera is elevated and unrotated."""
    # With no rotation, center point should project to center
    width, height = 1080, 1920
    fov_deg = 45
    cam_pos = (-5, 0, 2)
    x, y, z = 0, 0, 0
    nx, ny = project_to_keyframe(x, y, z, width=width, height=height, fov_deg=fov_deg, cam_pos=cam_pos, cam_euler=(0,0,0))
    assert 0.4 < nx < 0.6 and ny > 0.6, f"Center point not projected as expected: ({nx}, {ny})"

def test_projection_regression_values():
    """Lock projection outputs for baseline points to detect math regressions."""
    width, height = 1080, 1920
    fov_deg = 45
    cam_pos = (-5, 0, 2)
    cam_euler = (0, 0, 0)
    tol = 1e-4
    center_x = 0.5
    cases = [
        {
            "name": "origin",
            "point": (0, 0, 0),
            "exp_nx": center_x,
            "exp_ny": 0.771599,
        },
        {
            "name": "forward_5",
            "point": (5, 0, 0),
            "exp_nx": center_x,
            "exp_ny": 0.635799,
        },
        {
            "name": "forward_5_right_1",
            "point": (5, 1, 0),
            "exp_nx": 0.620711,
            "exp_ny": 0.635799,
        },
        {
            "name": "forward_5_up_1",
            "point": (5, 0, 1),
            "exp_nx": center_x,
            "exp_ny": 0.567900,
        },
    ]
    for case in cases:
        x, y, z = case["point"]
        nx, ny = project_to_keyframe(
            x,
            y,
            z,
            width=width,
            height=height,
            fov_deg=fov_deg,
            cam_pos=cam_pos,
            cam_euler=cam_euler,
        )
        assert np.isclose(nx, case["exp_nx"], atol=tol), (
            f"{case['name']} nx mismatch for {(x, y, z)}: {nx} vs {case['exp_nx']}"
        )
        assert np.isclose(ny, case["exp_ny"], atol=tol), (
            f"{case['name']} ny mismatch for {(x, y, z)}: {ny} vs {case['exp_ny']}"
        )

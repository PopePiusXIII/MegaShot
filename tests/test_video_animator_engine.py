import numpy as np
import pytest
from golf_ball_trajectory import GolfBallTrajectory
from VideoAnimatorEngine import project_to_keyframe

def test_projected_path_in_bounds():
    """Confirm projected trajectory stays within normalized screen space."""
    T_max = 5.0
    v0 = 75
    launch_angle_deg = 15
    azimuth_deg = -10
    side_spin_rpm = -500
    back_spin_rpm = 3000
    traj = GolfBallTrajectory(
        t_max=T_max,
        x0=0, y0=0, z0=0,
        v0=v0,
        launch_angle_deg=launch_angle_deg,
        azimuth_deg=azimuth_deg,
        side_spin_rpm=side_spin_rpm,
        back_spin_rpm=back_spin_rpm
    )
    traj.sim()
    norm_x = []
    norm_y = []
    for x, y, z in zip(traj.body.x, traj.body.y, traj.body.z):
        nx, ny = project_to_keyframe(x, y, z)
        norm_x.append(nx)
        norm_y.append(ny)
    norm_x = np.array(norm_x)
    norm_y = np.array(norm_y)
    assert np.all((norm_x >= 0) & (norm_x <= 1)), f"Normalized x out of bounds: {norm_x}"
    assert np.all((norm_y >= 0) & (norm_y <= 1)), f"Normalized y out of bounds: {norm_y}"

def test_projected_path_monotonic():
    """Ensure forward motion yields mostly increasing normalized x values."""
    # x should generally increase as the ball moves forward
    T_max = 5.0
    v0 = 75
    launch_angle_deg = 15
    azimuth_deg = 0
    side_spin_rpm = 0
    back_spin_rpm = 3000
    traj = GolfBallTrajectory(
        t_max=T_max,
        x0=0, y0=0, z0=0,
        v0=v0,
        launch_angle_deg=launch_angle_deg,
        azimuth_deg=azimuth_deg,
        side_spin_rpm=side_spin_rpm,
        back_spin_rpm=back_spin_rpm
    )
    traj.sim()
    norm_x = []
    for x, y, z in zip(traj.body.x, traj.body.y, traj.body.z):
        nx, ny = project_to_keyframe(x, y, z)
        norm_x.append(nx)
    norm_x = np.array(norm_x)
    # Check that the normalized x is mostly increasing (allowing for minor floating point noise)
    diffs = np.diff(norm_x)
    assert np.sum(diffs < -1e-3) < 3, f"Normalized x not monotonic: {norm_x}"

import numpy as np
import pytest
from golf_ball_trajectory import GolfBallTrajectory

def test_golf_ball_lands():
    """Check that simulated trajectories cross the ground for typical speeds and spins."""
    # Ball should always land (z crosses zero)
    speeds = [60, 70, 80]
    back_spins = [1500, 3000, 4500]
    for bs in back_spins:
        for v0 in speeds:
            traj = GolfBallTrajectory(
                t_max=15,
                x0=0, y0=0, z0=0,
                v0=v0,
                launch_angle_deg=15,
                azimuth_deg=0,
                side_spin_rpm=0,
                back_spin_rpm=bs
            )
            traj.sim()
            assert np.any(traj.z <= 0), f"Ball did not land for speed={v0}, spin={bs}"

def test_golf_ball_max_height_increases_with_speed():
    """Verify apex height increases with initial speed."""
    # Higher speed should result in higher apex
    v0s = [60, 70, 80]
    heights = []
    for v0 in v0s:
        traj = GolfBallTrajectory(
            t_max=15,
            x0=0, y0=0, z0=0,
            v0=v0,
            launch_angle_deg=15,
            azimuth_deg=0,
            side_spin_rpm=0,
            back_spin_rpm=3000
        )
        traj.sim()
        heights.append(np.max(traj.z))
    assert heights[0] < heights[1] < heights[2], f"Max height does not increase with speed: {heights}"

def test_simulation_deterministic_delta():
    """Ensure repeated simulations with identical inputs are numerically identical."""
    params = dict(
        t_max=8,
        x0=0, y0=0, z0=0,
        v0=70,
        launch_angle_deg=15,
        azimuth_deg=5,
        side_spin_rpm=500,
        back_spin_rpm=3000,
        dt=0.01
    )
    traj_a = GolfBallTrajectory(**params)
    traj_b = GolfBallTrajectory(**params)
    traj_a.sim()
    traj_b.sim()
    max_dx = float(np.max(np.abs(traj_a.x - traj_b.x)))
    max_dy = float(np.max(np.abs(traj_a.y - traj_b.y)))
    max_dz = float(np.max(np.abs(traj_a.z - traj_b.z)))
    assert max_dx < 1e-12 and max_dy < 1e-12 and max_dz < 1e-12, (
        f"Determinism regression: max_dx={max_dx}, max_dy={max_dy}, max_dz={max_dz}"
    )

"""
GPS utilities for converting coordinates to local meters and optimizing a trajectory
that hits a GPS target under a fixed bearing constraint.
"""

from dataclasses import dataclass
import math
import numpy as np

from golf_ball_trajectory import GolfBallTrajectory


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    v0: float
    launch_angle_deg: float
    azimuth_deg: float
    side_spin_rpm: float
    landing_north_m: float
    landing_east_m: float
    target_north_m: float
    target_east_m: float
    error_m: float
    trajectory: GolfBallTrajectory


def _meters_per_deg_lat(lat_deg: float) -> float:
    # Simple approximation, adequate for short distances.
    return 111_320.0


def _meters_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat_deg))


def latlon_to_local_meters(
    lat0_deg: float,
    lon0_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> tuple[float, float]:
    """Convert lat/lon to local meters north/east relative to lat0/lon0."""
    dlat = lat_deg - lat0_deg
    dlon = lon_deg - lon0_deg
    north_m = dlat * _meters_per_deg_lat(lat0_deg)
    east_m = dlon * _meters_per_deg_lon(lat0_deg)
    return north_m, east_m


def _simulate_landing_error(
    target_north_m: float,
    target_east_m: float,
    traj: GolfBallTrajectory,
) -> float:
    traj.sim()
    landing_north_m = float(traj.x[-1])
    landing_east_m = float(traj.y[-1])
    return math.hypot(landing_north_m - target_north_m, landing_east_m - target_east_m)


def optimize_trajectory_to_gps(
    start_lat: float,
    start_lon: float,
    target_lat: float,
    target_lon: float,
    bearing_deg: float,
    *,
    v0_bounds: tuple[float, float] = (30.0, 90.0),
    launch_angle_bounds: tuple[float, float] = (5.0, 30.0),
    side_spin_bounds: tuple[float, float] = (-4000.0, 4000.0),
    v0_steps: int = 13,
    angle_steps: int = 13,
    side_spin_steps: int = 9,
    refine_iters: int = 2,
    back_spin_rpm: float = 3000.0,
    with_drag: bool = True,
    dt: float = 0.01,
    t_max: float | None = None,
) -> TrajectoryOptimizationResult:
    """
    Optimize launch speed and angle to hit a GPS target with fixed azimuth.

    Coordinates are converted to a local tangent plane where:
    - x is North (meters)
    - y is East (meters)
    """
    target_north_m, target_east_m = latlon_to_local_meters(
        start_lat, start_lon, target_lat, target_lon
    )
    target_dist = math.hypot(target_north_m, target_east_m)
    if t_max is None:
        # Ensure enough time for landing across a wide range of speeds.
        t_max = max(8.0, target_dist / max(v0_bounds) * 3.0)

    azimuth_deg = bearing_deg

    best_error = float("inf")
    best_params = None
    v0_min, v0_max = v0_bounds
    ang_min, ang_max = launch_angle_bounds
    side_min, side_max = side_spin_bounds

    for _ in range(refine_iters + 1):
        v0_grid = np.linspace(v0_min, v0_max, v0_steps)
        ang_grid = np.linspace(ang_min, ang_max, angle_steps)
        side_grid = np.linspace(side_min, side_max, side_spin_steps)

        for v0 in v0_grid:
            for launch_angle_deg in ang_grid:
                for side_spin_rpm in side_grid:
                    traj = GolfBallTrajectory(
                        t_max=t_max,
                        x0=0.0,
                        y0=0.0,
                        z0=0.0,
                        v0=float(v0),
                        launch_angle_deg=float(launch_angle_deg),
                        azimuth_deg=float(azimuth_deg),
                        side_spin_rpm=float(side_spin_rpm),
                        back_spin_rpm=float(back_spin_rpm),
                        with_drag=with_drag,
                        dt=dt,
                    )
                    error_m = _simulate_landing_error(target_north_m, target_east_m, traj)
                    if error_m < best_error:
                        best_error = error_m
                        best_params = (float(v0), float(launch_angle_deg), float(side_spin_rpm), traj)

        # Refine search window around current best.
        if best_params is not None:
            best_v0, best_ang, best_side, _ = best_params
            v0_span = (v0_max - v0_min) * 0.35
            ang_span = (ang_max - ang_min) * 0.35
            side_span = (side_max - side_min) * 0.35
            v0_min = max(v0_bounds[0], best_v0 - v0_span)
            v0_max = min(v0_bounds[1], best_v0 + v0_span)
            ang_min = max(launch_angle_bounds[0], best_ang - ang_span)
            ang_max = min(launch_angle_bounds[1], best_ang + ang_span)
            side_min = max(side_spin_bounds[0], best_side - side_span)
            side_max = min(side_spin_bounds[1], best_side + side_span)

    if best_params is None:
        raise RuntimeError("Optimization failed to find a valid trajectory.")

    best_v0, best_ang, best_side, best_traj = best_params
    # Re-run to ensure landing is aligned with best parameters.
    best_traj.sim()
    landing_north_m = float(best_traj.x[-1])
    landing_east_m = float(best_traj.y[-1])

    return TrajectoryOptimizationResult(
        v0=best_v0,
        launch_angle_deg=best_ang,
        azimuth_deg=float(azimuth_deg),
        side_spin_rpm=best_side,
        landing_north_m=landing_north_m,
        landing_east_m=landing_east_m,
        target_north_m=float(target_north_m),
        target_east_m=float(target_east_m),
        error_m=float(math.hypot(landing_north_m - target_north_m, landing_east_m - target_east_m)),
        trajectory=best_traj,
    )


if __name__ == "__main__":
    result = optimize_trajectory_to_gps(
        start_lat=35.101902,
        start_lon=-80.817939,
        target_lat=35.103820,
        target_lon=-80.816993,
        bearing_deg=0.0,
    )

    print("Optimized launch parameters:")
    print(f"  v0: {result.v0:.2f} m/s")
    print(f"  launch_angle_deg: {result.launch_angle_deg:.2f}")
    print(f"  azimuth_deg: {result.azimuth_deg:.2f}")
    print(f"  side_spin_rpm: {result.side_spin_rpm:.0f}")
    print("Landing vs target (meters):")
    print(f"  landing_north_m: {result.landing_north_m:.2f}")
    print(f"  landing_east_m: {result.landing_east_m:.2f}")
    print(f"  target_north_m: {result.target_north_m:.2f}")
    print(f"  target_east_m: {result.target_east_m:.2f}")
    print(f"  error_m: {result.error_m:.2f}")

    import matplotlib.pyplot as plt

    traj = result.trajectory
    x_ft = traj.x * 3.28084
    y_ft = traj.y * 3.28084
    z_ft = traj.z * 3.28084
    target_north_ft = result.target_north_m * 3.28084
    target_east_ft = result.target_east_m * 3.28084

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_ft, z_ft, color="tab:blue", label="Trajectory")
    ax1.set_xlabel("x (ft)")
    ax1.set_ylabel("z (ft)")
    ax1.set_title("Side View (x-z)")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.set_ylim(0, max(150, float(z_ft.max()) * 1.1))

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x_ft, y_ft, color="tab:green", label="Trajectory")
    ax2.scatter([target_north_ft], [target_east_ft], color="tab:red", marker="x", s=60, label="Target")
    ax2.set_xlabel("x (ft)")
    ax2.set_ylabel("y (ft)")
    ax2.set_title("Top View (x-y)")
    ax2.grid(True)
    ax2.axis("equal")
    ax2.legend(loc="best")

    plt.suptitle(
        f"Optimized Trajectory (v0={result.v0:.1f} m/s, angle={result.launch_angle_deg:.1f} deg)\n"
        f"Landing error: {result.error_m:.2f} m"
    )
    plt.tight_layout()
    plt.show()



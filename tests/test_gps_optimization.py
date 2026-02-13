import math

from gps import optimize_trajectory_to_gps


def test_optimize_trajectory_to_gps_smoke():
    result = optimize_trajectory_to_gps(
        start_lat=35.101902,
        start_lon=-80.817939,
        target_lat=35.103820,
        target_lon=-80.816993,
        bearing_deg=0.0,
    )

    assert math.isfinite(result.error_m)
    assert result.trajectory.x.size > 1
    assert result.trajectory.y.size == result.trajectory.x.size
    assert result.trajectory.z.size == result.trajectory.x.size

    expected_error = math.hypot(
        result.landing_north_m - result.target_north_m,
        result.landing_east_m - result.target_east_m,
    )
    assert math.isclose(result.error_m, expected_error, rel_tol=1e-9, abs_tol=1e-9)

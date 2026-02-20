import json
import os
import numpy as np
import pytest
from golf_ball_trajectory import GolfBallTrajectory
from VideoAnimatorEngine import VideoAnimationEngine, project_to_keyframe

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


def _load_regression_fixture() -> dict:
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "video_animator_main_regression.json",
    )
    if not os.path.exists(fixture_path):
        pytest.fail(f"Missing regression fixture: {fixture_path}")

    with open(fixture_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def _require_value(name: str, value):
    if value is None:
        pytest.fail(f"Fixture value '{name}' is missing. Fill it in tests/data/video_animator_main_regression.json")
    return value


def test_main_workflow_regression_from_fixture():
    data = _load_regression_fixture()

    video_rel_path = _require_value("video", data.get("video"))
    start_keyframe_time = float(_require_value("start_keyframe_time", data.get("start_keyframe_time")))

    trajectory_cfg = _require_value("trajectory", data.get("trajectory"))
    projection_cfg = _require_value("projection", data.get("projection"))
    expected = _require_value("expected", data.get("expected"))

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    video_path = os.path.abspath(os.path.join(root_dir, video_rel_path))
    if not os.path.exists(video_path):
        pytest.fail(f"Video file not found: {video_path}")

    engine = VideoAnimationEngine(video_path)

    frame = engine.get_frame_at_time(start_keyframe_time)
    assert frame is not None

    detection = engine.find_golf_ball_in_keyframe(frame)
    assert detection is not None

    det_x, det_y, _ = detection
    frame_h, frame_w = frame.shape[:2]
    first_golf_point = (det_x / float(frame_w), det_y / float(frame_h))

    traj = GolfBallTrajectory(
        t_max=float(_require_value("trajectory.t_max", trajectory_cfg.get("t_max"))),
        x0=float(_require_value("trajectory.x0", trajectory_cfg.get("x0"))),
        y0=float(_require_value("trajectory.y0", trajectory_cfg.get("y0"))),
        z0=float(_require_value("trajectory.z0", trajectory_cfg.get("z0"))),
        v0=float(_require_value("trajectory.v0", trajectory_cfg.get("v0"))),
        launch_angle_deg=float(_require_value("trajectory.launch_angle_deg", trajectory_cfg.get("launch_angle_deg"))),
        azimuth_deg=float(_require_value("trajectory.azimuth_deg", trajectory_cfg.get("azimuth_deg"))),
        side_spin_rpm=float(_require_value("trajectory.side_spin_rpm", trajectory_cfg.get("side_spin_rpm"))),
        back_spin_rpm=float(_require_value("trajectory.back_spin_rpm", trajectory_cfg.get("back_spin_rpm"))),
        dt=float(_require_value("trajectory.dt", trajectory_cfg.get("dt"))),
    )
    traj.sim()

    sim_start_x, sim_start_y = project_to_keyframe(
        traj.body.x[0],
        traj.body.y[0],
        traj.body.z[0],
        width=int(_require_value("projection.width", projection_cfg.get("width"))),
        height=int(_require_value("projection.height", projection_cfg.get("height"))),
        fov_deg=float(_require_value("projection.fov_deg", projection_cfg.get("fov_deg"))),
        cam_pos=tuple(_require_value("projection.cam_pos", projection_cfg.get("cam_pos"))),
        cam_euler=tuple(_require_value("projection.cam_euler", projection_cfg.get("cam_euler"))),
    )

    offset_x = first_golf_point[0] - sim_start_x
    offset_y = first_golf_point[1] - sim_start_y

    keyframes = []
    for i in range(len(traj.t)):
        proj_x, proj_y = project_to_keyframe(
            traj.body.x[i],
            traj.body.y[i],
            traj.body.z[i],
            width=int(_require_value("projection.width", projection_cfg.get("width"))),
            height=int(_require_value("projection.height", projection_cfg.get("height"))),
            fov_deg=float(_require_value("projection.fov_deg", projection_cfg.get("fov_deg"))),
            cam_pos=tuple(_require_value("projection.cam_pos", projection_cfg.get("cam_pos"))),
            cam_euler=tuple(_require_value("projection.cam_euler", projection_cfg.get("cam_euler"))),
        )
        keyframes.append(
            (
                float(np.clip(proj_x + offset_x, 0.0, 1.0)),
                float(np.clip(proj_y + offset_y, 0.0, 1.0)),
            )
        )

    expected_keyframes = _require_value("expected.keyframes", expected.get("keyframes"))
    keyframe_tol = float(_require_value("expected.keyframe_tol", expected.get("keyframe_tol")))

    expected_count = int(_require_value("expected.keyframes.count", expected_keyframes.get("count")))
    assert len(keyframes) == expected_count

    expected_first = _require_value("expected.keyframes.first", expected_keyframes.get("first"))
    expected_middle = _require_value("expected.keyframes.middle", expected_keyframes.get("middle"))
    expected_last = _require_value("expected.keyframes.last", expected_keyframes.get("last"))

    first_idx = 0
    middle_idx = len(keyframes) // 2
    last_idx = len(keyframes) - 1

    assert abs(keyframes[first_idx][0] - float(_require_value("expected.keyframes.first.x", expected_first.get("x")))) <= keyframe_tol
    assert abs(keyframes[first_idx][1] - float(_require_value("expected.keyframes.first.y", expected_first.get("y")))) <= keyframe_tol

    assert abs(keyframes[middle_idx][0] - float(_require_value("expected.keyframes.middle.x", expected_middle.get("x")))) <= keyframe_tol
    assert abs(keyframes[middle_idx][1] - float(_require_value("expected.keyframes.middle.y", expected_middle.get("y")))) <= keyframe_tol

    assert abs(keyframes[last_idx][0] - float(_require_value("expected.keyframes.last.x", expected_last.get("x")))) <= keyframe_tol
    assert abs(keyframes[last_idx][1] - float(_require_value("expected.keyframes.last.y", expected_last.get("y")))) <= keyframe_tol

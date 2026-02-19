"""
Animated azimuth sweep: renders a new golf ball trajectory for each frame, with azimuth smoothly varying from 0 to -15 to +15 degrees over 3 seconds.
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Allow running from repo root without installation.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from golf_ball_trajectory import GolfBallTrajectory
from VideoAnimatorEngine import VideoAnimationEngine, project_to_keyframe

def smooth_azimuth(t, T):
    # t in [0, T], returns azimuth in degrees: 0 -> -15 -> +15
    # Use a sine wave for smoothness
    # At t=0: 0 deg, t=T/2: -15 deg, t=T: +15 deg
    return -15 * np.sin(np.pi * (t / T) - np.pi/2)

if __name__ == "__main__":
    video_path = os.path.join(REPO_ROOT, "tests", "data", "videos", "golfVideo.MOV")
    output_video_path = "golfAzimuthSweep.mp4"
    T_max = 3.0  # seconds (video duration)
    traj_t_max = 8.0  # seconds (trajectory simulation duration, should be long enough for ball to land)
    fps = 30
    samples = int(T_max * fps)
    v0 = 70
    launch_angle_deg = 15
    side_spin_rpm = 0
    back_spin_rpm = 3000
    dt_traj = 0.01

    engine = VideoAnimationEngine(video_path)

    # Store all trajectories for plotting
    all_trajs = []
    all_azimuths = []

    for frame_idx in range(samples):
        t = frame_idx / fps
        azimuth = smooth_azimuth(t, T_max)
        # Only recompute trajectory if azimuth changes significantly
        if frame_idx == 0 or abs(azimuth - prev_azimuth) > 0.2:
            traj = GolfBallTrajectory(
                t_max=traj_t_max,
                x0=0, y0=0, z0=0,
                v0=v0,
                launch_angle_deg=launch_angle_deg,
                azimuth_deg=azimuth,
                side_spin_rpm=side_spin_rpm,
                back_spin_rpm=back_spin_rpm,
                dt=dt_traj
            )
            traj.sim()
            prev_azimuth = azimuth
            all_trajs.append((traj.body.x.copy(), traj.body.y.copy(), traj.body.z.copy()))
            all_azimuths.append(azimuth)
        # Add all trajectory points up to current time as keyframes
        engine.clear_keyframes()
        for i in range(len(traj.t)):
            norm_x, norm_y = project_to_keyframe(traj.body.x[i], traj.body.y[i], traj.body.z[i])
            color = (0, 255, 255)
            brush_size = 1.0
            engine.add_keyframe({'timestamp': traj.t[i], 'x': norm_x, 'y': norm_y, 'color': color, 'brush_size': brush_size})
        # Save overlay for this frame
        frame = engine.get_frame_at_time(t)
        if frame is not None:
            overlay = engine.render_full_trajectory_overlay(frame, color=(0, 255, 255), brush_size=1.0)
            if frame_idx == 0:
                h, w = overlay.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            out_vid.write(overlay)
    out_vid.release()
    print(f"Saved azimuth sweep animation as {output_video_path}")

    # 3D plot of all computed trajectories
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for (x, y, z), az in zip(all_trajs, all_azimuths):
        ax.plot(x, y, z, label=f'Az={az:.1f}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Golf Ball Trajectories (Azimuth Sweep)')
    ax.set_box_aspect([1,1,1])  # axis equal
    ax.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from golf_ball_trajectory import GolfBallTrajectory
from VideoAnimatorEngine import project_to_keyframe

# Test parameters
T_max = 5.0
v0 = 75  # m/s
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

# Project to normalized keyframe coordinates
norm_x = []
norm_y = []
for x, y, z in zip(traj.x, traj.y, traj.z):
    nx, ny = project_to_keyframe(x, y, z)
    norm_x.append(nx)
    norm_y.append(ny)
norm_x = np.array(norm_x)
norm_y = np.array(norm_y)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(traj.t, norm_x, label='Normalized x')
plt.plot(traj.t, norm_y, label='Normalized y')
plt.xlabel('Time (s)')
plt.ylabel('Normalized coordinate')
plt.title('Normalized x and y vs Time')
plt.legend()
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(norm_x, norm_y)
plt.xlabel('Normalized x')
plt.ylabel('Normalized y')
plt.title('Normalized x vs y (Projected Path)')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(traj.x, norm_x)
plt.xlabel('x (m)')
plt.ylabel('Normalized x')
plt.title('x vs Normalized x')
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(traj.y, norm_y)
plt.xlabel('y (m)')
plt.ylabel('Normalized y')
plt.title('y vs Normalized y')
plt.grid(True)

plt.tight_layout()
plt.show()

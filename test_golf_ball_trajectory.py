import matplotlib.pyplot as plt
import numpy as np
from golf_ball_trajectory import GolfBallTrajectory

# Test script: 3 speeds x 3 back spin rates
speeds = [60, 70, 80]  # m/s
back_spins = [1500, 3000, 4500]  # rpm
colors = ['b', 'g', 'r']  # Same color for same spin
linestyles = ['-', '--', ':']  # Different style for each speed

plt.figure(figsize=(12, 6))
for j, bs in enumerate(back_spins):
    for i, v0 in enumerate(speeds):
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
        x_ft = traj.x * 3.28084
        z_ft = traj.z * 3.28084
        label = f"{bs} rpm, {v0} m/s" if i == 0 else None  # Only label once per spin
        plt.plot(x_ft, z_ft, color=colors[j], linestyle=linestyles[i], label=label)

plt.xlabel('x (ft)')
plt.ylabel('z (ft)')
plt.title('Golf Ball Trajectories: Effect of Speed and Spin')
plt.grid(True)
plt.axis('equal')
plt.ylim(0, 150)
plt.legend(title='Spin (rpm), Speed (m/s)')
plt.tight_layout()
plt.show()

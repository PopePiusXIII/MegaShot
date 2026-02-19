import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from three_d_body import ThreeDBody

def Cd_func(spin_rpm):
    # Cd increases with spin rate: 0.18 (low spin) to 0.30 (high spin)
    spin_norm = min(max(max((spin_rpm-2000), 0) / 2000, 0), 2)
    return 0.18 + 0.32* spin_norm  # linear increase with spin

# Lift coefficient as a function of spin rate (module level)
def C_lift_func(spin_rpm):
    # C_lift increases with spin rate, typical range: 0.08 (low spin) to 0.25 (high spin)
    spin_norm = min(max(spin_rpm / 4000, 0), 2)
    return 0.0001 + 0.00085 * spin_norm  # linear increase with spin


def rk4_step(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class GolfBallTrajectory:
    def __init__(self, t_max, x0, y0, z0, v0, launch_angle_deg, azimuth_deg, side_spin_rpm=0, back_spin_rpm=3000, g=9.81, with_drag=True, dt=0.01, use_external_drag_funcs=False, ext_Cd_func=None, ext_C_lift_func=None):
        self.dt = dt
        self.t_max = t_max
        self.n_steps = int(t_max / dt) + 1
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.v0 = v0
        self.launch_angle_deg = launch_angle_deg
        self.azimuth_deg = azimuth_deg
        self.side_spin_rpm = side_spin_rpm
        self.back_spin_rpm = back_spin_rpm
        self.g = g
        self.with_drag = with_drag
        self.use_external_drag_funcs = use_external_drag_funcs
        self.ext_Cd_func = ext_Cd_func
        self.ext_C_lift_func = ext_C_lift_func
        # Drag parameters
        self.rho = 1.225
        self.r = 0.02135
        self.A = np.pi * self.r**2
        self.m = 0.04593
        self.tau_spin = 8.0
        # Allocate arrays (will be filled in sim)
        self.t = np.zeros(self.n_steps)
        self.body = ThreeDBody(
            name="golf_ball",
            x=np.zeros(self.n_steps),
            y=np.zeros(self.n_steps),
            z=np.zeros(self.n_steps),
        )
        self.vx = np.zeros(self.n_steps)
        self.vy = np.zeros(self.n_steps)
        self.vz = np.zeros(self.n_steps)
        self.spin = np.zeros(self.n_steps)
        self.cl = np.zeros(self.n_steps)
        self.cd = np.zeros(self.n_steps)

    def sim(self, body_name: str = "golf_ball") -> ThreeDBody:
        theta = np.radians(self.launch_angle_deg)
        phi = np.radians(self.azimuth_deg)
        vx = self.v0 * np.cos(theta) * np.cos(phi)
        vy = self.v0 * np.cos(theta) * np.sin(phi)
        vz = self.v0 * np.sin(theta)
        state = np.array([self.x0, self.y0, self.z0, vx, vy, vz, self.side_spin_rpm, self.back_spin_rpm])
        for i in range(self.n_steps):
            px, py, pz, vx_, vy_, vz_, omega_side_rpm, omega_back_rpm = state
            v = np.sqrt(vx_**2 + vy_**2 + vz_**2)
            spin_rpm = np.sqrt(omega_side_rpm**2 + omega_back_rpm**2)
            if self.use_external_drag_funcs and self.ext_Cd_func and self.ext_C_lift_func:
                Cd = self.ext_Cd_func(spin_rpm)
                Cl = self.ext_C_lift_func(spin_rpm)
            else:
                Cd = Cd_func(spin_rpm)
                Cl = C_lift_func(spin_rpm)
            self.t[i] = i * self.dt
            self.body.x[i] = px
            self.body.y[i] = py
            self.body.z[i] = pz
            self.vx[i] = vx_
            self.vy[i] = vy_
            self.vz[i] = vz_
            self.spin[i] = spin_rpm
            self.cl[i] = Cl
            self.cd[i] = Cd
            if pz <= 0 and i > 0:
                self.body.x = self.body.x[:i+1]
                self.body.y = self.body.y[:i+1]
                self.body.z = self.body.z[:i+1]
                self.vx = self.vx[:i+1]
                self.vy = self.vy[:i+1]
                self.vz = self.vz[:i+1]
                self.spin = self.spin[:i+1]
                self.cl = self.cl[:i+1]
                self.cd = self.cd[:i+1]
                self.t = self.t[:i+1]
                break
            state = rk4_step(self._deriv, state, self.t[i], self.dt)
            if state[2] < 0:
                state[2] = 0

        return self.to_3d_body(body_name)

    def to_3d_body(self, name: str = "golf_ball") -> ThreeDBody:
        self.body.name = name
        return self.body

    def _deriv(self, state, t):
        px, py, pz, vx_, vy_, vz_, omega_side_rpm, omega_back_rpm = state
        v = np.sqrt(vx_**2 + vy_**2 + vz_**2)
        omega_side = (2 * np.pi * omega_side_rpm) / 60
        omega_back = (2 * np.pi * omega_back_rpm) / 60
        spin_rpm = np.sqrt(omega_side_rpm**2 + omega_back_rpm**2)
        if self.use_external_drag_funcs and self.ext_Cd_func and self.ext_C_lift_func:
            Cd = self.ext_Cd_func(spin_rpm)
            Cl = self.ext_C_lift_func(spin_rpm)
        else:
            Cd = Cd_func(spin_rpm)
            Cl = C_lift_func(spin_rpm)
        if self.with_drag and v > 0:
            Fd = 0.5 * self.rho * Cd * self.A * v**2
            ax_drag = -Fd * vx_ / (self.m * v)
            ay_drag = -Fd * vy_ / (self.m * v)
            az_drag = -Fd * vz_ / (self.m * v)
        else:
            ax_drag = ay_drag = az_drag = 0
        az_magnus = Cl * omega_back * v
        ay_magnus = Cl * omega_side * v
        ax = ax_drag
        ay = ay_drag + ay_magnus
        az = az_drag - self.g + az_magnus
        domega_side_rpm = -omega_side_rpm / self.tau_spin
        domega_back_rpm = -omega_back_rpm / self.tau_spin
        return np.array([vx_, vy_, vz_, ax, ay, az, domega_side_rpm, domega_back_rpm])

    def log_variables(self):
        return {
            't': self.t,
            'x': self.body.x,
            'y': self.body.y,
            'z': self.body.z,
            'vx': self.vx,
            'vy': self.vy,
            'vz': self.vz,
            'spin': self.spin,
            'cl': self.cl,
            'cd': self.cd
        }


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Create and run the simulation
    traj = GolfBallTrajectory(
        t_max=8,
        x0=0, y0=0, z0=0,
        v0=70,
        launch_angle_deg=15,
        azimuth_deg=0,
        side_spin_rpm=2000,
        back_spin_rpm=3000
    )
    traj.sim()
    result = traj.log_variables()
    x_ft = result['x'] * 3.28084
    y_ft = result['y'] * 3.28084
    z_ft = result['z'] * 3.28084
    t_vals = result['t']
    cl_vals = result['cl']
    cd_vals = result['cd']
    spin_vals = result['spin']
    # Trajectory plots
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(x_ft, z_ft)
    plt.xlabel('x (ft)')
    plt.ylabel('z (ft)')
    plt.title('Side View (x-z)')
    plt.grid(True)
    plt.axis('equal')
    plt.ylim(0, 150)
    plt.subplot(1,2,2)
    plt.plot(x_ft, y_ft)
    plt.xlabel('x (ft)')
    plt.ylabel('y (ft)')
    plt.title('Top View (x-y)')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    # Cl, Cd, and spin plots
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(t_vals, cl_vals)
    plt.xlabel('Time (s)')
    plt.ylabel('C_lift')
    plt.title('Lift Coefficient vs Time')
    plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(t_vals, cd_vals)
    plt.xlabel('Time (s)')
    plt.ylabel('C_drag')
    plt.title('Drag Coefficient vs Time')
    plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(t_vals, spin_vals)
    plt.xlabel('Time (s)')
    plt.ylabel('Spin (rpm)')
    plt.title('Total Spin vs Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Print final position in feet
    print(f"Final position: x={x_ft[-1]:.2f} ft, y={y_ft[-1]:.2f} ft, z={z_ft[-1]:.2f} ft")

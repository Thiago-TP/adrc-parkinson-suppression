import scipy
import numpy as np
from system import System, ModelParameters


class ADRControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
        omega_c: float = 0.135
    ) -> None:
        super().__init__(name, params, ic)
        
        # Proportional, derivative gains
        self.kp = omega_c ** 2
        self.kd = 2 * omega_c

        # Error based Extended State Observer gains and states
        omega_o = 5 * omega_c
        self.lambda1 = 3 * omega_o
        self.lambda2 = 3 * (omega_o ** 2)
        self.lambda3 = omega_o ** 3
        self.xe1_hat = 0.0
        self.xe2_hat = 0.0
        self.z = 0.0
        
        j33 = self.j[2, 2]
        j31 = self.j[2, 0]
        j23 = self.j[1, 2]
        j12 = self.j[0, 1]
        j = j33 - (j31 * j23 / j12)
        self.j_adrc = j
        return

    def control(self) -> np.ndarray:
        # ADRC models dynamic as a cascated integrator 
        # dx^n/dt = u + zeta,
        # where zeta is considered a disturbance that may depend on theta.
        # Original EDO on theta3 is 
        # j*dy^2/dt + c*dy/dt + ky = u3 + f
        # And so, in the ADRC framework,
        # dy^2/dt = u3/j + (f-c*dy/dt - ky) / j
        # leading to
        # u = u3/j
        # zeta = (f-c*dy/dt + ky) / j

        # Tracking error and its derivative
        e = np.array(self.theta_v) - np.array(self.theta)
        u_adrc = self.u[-1][2] / self.j_adrc
        xe1 = e[-1, 2] # error on theta3
        xe2 = np.diff(e, axis=0)[-1, 2]  if len(e) > 1 else 0.0
        
        # Extended State Observer of error
        dxe1_hat = self.xe2_hat + self.lambda1 * (xe1 - self.xe1_hat)
        dxe2_hat = -u_adrc + self.z + self.lambda2 * (xe2 - self.xe2_hat)
        dz = self.lambda3 * (xe1 - self.xe1_hat)

        self.xe1_hat += dxe1_hat
        self.xe2_hat += dxe2_hat
        self.z += dz

        # Control (PD)
        new_u_adrc = (self.kp * self.xe1_hat) + (self.kd * self.xe2_hat) + self.z
        u3 = new_u_adrc * self.j_adrc

        # Estimation of voluntary torque applied at wrist
        # dx = ax + bu
        # btbu = bt(dx-ax)
        # u = (btb)^-1 bt (dx-ax)
        dx = np.diff(self.x, axis=0) if len(self.x) > 1 else np.array([0., 0., 0., 0., 0., 0.])
        bt_binv = np.linalg.inv(self.b.T @ self.b)
        tau = [bt_binv @ (self.b.T @ (x_dot - self.a @ x)) for x, x_dot in zip(self.x, dx)]
        # print(tau)

        fs = 1 / self.dt  # sampling frequency in Hz
        f_cutoff = 1.0
        b, a = scipy.signal.butter(
            N=4, Wn=2 * np.pi * f_cutoff, fs=fs, btype="low", analog=False
        )
        # Apply the filter to each column
        try:
            tau3_v_hat = scipy.signal.filtfilt(b, a, tau, axis=0)[-1][-1]
        except ValueError:
            tau3_v_hat = tau[-1][-1]
        self.u.append(np.array([0.0, 0.0, u3 + tau3_v_hat]))

        return self.u[-1]

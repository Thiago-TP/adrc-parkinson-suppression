import scipy
import numpy as np
from system import System, ModelParameters


class ADRControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
        omega_c: float = 0.8
    ) -> None:
        super().__init__(name, params, ic)
        
        # Proportional, derivative gains
        self.kp = omega_c ** 2
        self.kd = 2 * omega_c / self.dt

        # Error based Extended State Observer gains and states
        omega_o = 5 * omega_c
        self.lambda1 = 3 * omega_o
        self.lambda2 = 3 * (omega_o ** 2)
        self.lambda3 = omega_o ** 3
        self.xe1_hat = 0.0
        self.xe2_hat = 0.0
        self.z = 0.0

        return


    def control(self) -> np.ndarray:
        # j*dy2/dt2 + c*dy/dt + k*y = u3 + tau_v + tau_i + f
        # dy2/dt2 = (u3 + tau_v)/j + zeta
        # dy2/dt2 = u3' + zeta
        # u3 = ju3' - tau_v

        # Last control output
        u3_old = self.u[-1][2]

        # Tracking error 
        e = np.array(self.theta_v) - np.array(self.theta_true)
        xe1 = e[-1, 2] # error on theta3
        delta_xe1 = xe1 - self.xe1_hat

        # Estimation of voluntary torque
        # tau3_v_hat = self._estimate_voluntary_torque()
        
        # Extended State Observer of error
        dxe1_hat = self.xe2_hat + self.lambda1 * delta_xe1
        dxe2_hat = - u3_old \
                   + self.z \
                   + self.lambda2 * delta_xe1
        dz = self.lambda3 * delta_xe1

        self.xe1_hat += (dxe1_hat * self.dt)
        self.xe2_hat += (dxe2_hat * self.dt)
        self.z += (dz * self.dt)

        # Control (PD)
        u3_adrc = (self.kp * self.xe1_hat) + (self.kd * self.xe2_hat) + self.z
        u3 = u3_adrc

        # Update of control history
        self.u.append(np.array([0.0, 0.0, u3]))

        return self.u[-1]
    
    def _estimate_voluntary_torque(self) -> np.ndarray:
        # dx = Ax + Bu
        # Bu = dx - Ax
        # (B^TB)u = B^T (dx-Ax)
        # u = (B^TB)^-1 B^T (dx - Ax)
        #   = g (dx - Ax)
        # If u3 ~ tau_i, then u ~ tau_v
        g = np.linalg.inv(self.b.T @ self.b) @ self.b.T
        x_hat_dot = np.diff(self.x_hat, axis=0, prepend=[self.x_hat[0]])
        tau_v_hat = g @ (x_hat_dot[-1] - self.a @ self.x_hat[-1])
        tau3_v_hat = tau_v_hat[2]
        return tau3_v_hat

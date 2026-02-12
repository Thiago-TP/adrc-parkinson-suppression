import numpy as np
from system import System, ModelParameters


class PIDControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
        kp: float = 2.0,
        ki: float = 4.0,
        kd: float = 0.42
    ) -> None:
        super().__init__(name, params, ic)
        
        # Proportional, integral, derivative gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Error coefficients
        self.a0 = self.kp + (self.ki * self.dt) + (self.kd / self.dt)
        self.a1 = -self.kp - (2 * self.kd / self.dt)
        self.a2 = (self.kd / self.dt)
        self.a_vec = np.array([self.a0, self.a1, self.a2])
        
        # Tracking error of wrist angle
        self.e = np.array([0.0, 0.0, 0.0])
        return

    def control(self) -> np.ndarray:
        # PID control with fixed gains

        self.e[2] = self.e[1]
        self.e[1] = self.e[0]
        self.e[0] = (self.theta_v[-1] - self.theta[-1])[-1]
        update = np.array([0.0, 0.0, np.dot(self.a_vec, self.e)])
        self.u.append(self.u[-1] + update)

        # self.u[-1] = np.array([0.0, 0.0, 0.0]) # emergency button
        return self.u[-1]
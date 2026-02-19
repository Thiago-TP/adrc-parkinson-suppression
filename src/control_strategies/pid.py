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
        self.ki = ki * self.dt
        self.kd = kd / self.dt
        
        # Errors for calculating control
        self.error_control = 0.0
        self.error_sum = 0.0
        self.error_delta = 0.0
        self.error_previous = 0.0
        
        return

    def control(self) -> np.ndarray:
        # PID control with fixed gains
        # For more details, check out 
        # https://alphaville.github.io/qub/pid-101/#/

        self.error_control = self.theta_v_hat[-1][2] - self.theta_filtered[-1][2]
        self.error_delta = self.error_control - self.error_previous

        u3 = np.dot([self.kp, self.ki, self.kd], 
                    [self.error_control, self.error_sum, self.error_delta])

        self.error_sum += self.error_control
        self.error_previous = self.error_control

        self.u.append(np.array([0.0, 0.0, u3]))

        return self.u[-1]
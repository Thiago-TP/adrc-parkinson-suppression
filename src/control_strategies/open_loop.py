import numpy as np
from system import System, ModelParameters, InitialConditions


class OpenLoopControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: InitialConditions,
        amplitude_voluntary: float = 1.0,
    ) -> None:
        super().__init__(name, params, ic,
                         amplitude_voluntary=amplitude_voluntary)
        return

    def _control(self) -> np.ndarray:
        # Null control signal in open loop
        control = np.array([0.0, 0.0, 0.0])
        self.u.append(control)  # register control signal to history
        return self.u[-1]

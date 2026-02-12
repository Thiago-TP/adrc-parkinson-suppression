import numpy as np
from system import System, ModelParameters


class OpenLoopControl(System):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
    ) -> None:
        super().__init__(name, params, ic)
        return

    def control(self) -> np.ndarray:
        # Null control signal in open loop
        control = np.array([0.0, 0.0, 0.0])
        self.u.append(control) # register control signal to history
        return self.u[-1]

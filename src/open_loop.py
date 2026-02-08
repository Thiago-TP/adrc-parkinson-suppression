from model import Model, ModelParameters

import numpy as np


class OpenLoopModel(Model):
    def __init__(
        self,
        name: str,
        params: ModelParameters,
        ic: tuple[float],
    ) -> None:
        super().__init__(name, params, ic)
        return

    def control(self, t: float) -> np.ndarray:
        # No control signal in open loop
        self.u.append(np.array([0.0, 0.0, 0.0]))
        return self.u[-1]

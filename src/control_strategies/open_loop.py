from system import System, ModelParameters

import numpy as np


class OpenLoopModel(System):
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
        return np.array([0.0, 0.0, 0.0])

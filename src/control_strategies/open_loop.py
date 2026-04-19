
from system import InitialConditions, ModelParameters, System


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

    def _update_control(self, k: int) -> None:
        # Null control signal in open loop
        self.u[k, 2] = 0.0

    def _update_estimates(self, k: int) -> None:
        # No estimation in open loop control
        return

    def _reset_control_variables(self) -> None:
        # Nothing to reset
        return

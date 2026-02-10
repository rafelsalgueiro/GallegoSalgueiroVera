import numpy as np
from .arm import Arm

class BernoulliArm(Arm):
    """
    Brazo que sigue una distribución de Bernoulli (éxito o fracaso).
    """
    def __init__(self, p: float):
        """
        :param p: Probabilidad de éxito (recompensa = 1).
        """
        self.p = p
        # La media de una Bernoulli es simplemente p
        super().__init__(p)

    def draw(self) -> float:
        """
        Retorna 1 con probabilidad p, y 0 en caso contrario.
        """
        return 1.0 if np.random.random() < self.p else 0.0
import numpy as np
from .arm import Arm

class BinomialArm(Arm):
    """
    Brazo que sigue una distribución Binomial B(n, p).
    """
    def __init__(self, n: int, p: float):
        """
        :param n: Número de ensayos.
        :param p: Probabilidad de éxito en cada ensayo.
        """
        self.n = n
        self.p = p
        # La media de una Binomial es n * p
        super().__init__(n * p)

    def draw(self) -> float:
        """
        Genera una recompensa basada en la distribución binomial.
        """
        return float(np.random.binomial(self.n, self.p))
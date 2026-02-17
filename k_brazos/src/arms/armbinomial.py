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

    def pull(self) -> float:
        """
        Retorna una recompensa basada en la distribución binomial.
        """
        return float(np.random.binomial(self.n, self.p)) / self.n  # Normalizamos para que la recompensa esté entre 0 y 1
    
    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Binomial.
        :return: Valor esperado de la distribución.
        """
        return self.n * self.p
    
    def __str__(self):
        """
        Representación en cadena del brazo de Binomial.
        :return: Descripción detallada del brazo de Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"
    
    def generate_arms(k: int, n: int = 10, min_p: float = 0.0, max_p: float = 1.0) -> list['BinomialArm']:
        """
        Genera una lista de brazos de Binomial con probabilidades distribuidas uniformemente entre min_p y max_p.
        :param k: Número de brazos a generar.
        :param n: Número de ensayos para cada brazo.
        :param min_p: Probabilidad mínima de éxito.
        :param max_p: Probabilidad máxima de éxito.
        :return: Lista de brazos de Binomial.
        """
        return [BinomialArm(n, p) for p in np.linspace(min_p, max_p, k)]
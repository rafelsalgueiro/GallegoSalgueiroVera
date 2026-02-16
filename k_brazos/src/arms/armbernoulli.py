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

    def pull(self) -> float:
        """
        Retorna 1 con probabilidad p, y 0 en caso contrario.
        """
        return 1.0 if np.random.random() < self.p else 0.0

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.

        :return: Valor esperado de la distribución.
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo de Bernoulli.

        :return: Descripción detallada del brazo de Bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"
        
    @classmethod
    def generate_arms(cls, k: int, min_p: float = 0.0, max_p: float = 1.0) -> list['BernoulliArm']:
        """
        Genera una lista de brazos de Bernoulli con probabilidades distribuidas uniformemente entre min_p y max_p.

        :param k: Número de brazos a generar.
        :param min_p: Probabilidad mínima de éxito.
        :param max_p: Probabilidad máxima de éxito.
        :return: Lista de brazos de Bernoulli.
        """
        return [cls(p) for p in np.linspace(min_p, max_p, k)]
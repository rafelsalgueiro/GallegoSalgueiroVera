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
        
    def get_lai_robbins_term(self, optimal_value: float) -> float:
        """
        Calcula el término de Lai-Robbins para este brazo de Bernoulli.

        :param optimal_value: El valor esperado (probabilidad p) del brazo óptimo.
        :return: El término de Lai-Robbins para este brazo.
        """
        delta = optimal_value - self.p
        
        # Si este brazo es óptimo (o casi) el término es 0
        if delta <= 1e-9:
            return 0.0
            
        p = self.p
        q = optimal_value
        
        # Calcular Kullback-Leibler div (KL(p || q))
        # Hacemos manejo para log(0)
        # KL(p||q) = p * ln(p/q) + (1-p) * ln((1-p)/(1-q))
        
        kl_div = 0.0
        
        if p > 0:
            if q == 0:
                kl_div = float('inf')
            else:
                kl_div += p * np.log(p / q)
                
        if p < 1:
            if q == 1:
                kl_div = float('inf')
            else:
                kl_div += (1 - p) * np.log((1 - p) / (1 - q))
                
        if kl_div == 0.0 or kl_div == float('inf'):
            return 0.0
            
        return delta / kl_div

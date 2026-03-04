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
        Retorna una recompensa basada en la distribución binomial. Rango de recompensa normalizado [0, 1].
        """
        return float(np.random.binomial(self.n, self.p)) / self.n  # Normalizamos la recompensa al rango [0, 1]
    
    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Binomial normalizado al rango [0, 1].
        :return: Valor esperado de la distribución.
        """
        return self.p
    
    def __str__(self):
        """
        Representación en cadena del brazo de Binomial.
        :return: Descripción detallada del brazo de Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"
    
    @classmethod
    def generate_arms(cls, k: int, n: int = 10, min_p: float = 0.0, max_p: float = 1.0) -> list['BinomialArm']:
        """
        Genera una lista de brazos de Binomial con probabilidades distribuidas uniformemente entre min_p y max_p.
        :param k: Número de brazos a generar.
        :param n: Número de ensayos para cada brazo.
        :param min_p: Probabilidad mínima de éxito.
        :param max_p: Probabilidad máxima de éxito.
        :return: Lista de brazos de Binomial.
        """
        return [cls(n, p) for p in np.linspace(min_p, max_p, k)]

    def get_lai_robbins_term(self, optimal_value: float) -> float:
        """
        Calcula el término de Lai-Robbins para este brazo Binomial.
        
        Como pull() devuelve X/n donde X ~ Bin(n,p), la distribución por pull es
        equivalente a la media de n ensayos Bernoulli. La KL divergence es:
        KL(Bin(n,p_a)/n || Bin(n,p*)/n) = n * KL_Bernoulli(p_a || p*)
        
        El término de Lai-Robbins es: delta / KL = (p* - p_a) / (n * KL_Bern(p_a || p*))

        :param optimal_value: El valor esperado (p) del brazo óptimo.
        :return: El término de Lai-Robbins para este brazo.
        """
        delta = optimal_value - self.p
        
        if delta <= 1e-9:
            return 0.0
        
        p = self.p
        q = optimal_value
        
        # KL divergence de Bernoulli: KL(p || q)
        kl_bern = 0.0
        if p > 0:
            if q == 0:
                kl_bern = float('inf')
            else:
                kl_bern += p * np.log(p / q)
        if p < 1:
            if q == 1:
                kl_bern = float('inf')
            else:
                kl_bern += (1 - p) * np.log((1 - p) / (1 - q))
                
        # La KL para Bin(n,p)/n es n veces la KL de Bernoulli
        kl_div = self.n * kl_bern
        
        if kl_div == 0.0 or kl_div == float('inf'):
            return 0.0
        
        return delta / kl_div

import numpy as np

from .arm import Arm


class ArmNormal(Arm):
    def __init__(self, mu: float, sigma: float):
        """
        Inicializa el brazo con distribución normal.

        :param mu: Media de la distribución.
        :param sigma: Desviación estándar de la distribución.
        """
        assert sigma > 0, "La desviación estándar sigma debe ser positiva."

        self.mu = mu
        self.sigma = sigma

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución normal.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.normal(self.mu, self.sigma)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución normal.

        :return: Valor esperado de la distribución.
        """

        return self.mu

    def __str__(self):
        """
        Representación en cadena del brazo normal.

        :return: Descripción detallada del brazo normal.
        """
        return f"ArmNormal(mu={self.mu}, sigma={self.sigma})"

    @classmethod
    def generate_arms(cls, k: int, mu_min: float = 1, mu_max: float = 10.0, sigma: float = 1.0) -> list:
        """
        Genera k brazos con medias únicas en el rango [mu_min, mu_max].

        :param k: Número de brazos a generar.
        :param mu_min: Valor mínimo de la media.
        :param mu_max: Valor máximo de la media.
        :param sigma: Desviación estándar de los brazos generados.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert mu_min < mu_max, "El valor de mu_min debe ser menor que mu_max."

        # Generar k- valores únicos de mu con decimales
        mu_values = set()
        while len(mu_values) < k:
            mu = np.random.uniform(mu_min, mu_max)
            mu = round(mu, 2)
            mu_values.add(mu)

        mu_values = list(mu_values)

        arms = [ArmNormal(mu, sigma) for mu in mu_values]

        return arms
    
    def get_lai_robbins_term(self, optimal_value: float) -> float:
        """
        Calcula el término de Lai-Robbins para este brazo.

        :param optimal_value: El valor esperado del brazo óptimo.
        :return: El término de Lai-Robbins para este brazo.
        """
        delta = optimal_value - self.mu

        if delta <= 1e-9:
            return 0.0
        
        # 2 * sigma^2 / delta, asumiendo que el óptimo tiene la misma desviación estándar
        return 2 * (self.sigma ** 2) / delta
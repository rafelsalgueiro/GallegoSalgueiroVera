import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Clase base abstracta para los agentes del Bandido de k-brazos.
    Define la interfaz que deben seguir todos los algoritmos.
    """
    
    def __init__(self, k_arms: int, name: str = "GenericAgent"):
        """
        Inicializa los parámetros básicos del agente.
        
        :param k_arms: Número de brazos disponibles (acciones).
        :param name: Nombre identificativo del algoritmo.
        """
        self.k_arms = k_arms
        self.name = name
        self.n_steps = 0  # Contador de pasos totales
        
        # Inicialización de estadísticas por brazo
        self.arm_counts = np.zeros(k_arms)  # Veces que se ha elegido cada brazo
        self.arm_rewards = np.zeros(k_arms) # Recompensa acumulada por brazo
        self.values = np.zeros(k_arms)      # Estimación del valor Q (promedio)
        
    @abstractmethod
    def get_action(self) -> int:
        """
        Selecciona una acción siguiendo la política del agente.
        Este método debe ser implementado por cada algoritmo específico.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def update(self, action: int, reward: float):
        """
        Actualiza el conocimiento del agente tras recibir una recompensa.
        Utiliza la actualización incremental de la media.
        
        :param action: El brazo que fue seleccionado.
        :param reward: La recompensa obtenida.
        """
        self.n_steps += 1
        self.arm_counts[action] += 1
        
        # Fórmula de actualización incremental para el valor estimado (Q)
        # Q_{n+1} = Q_n + 1/n * (R_n - Q_n)
        step_size = 1.0 / self.arm_counts[action]
        self.values[action] += step_size * (reward - self.values[action])
        self.arm_rewards[action] += reward
        
    def reset(self):
        """
        Reinicia las estadísticas del agente para una nueva simulación.
        Es vital para mantener la independencia entre experimentos.
        """
        self.n_steps = 0
        self.arm_counts.fill(0)
        self.arm_rewards.fill(0)
        self.values.fill(0)
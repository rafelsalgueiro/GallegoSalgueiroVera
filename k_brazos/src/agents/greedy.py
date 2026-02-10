import numpy as np
from .agent import Agent

class EpsilonGreedyAgent(Agent):
    def __init__(self, k_arms: int, epsilon: float = 0.1):
        super().__init__(k_arms, name=f"EpsilonGreedy (e={epsilon})")
        self.epsilon = epsilon

    def get_action(self) -> int:
        # Exploración: elegir brazo al azar
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k_arms)
        
        # Explotación: elegir el mejor brazo (valor Q más alto)
        # Se usa argmax con desempate aleatorio para evitar sesgos
        return np.random.choice(np.where(self.values == np.max(self.values))[0])
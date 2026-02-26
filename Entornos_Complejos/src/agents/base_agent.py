import numpy as np
import random
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Esquema general del agente tabular para Gymnasium."""
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """Inicializa todo lo necesario para el aprendizaje."""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {} 

    def get_q_value(self, state, action):
        """Devuelve el valor Q de un par estado-acción. Si no existe, devuelve 0.0."""
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist())
        return self.q_table.get((state, action), 0.0)

    @abstractmethod
    def get_action(self, state):
        """Indicará qué acción realizar de acuerdo al estado."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """Aplica el algoritmo de aprendizaje del agente."""
        pass
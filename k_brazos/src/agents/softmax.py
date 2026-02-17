import numpy as np
from .agent import Agent

class SoftmaxAgent(Agent):
    def __init__(self, k_arms: int, temperature: float = 0.1):
        super().__init__(k_arms, name=f"Softmax (tau={temperature})")
        self.temperature = temperature

    def get_action(self) -> int:
        # Evitar división por cero o temperaturas extremadamente bajas
        tau = max(self.temperature, 1e-5)
        
        preferences = self.values / tau
    
        max_val = np.max(preferences)
        exp_values = np.exp(preferences - max_val)
        probs = exp_values / np.sum(exp_values)
        
        return np.random.choice(self.k_arms, p=probs)
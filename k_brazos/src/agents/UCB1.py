import numpy as np
from .agent import Agent

class UCBAgent(Agent):
    def __init__(self, k_arms: int, c: float = 2.0):
        super().__init__(k_arms, name=f"UCB1 (c={c})")
        self.c = c

    def get_action(self) -> int:
        # Obligatorio: Probar cada brazo al menos una vez antes de aplicar la fórmula
        if 0 in self.arm_counts:
            return np.where(self.arm_counts == 0)[0][0]

        # Fórmula UCB1: Q(a) + c * sqrt(ln(t) / n_a)
        exploration_term = self.c * np.sqrt(np.log(self.n_steps) / self.arm_counts)
        ucb_values = self.values + exploration_term
        
        return np.random.choice(np.where(ucb_values == np.max(ucb_values))[0])
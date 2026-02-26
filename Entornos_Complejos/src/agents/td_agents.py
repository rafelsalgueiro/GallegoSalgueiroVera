import random
from .base_agent import BaseAgent

class TDAgent(BaseAgent):
    """Clase base para agentes de Diferencias Temporales con política Epsilon-Greedy."""
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Buscar la acción con el mayor valor Q
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
        max_q = max(q_values)
        
        # Romper empates de forma aleatoria si varias acciones tienen el mismo valor Q máximo
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)
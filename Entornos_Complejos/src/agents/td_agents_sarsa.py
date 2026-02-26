import numpy as np
from .td_agents import TDAgent

class SarsaAgent(TDAgent):
    """Agente SARSA"""
    
    def update(self, state, action, reward, next_state, next_action, done):
        if isinstance(state, np.ndarray): state = tuple(state.tolist())
        if isinstance(next_state, np.ndarray): next_state = tuple(next_state.tolist())

        current_q = self.get_q_value(state, action)
        next_q = self.get_q_value(next_state, next_action) if not done else 0.0
        
        # Ecuación de actualización de SARSA
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[(state, action)] = new_q

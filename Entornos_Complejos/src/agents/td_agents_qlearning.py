import numpy as np
from .td_agents import TDAgent

class QLearningAgent(TDAgent):
    """Agente Q-Learning"""
    
    def update(self, state, action, reward, next_state, done):
        if isinstance(state, np.ndarray): state = tuple(state.tolist())
        if isinstance(next_state, np.ndarray): next_state = tuple(next_state.tolist())

        current_q = self.get_q_value(state, action)
        
        if done:
            max_next_q = 0.0
        else:
            max_next_q = max([self.get_q_value(next_state, a) for a in range(self.env.action_space.n)])
        
        # Ecuación de actualización de Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
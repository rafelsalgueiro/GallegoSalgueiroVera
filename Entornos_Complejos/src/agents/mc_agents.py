import numpy as np
import random
from collections import defaultdict
from .base_agent import BaseAgent

class MCOnPolicyAgent(BaseAgent):
    """Agente Monte Carlo On-Policy"""
    
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        super().__init__(env, alpha=0.0, gamma=gamma, epsilon=epsilon)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.episode_memory = [] 

    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def store_transition(self, state, action, reward):
        """Guarda la transición durante el episodio."""
        if isinstance(state, np.ndarray): state = tuple(state.tolist())
        self.episode_memory.append((state, action, reward))

    def update(self):
        """Se ejecuta al final del episodio para procesar las recompensas hacia atrás."""
        G = 0
        for state, action, reward in reversed(self.episode_memory):
            G = self.gamma * G + reward
            sa_pair = (state, action)
            
            self.returns_sum[sa_pair] += G
            self.returns_count[sa_pair] += 1.0
            self.q_table[sa_pair] = self.returns_sum[sa_pair] / self.returns_count[sa_pair]
        
        # Limpiar memoria para el próximo episodio
        self.episode_memory = []

class MCOffPolicyAgent(BaseAgent):
    """Agente Monte Carlo Off-Policy usando Muestreo de Importancia."""
    
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        super().__init__(env, alpha=0.0, gamma=gamma, epsilon=epsilon)
        self.c_table = defaultdict(float)
        self.episode_memory = []

    def get_action(self, state):
        # Política de comportamiento (b): Epsilon-Greedy para asegurar exploración
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def store_transition(self, state, action, reward):
        if isinstance(state, np.ndarray): state = tuple(state.tolist())
        self.episode_memory.append((state, action, reward))

    def update(self):
        G = 0.0
        W = 1.0 # Peso de importancia inicial
        
        for state, action, reward in reversed(self.episode_memory):
            G = self.gamma * G + reward
            sa_pair = (state, action)
            
            # C(s,a) acumula los pesos
            self.c_table[sa_pair] += W
            
            # Actualizamos la tabla Q
            current_q = self.get_q_value(state, action)
            self.q_table[sa_pair] = current_q + (W / self.c_table[sa_pair]) * (G - current_q)
            
            # Buscamos la acción óptima (política objetivo)
            q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
            best_actions = [a for a, q in enumerate(q_values) if q == max(q_values)]
            
            # Si la acción tomada no coincide con la política objetivo, el peso es 0 y se corta la cadena
            if action not in best_actions:
                break 
            
            # Actualizamos W dividiendo por la probabilidad de la política de comportamiento
            # Probabilidad de tomar la acción greedy en epsilon-greedy:
            prob_b = (1 - self.epsilon) + (self.epsilon / self.env.action_space.n)
            W = W / prob_b
            
        self.episode_memory = []
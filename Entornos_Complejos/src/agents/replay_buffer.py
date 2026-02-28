import numpy as np
import random
import torch

class ReplayBuffer:
    """Buffer de repetición de experiencias tipo memoria cíclica."""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        # Pre-alocando memoria para más rendimiento
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_actions = np.zeros((capacity, 1), dtype=np.int64)
        self.dones = np.zeros((capacity, 1), dtype=bool)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, next_action, done):
        """Añade una experiencia al buffer, sobreescribiendo si está lleno."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_actions[self.ptr] = next_action
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Retorna un bloque aleatorio de experiencias como tensores."""
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[idxs])
        actions = torch.LongTensor(self.actions[idxs])
        rewards = torch.FloatTensor(self.rewards[idxs])
        next_states = torch.FloatTensor(self.next_states[idxs])
        next_actions = torch.LongTensor(self.next_actions[idxs])
        dones = torch.BoolTensor(self.dones[idxs])
        
        return states, actions, rewards, next_states, next_actions, dones

    def __len__(self):
        return self.size

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy


class DQNNet(nn.Module):
    """
    Aproximador de la función de valor Q(s, a) mediante un Perceptrón Multicapa (MLP).
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim=128):
        """
        Inicializa la arquitectura de la red neuronal.

        Args:
            input_dim (int): Dimensión del vector de estado.
            output_dim (int): Número de acciones discretas posibles.
            hidden_dim (int | list): Neuronas por capa oculta. Si es un entero, 
                                     crea dos capas de ese tamaño.
        """
        super().__init__()
        
        hidden_dims = [hidden_dim, hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagación hacia adelante para estimar los valores Q."""
        return self.net(x)


class DQNReplayBuffer:
    """
    Memoria circular para almacenar transiciones y romper la correlación temporal 
    durante el entrenamiento (Experience Replay).
    """
    def __init__(self, capacity: int, state_dim: int):
        """
        Inicializa los arrays de NumPy preasignados en memoria por eficiencia.

        Args:
            capacity (int): Número máximo de transiciones a almacenar.
            state_dim (int): Dimensión del vector de estado.
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Añade una transición a la memoria, sobrescribiendo las más antiguas si está llena."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """
        Extrae un lote aleatorio de transiciones y las convierte de forma segura a tensores.
        """
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        # Uso de torch.tensor() en lugar de torch.FloatTensor() para un manejo de memoria seguro
        return (
            torch.tensor(self.states[idxs], dtype=torch.float32),
            torch.tensor(self.actions[idxs], dtype=torch.int64),
            torch.tensor(self.rewards[idxs], dtype=torch.float32),
            torch.tensor(self.next_states[idxs], dtype=torch.float32),
            torch.tensor(self.dones[idxs], dtype=torch.bool),
        )

    def __len__(self) -> int:
        return self.size


class DQNAgent:
    """
    Agente Deep Q-Network independiente (Stand-alone) con soporte para 
    Double DQN, Experience Replay y actualizaciones suaves/duras de la red objetivo.
    """
    def __init__(
        self,
        env,
        alpha=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        hidden_dim=128,
        use_target_network=True,
        target_update_freq=1000,
        grad_clip=1.0,
        double_dqn=False,
        tau=None,
        device="cpu"
    ):
        """
        Configura los hiperparámetros de aprendizaje y las redes neuronales.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)
        self.double_dqn = double_dqn
        self.tau = tau

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Red principal (Policy)
        self.q_net = DQNNet(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.criterion = nn.SmoothL1Loss() 

        # Red objetivo (Target)
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self._optim_steps = 0

        if self.use_target_network:
            self.target_net = copy.deepcopy(self.q_net).to(self.device)
            self.target_net.eval()  
        else:
            self.target_net = self.q_net  

    def decay_epsilon(self):
        """Aplica el decaimiento exponencial a la tasa de exploración epsilon."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, state: np.ndarray, action: int = None) -> np.ndarray:
        """Infiere los valores Q para un estado dado sin computar gradientes."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).cpu().numpy()[0]
        
        if action is not None:
            return q_values[action]
        return q_values

    def get_action(self, state: np.ndarray) -> int:
        """Selecciona una acción siguiendo la política epsilon-greedy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.get_q_value(state)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return random.choice(best_actions)

    def train_step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                   next_states: torch.Tensor, dones: torch.Tensor) -> float:
        """
        Calcula la pérdida TD sobre un lote de transiciones y actualiza los pesos 
        de la red principal mediante retropropagación.
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.q_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                best_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q_max = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                next_q_max = self.target_net(next_states).max(1)[0]

        target_q = rewards.squeeze(1) + self.gamma * next_q_max * (~dones.squeeze(1))

        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
            
        self.optimizer.step()
        self._optim_steps += 1
        self._maybe_update_target()

        return loss.item()

    def _maybe_update_target(self):
        """Sincroniza la red objetivo mediante soft update (Polyak) o hard update."""
        if not self.use_target_network:
            return
            
        if self.tau is not None:
            # Uso de torch.no_grad() para no interferir con el grafo de computación
            with torch.no_grad():
                for tp, mp in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(1.0 - self.tau)
                    tp.data.add_(self.tau * mp.data)
        elif self._optim_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save_weights(self, path: str):
        """Persiste los pesos de la red principal en disco."""
        torch.save(self.q_net.state_dict(), path)

    def load_weights(self, path: str) -> bool:
        """Restaura los pesos desde disco hacia la red principal y objetivo."""
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, weights_only=True))
            if self.use_target_network:
                self.target_net.load_state_dict(self.q_net.state_dict())
            self.q_net.eval()
            return True
        return False
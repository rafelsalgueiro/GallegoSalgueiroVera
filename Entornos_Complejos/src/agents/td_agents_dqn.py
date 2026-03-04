import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy

from .td_agents import TDAgent


class DQNNet(nn.Module):
    """
    Red neuronal para aproximar la función de valor Q(s, a) en un algoritmo Deep Q-Network.
    Utiliza un Perceptrón Multicapa (MLP) estándar con funciones de activación ReLU.
    Intenta predecir la recompensa esperada de tomar cada acción posible en un estado dado.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        """
        Inicializa la red neuronal.

        :param input_dim: Dimensión del espacio de estados (entradas). Ej: 12 para Flappy Bird.
        :param output_dim: Número total de acciones posibles (salidas). Ej: 2 (caer, saltar).
        :param hidden_dim: Número de neuronas en las capas ocultas.
        """
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante (forward pass) de la red para calcular los Valores Q.

        :param x: Tensor de PyTorch que representa el estado o un lote (batch) de estados.
        :return: Tensor con los Valores Q estimados para cada acción posible.
        """
        return self.net(x)


class DQNReplayBuffer:
    """
    Almacena las transiciones del agente para romper la correlación temporal 
    de los datos durante el entrenamiento, estabilizando así el aprendizaje.
    
    Al ser DQN un algoritmo off-policy, no necesitamos almacenar la siguiente acción.
    """
    def __init__(self, capacity: int, state_dim: int):
        """
        Inicializa el buffer de repetición.

        :param capacity: Capacidad máxima del búfer (número de transiciones a recordar).
        :param state_dim: Dimensión del vector de estado del entorno.
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=bool)

        self.ptr = 0  # Siguiente transición
        self.size = 0 # Cantidad actual de elementos almacenados

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Añade una nueva transición al buffer. Si el bufer está lleno, sobrescribe 
        la transición más antigua (cola circular).

        :param state: Estado actual observado por el agente.
        :param action: Acción tomada por el agente en el estado actual.
        :param reward: Recompensa recibida tras ejecutar la acción.
        :param next_state: Siguiente estado observado tras la transición.
        :param done: Booleano que indica si el episodio ha terminado (terminal o truncado).
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Extrae un batch aleatorio de transiciones para entrenar la red neuronal.
        Convierte automáticamente los arrays de NumPy en Tensores de PyTorch.

        :param batch_size: Tamaño del lote de datos a extraer.
        :return: Una tupla de tensores de PyTorch: 
                 (estados, acciones, recompensas, siguientes_estados, dones).
        """
        # Selección aleatoria de índices sin reemplazo
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.LongTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.BoolTensor(self.dones[idxs]),
        )

    def __len__(self) -> int:
        """
        Devuelve el número actual de transiciones almacenadas en el búfer.
        Permite usar la función len(buffer) directamente sobre el objeto.
        """
        return self.size


class DQNAgent(TDAgent):
    """
    Agente Deep Q-Network (DQN) con red objetivo (Target Network) opcional, 
    Experience Replay y decaimiento de exploración (epsilon-decay).
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
        device="cpu"
    ):
        """Inicializa el agente DQN con sus hiperparámetros y redes neuronales.
        
        :param env: Entorno de con espacios de observación y acción definidos.
        :param alpha: Tasa de aprendizaje para el optimizador de la red neuronal.
        :param gamma: Factor de descuento para el cálculo de la recompensa futura.
        :param epsilon_start: Valor inicial de epsilon para la política epsilon-greedy.
        :param epsilon_min: Valor mínimo de epsilon para asegurar algo de exploración.
        :param epsilon_decay: Factor multiplicativo para reducir epsilon después de cada episodio.
        :param hidden_dim: Número de neuronas en las capas ocultas de la red neuronal
        :param use_target_network: Si True, utiliza una red objetivo separada para estabilizar el entrenamiento.
        :param target_update_freq: Frecuencia (en pasos de optimización) para actualizar la red objetivo.
        :param grad_clip: Valor máximo para el recorte de gradientes (None para no usar).
        :param device: Dispositivo para ejecutar el modelo ("cpu" o "cuda").
        """
        super().__init__(env, alpha, gamma, epsilon_start)
        
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

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
        """Reduce epsilon multiplicándolo por su factor de decaimiento hasta el límite mínimo."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, state, action=None):
        """Calcula los Valores Q del estado actual usando la red principal (sin gradientes)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).cpu().numpy()[0]
        
        if action is not None:
            return q_values[action]
        return q_values

    def get_action(self, state):
        """Selecciona una acción siguiendo una política epsilon-greedy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.get_q_value(state)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, done):
        """Utiliza la transición para actualizar los pesos de la red principal."""
        states = torch.FloatTensor(np.array([state]))
        actions = torch.LongTensor([[action]])
        rewards = torch.FloatTensor([[reward]])
        next_states = torch.FloatTensor(np.array([next_state]))
        dones = torch.BoolTensor([[done]])
        return self.update_batch(states, actions, rewards, next_states, dones)

    def update_batch(self, states, actions, rewards, next_states, dones):
        """Actualiza los pesos de la red principal usando un lote de transiciones (Experience Replay)."""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Predicciones actuales Q(s, a)
        current_q = self.q_net(states).gather(1, actions).squeeze(1)

        # Valor máximo esperado en el siguiente estado: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]

        # Objetivo TD: r + gamma * max_a' Q(s', a')
        target_q = rewards.squeeze(1) + self.gamma * next_q_max * (~dones.squeeze(1))

        # Cálculo de pérdida y retropropagación
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
        """Copia los pesos de la red principal a la red objetivo según la frecuencia definida."""
        if self.use_target_network and self._optim_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def sync_target_network(self):
        """Fuerza la sincronización manual inmediata de la red objetivo."""
        if self.use_target_network:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save_weights(self, path):
        """Guarda los pesos del modelo en disco."""
        torch.save(self.q_net.state_dict(), path)
        print(f"Pesos guardados en {path}")

    def load_weights(self, path):
        """Carga los pesos desde el disco en la red principal y en la objetivo."""
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, weights_only=True))
            if self.use_target_network:
                self.target_net.load_state_dict(self.q_net.state_dict())
            self.q_net.eval()
            print(f"Pesos cargados desde {path}")
            return True
        else:
            print(f"Error: Ruta {path} no encontrada.")
            return False
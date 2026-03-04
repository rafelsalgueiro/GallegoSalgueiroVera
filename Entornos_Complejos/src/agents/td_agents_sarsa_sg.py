import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

from .td_agents import TDAgent

class QNet(nn.Module):
    """Red neuronal somera para aproximar la función de valor Q(s, a)"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class SemiGradientSarsaAgent(TDAgent):
    """
    Agente SARSA con Aproximación de Valor Lineal/Red Neuronal (Semi-Gradiente).
    Ideal para espacios de estado continuos.
    """
    def __init__(self, env, alpha=1e-3, gamma=0.99, epsilon=0.1, hidden_dim=64):
        super().__init__(env, alpha, gamma, epsilon)
        
        # El espacio de estado en FlappyBird sin lidar es de 12 dimensiones
        self.state_dim = env.observation_space.shape[0] 
        self.action_dim = env.action_space.n
        
        # Red Neuronal y Optimizador
        self.q_net = QNet(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        # Función de pérdida MSE
        self.criterion = nn.MSELoss()

    def get_q_value(self, state, action=None):
        """
        Devuelve el valor Q estimado por la red neuronal.
        Si action es None, devuelve los valores Q para todas las acciones.
        """
        # Convertir estado a tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # Batch size de 1
        
        # Evitando cálculo de gradientes solo si solo estamos prediciendo (no entrenando)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).numpy()[0]
            
        if action is not None:
            return q_values[action]
        return q_values

    def get_action(self, state):
        """Epsilon-greedy con los valores de la red"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
            
        q_values = self.get_q_value(state)
        # Romper empates de forma aleatoria (menos común en NN pero posible inicial)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        """Aplicación de un paso de Semi-Gradient SARSA"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # q_hat(S, A, w) calculando gradientes
        current_q_values = self.q_net(state_tensor)
        current_q = current_q_values[0, action]
        
        # q_hat(S', A', w) sin calcular gradiente
        if done:
            target_q = torch.tensor(reward, dtype=torch.float32)
        else:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                next_q = self.q_net(next_state_tensor)[0, next_action]
            target_q = torch.tensor(reward, dtype=torch.float32) + self.gamma * next_q
            
        # Calcular Loss y actualizar
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_batch(self, states, actions, rewards, next_states, next_actions, dones):
        """
        Calcula el optimizador utilizando un lote (batch) de memorias enviadas 
        por un ReplayBuffer, reduciendo la varianza y estabilizando la convergencia.
        """
        # Calcular los Q-values de los estados actuales Q(S, A, w)
        current_q_values = self.q_net(states)
        current_q = current_q_values.gather(1, actions).squeeze(1)
        
        # Calcular los Q-values de los siguientes estados (Target On-Policy SARSA) Q(S', A', w)
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            next_q = next_q_values.gather(1, next_actions).squeeze(1)
            
        # Cuando el episodio termina, next_q value debe ser 0
        target_q = rewards.squeeze(1) + self.gamma * next_q * (~dones.squeeze(1))
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def save_weights(self, path):
        """Guarda el state_dict de la red neuronal"""
        torch.save(self.q_net.state_dict(), path)
        print(f"Pesos guardados en {path}")
        
    def load_weights(self, path):
        """Carga el state_dict si existe"""
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path))
            # Modo evaluación (sin afecto real en MSE por no tener dropout/batchnorm, pero es buena práctica)
            self.q_net.eval()
            print(f"Pesos cargados correctamente desde {path}")
            return True
        else:
            print(f"Error: ruta {path} no encontrada.")
            return False

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium  # Necesario para que Gym reconozca el entorno
from gymnasium.wrappers import RecordVideo

# ==============================================================================
# 1. CONSTANTES E HIPERPARÁMETROS
# ==============================================================================
# Ajusta estas rutas si tus carpetas están en otro sitio respecto a este script
MODELS_DIR = "./models"
RESULTS_DIR = "./results/videos"

HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Estos parámetros son necesarios para instanciar el agente, aunque en 
# modo evaluación (épsilon=0) realmente no los va a usar para aprender.
LR = 1e-4
GAMMA = 0.99
TARGET_UPDATE_FREQ = 10

# ==============================================================================
# 2. DEFINICIÓN DE LA RED Y EL AGENTE (Solo para Inferencia)
# ==============================================================================
# Copiamos la estructura exacta que definiste en tu proyecto (12 -> 64 -> 64 -> 2)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        # Definimos la red exactamente igual que en tu Notebook original (con nn.Sequential)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, hidden_dim, use_target_network, target_update_freq, device):
        self.device = device
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.epsilon = epsilon_start  # Para grabar, esto será 0.0

    def get_action(self, state):
        # Modo greedy (explotación pura)
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def load_weights(self, path):
        if os.path.exists(path):
            # map_location asegura que si entrenaste en GPU y evalúas en CPU, no falle
            self.q_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            return True
        return False

# ==============================================================================
# 3. BUCLE PRINCIPAL DE GRABACIÓN
# ==============================================================================
if __name__ == "__main__":
    print("=== INICIANDO GRABACIÓN DEL AGENTE ===")

    # Asegurarnos de que el directorio de vídeos existe
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Elige el modelo que quieres cargar (cámbialo según necesites)
    model_filename = "dqn_Agente_Definitivo.pth" 
    model_path = os.path.join(MODELS_DIR, model_filename)

    # Crear entorno puro con render_mode="rgb_array" para poder grabar
    print("Instanciando entorno FlappyBird-v0...")
    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)

    # Envolver el entorno para grabar en vídeo
    env = RecordVideo(
        env, 
        video_folder=RESULTS_DIR, 
        episode_trigger=lambda x: True, # Graba todos los episodios que juegue
        name_prefix="flappy_gameplay"
    )

    # Instanciar el agente forzando epsilon a 0.0 (cero exploración aleatoria)
    agent = DQNAgent(
        env=env, alpha=LR, gamma=GAMMA, epsilon_start=0.0, 
        epsilon_min=0.0, epsilon_decay=1.0, hidden_dim=HIDDEN_DIM, 
        use_target_network=True, target_update_freq=TARGET_UPDATE_FREQ, device=DEVICE
    )

    if agent.load_weights(model_path):
        print(f"Pesos cargados correctamente desde {model_filename}.")
        print("Grabando durante 1 minuto de reloj. Por favor, espera...")
        
        start_time = time.time()
        ep = 0
        
        # Bucle basado en tiempo: 60 segundos
        while time.time() - start_time < 60:
            state, _ = env.reset()
            done = False
            ep += 1
            print(f"  -> Grabando intento {ep}...")
            
            while not done:
                action = agent.get_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Cortar en seco si pasamos del minuto
                if time.time() - start_time >= 60:
                    break
                    
        print("\n¡Se alcanzó el límite de 1 minuto de grabación!")
    else:
        print(f"\n[ERROR CRÍTICO] No se encontró el archivo de pesos en:\n{model_path}")
        print("Revisa que la constante MODELS_DIR apunte a la carpeta correcta.")

    env.close()
    print(f"=== GRABACIÓN FINALIZADA ===")
    print(f"Tus vídeos .mp4 están listos en: {RESULTS_DIR}")
from typing import List, Dict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Configuración visual global
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def plot_episode_lengths(episode_lengths: List[int], title: str = "Longitud del Episodio en el Tiempo", window_size: int = 50):
    if window_size is not None and (window_size <= 0 or window_size > len(episode_lengths)):
        print("Aviso: Tamaño de ventana inválido para plot_episode_lengths. Omitiendo suavizado.")
        window_size = 1

    plt.figure(figsize=(14, 7))
    plt.plot(episode_lengths, color='lightgray', alpha=0.6, label='Pasos reales')
    
    if window_size > 1:
        smoothed = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_lengths)), smoothed, color='red', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.xlabel('Episodios (t)')
    plt.ylabel('Pasos por Episodio')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparative_curves(data_dict: Dict[str, List[float]], window_size: int = 50, title: str = "Comparativa de Rendimiento", ylabel: str = "Métrica"):
    plt.figure(figsize=(14, 7))
    colores = sns.color_palette("muted", n_colors=len(data_dict))
    
    for i, (agent_name, data) in enumerate(data_dict.items()):
        if len(data) >= window_size and window_size > 1:
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(data)), smoothed, linewidth=2, label=agent_name, color=colores[i])
        else:
            plt.plot(data, linewidth=2, label=agent_name, color=colores[i])
            
    plt.xlabel('Episodios')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_rewards(rewards: List[float], window_size: int = 50, title: str = "Recompensa Acumulada por Episodio"):
    if window_size is not None and (window_size <= 0 or window_size > len(rewards)):
         print("Aviso: Tamaño de ventana inválido para plot_rewards. Omitiendo suavizado.")
         window_size = 1

    plt.figure(figsize=(14, 7))
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Recompensa bruta')
    
    if window_size > 1:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), smoothed, color='green', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Total')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_rewards(labels: List[str], data_rewards: List[List[float]], window_size: int = 50, title: str = "Comparativa de Recompensa Acumulada"):
    plt.figure(figsize=(14, 7))
    
    for lbl, rewards in zip(labels, data_rewards):
        current_window = window_size
        
        if current_window is not None and (current_window <= 0 or current_window > len(rewards)):
            print(f"Aviso: Tamaño de ventana inválido para {lbl}. Omitiendo suavizado.")
            current_window = 1

        line = plt.plot(rewards, alpha=0.15) 
        color = line[0].get_color() 
        
        if current_window > 1:
            smoothed = np.convolve(rewards, np.ones(current_window)/current_window, mode='valid')
            plt.plot(range(current_window-1, len(rewards)), smoothed, color=color, linewidth=2, label=lbl)
        else:
            line[0].set_alpha(0.8)
            line[0].set_label(lbl)

    plt.xlabel('Episodios')
    plt.ylabel('Recompensa Total')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_success_rate(successes: List[int], window_size: int = 50, title: str = "Tasa de Éxito Promedio"):
    plt.figure(figsize=(14, 7))
    
    if len(successes) >= window_size and window_size > 1:
        success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid') * 100
        plt.plot(range(window_size-1, len(successes)), success_rate, color='orange', linewidth=2, label=f'Tasa de éxito (n={window_size})')
    
    plt.xlabel('Episodios')
    plt.ylabel('Tasa de Éxito (%)')
    plt.title(title)
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_robust_learning_curves(data_dict: Dict[str, np.ndarray], window_size: int = 50, title: str = "Rendimiento Promedio (Múltiples Semillas)", ylabel: str = "Recompensa Acumulada"):
    plt.figure(figsize=(14, 7))
    colores = sns.color_palette("muted", n_colors=len(data_dict))
    
    for i, (agent_name, data_matrix) in enumerate(data_dict.items()):
        num_seeds, num_episodes = data_matrix.shape
        
        if window_size > 1 and num_episodes >= window_size:
            smoothed_data = np.array([np.convolve(seed_data, np.ones(window_size)/window_size, mode='valid') for seed_data in data_matrix])
            x_axis = range(window_size - 1, num_episodes)
        else:
            smoothed_data = data_matrix
            x_axis = range(num_episodes)
            
        mean_rewards = np.mean(smoothed_data, axis=0)
        std_rewards = np.std(smoothed_data, axis=0)
        
        plt.plot(x_axis, mean_rewards, linewidth=2, label=f"{agent_name} (Media de {num_seeds} semillas)", color=colores[i])
        plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, color=colores[i], alpha=0.2)

    plt.xlabel('Episodios')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_dqn_loss(losses: List[float], window_size: int = 100, title: str = "Evolución de la Pérdida (Loss) de la Red Neuronal"):
    if not losses:
        print("No hay datos de pérdida para graficar.")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(losses, color='lightgray', alpha=0.4, label='Pérdida media por episodio')
    
    if window_size > 1 and len(losses) >= window_size:
        smoothed_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(losses)), smoothed_loss, color='purple', linewidth=2, label=f'Media móvil (n={window_size})')

    plt.xlabel('Episodios')
    plt.ylabel('Loss')
    plt.title(title)
    
    if max(losses) > 10 * np.mean(losses):
        plt.yscale('log')
        plt.ylabel('Loss (Log Scale)')
        
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_q_values(avg_q_values: List[float], title: str = "Evolución de las Estimaciones de Valor Q (Flappy Bird)"):
    plt.figure(figsize=(14, 7))
    plt.plot(avg_q_values, color='teal', linewidth=2, label='Q-Value Máximo Promedio')
    plt.xlabel('Episodios')
    plt.ylabel('Valor Q Estimado')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_flappy_bird_agent(agent, env, num_episodes=3, render=True, fps=30):
    try:
        # Gymnasium devuelve (estado, info) en el reset
        _ = env.reset()
    except Exception as e:
        print(f"Error inicializando gráfica del entorno: {str(e)}")
        return
    
    # Guardar epsilon de forma segura (Duck Typing: si no lo tiene, asumimos 0)
    original_eps = getattr(agent, 'epsilon', 0.0)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0 

    try:
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            frames = 0
            
            while not done:
                action = agent.get_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frames += 1
                
                # O si se forzó un entorno sin render_mode, usamos un delay manual para visualización de consola/custom.
                if render and hasattr(env, 'render'):
                    env.render()
                    time.sleep(1.0 / fps)
                    
            # El bucle while termina naturalmente cuando done == True
            score = info.get('score', 0) if isinstance(info, dict) else 0
            print(f"[Visual-Ep {ep+1}] Agente colisionó en el timestep {frames}. Puntuación final: {score}")
            
    finally:
        if hasattr(agent, 'epsilon'):
            agent.epsilon = original_eps
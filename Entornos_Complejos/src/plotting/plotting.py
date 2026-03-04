from typing import List, Dict, Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que tus agentes heredan de una clase base Agent
from ..agents import BaseAgent 

# Configuración visual global al estilo de tu archivo
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def plot_episode_lengths(
        episode_lengths: List[int], 
        title: str = "Longitud del Episodio en el Tiempo",
        window_size: int = 50
    ):
    """
    Construye la gráfica f(t) = len(episodio_t).
    Es un buen indicador de aprendizaje en laberintos.

    :param episode_lengths: Lista con el número de pasos por episodio.
    :param title: Título de la gráfica.
    :param window_size: Tamaño de la ventana para suavizado de media móvil.
    """
    if window_size is not None and (window_size <= 0 or window_size > len(episode_lengths)):
        raise ValueError("El tamaño de la ventana debe ser un entero positivo menor o igual al número de episodios.")

    plt.figure(figsize=(14, 7))
    
    # Datos brutos
    plt.plot(episode_lengths, color='lightgray', alpha=0.6, label='Pasos reales')
    
    # Suavizado con media móvil
    if window_size > 1:
        smoothed = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(episode_lengths)), smoothed, color='red', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.xlabel('Episodios (t)', fontsize=14)
    plt.ylabel('$f(t) = len(episodio_t)$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_comparative_curves(
        data_dict: Dict[str, List[float]], 
        window_size: int = 50, 
        title: str = "Comparativa de Rendimiento", 
        ylabel: str = "Métrica"
    ):
    """
    Permite graficar los resultados de varios algoritmos en una sola imagen.

    :param data_dict: Diccionario donde la clave es el nombre del agente y el valor es la lista de datos.
    :param window_size: Tamaño de la ventana para suavizado de media móvil.
    :param title: Título general de la gráfica.
    :param ylabel: Etiqueta del eje Y.
    """
    plt.figure(figsize=(14, 7))
    
    # Usamos la paleta de Seaborn definida globalmente
    colores = sns.color_palette("muted", n_colors=len(data_dict))
    
    for i, (agent_name, data) in enumerate(data_dict.items()):
        color = colores[i]
        
        # Suavizado con media móvil
        if len(data) >= window_size and window_size > 1:
            smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(data)), smoothed, linewidth=2, label=agent_name, color=color)
        else:
            plt.plot(data, linewidth=2, label=agent_name, color=color)
            
    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_rewards(
        rewards: List[float], 
        window_size: int = 50, 
        title: str = "Recompensa Acumulada por Episodio"
    ):
    """
    Grafica la evolución de la recompensa obtenida en cada episodio para un agente individual.

    :param rewards: Lista de recompensas totales por episodio.
    :param window_size: Tamaño de la ventana para suavizado.
    :param title: Título de la gráfica.
    """
    if window_size is not None and (window_size <= 0 or window_size > len(rewards)):
         raise ValueError("El tamaño de la ventana debe ser un entero positivo menor o igual al número de episodios.")

    plt.figure(figsize=(14, 7))
    plt.plot(rewards, color='lightgray', alpha=0.6, label='Recompensa bruta')
    
    if window_size > 1:
        smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), smoothed, color='green', linewidth=2, label=f'Media móvil (n={window_size})')
    
    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel('Recompensa Total', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_success_rate(
        successes: List[int], 
        window_size: int = 50, 
        title: str = "Tasa de Éxito Promedio"
    ):
    """
    Grafica el porcentaje de éxito (llegar a la meta) en una ventana de episodios.

    :param successes: Lista de enteros (1 éxito, 0 fracaso).
    :param window_size: Ventana para calcular el porcentaje.
    :param title: Título de la gráfica.
    """
    plt.figure(figsize=(14, 7))
    
    if len(successes) >= window_size and window_size > 1:
        success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid') * 100
        plt.plot(range(window_size-1, len(successes)), success_rate, color='orange', linewidth=2, label=f'Tasa de éxito (n={window_size})')
    
    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel('Tasa de Éxito (%)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(-5, 105)
    plt.legend()
    plt.tight_layout()
    plt.show()

import time
import pygame

def evaluate_flappy_bird_agent(agent, env, num_episodes=3, render=True, fps=30):
    """
    Evalúa gráficamente (o lógicamente) un agente en el entorno Flappy Bird y printea la colisión y puntuación final.
    Maneja dependencias de pygame si el motor de renderizado falla por instalación.

    :param agent: El agente entrenado con un método get_action(state).
    :param env: Entorno de evaluación configurado (preferiblemente make_flappy_bird_env).
    :param num_episodes: Número de episodios de evaluación.
    :param render: Si es verdadero se asume que el env tiene render_mode="human" y se hace un sleep.
    :param fps: Puntos por segundo en los que se bloquea el render para que sea apreciable al ojo humano.
    """    
    # Manejo temporal de la excepción de Pygame si la biblioteca de C++ no soporta PNG en el venv actual
    try:
        env.reset()
    except Exception as e:
        print(f"Error inicializando gráfica del entorno: {str(e)}")
        return
    
    original_eps = agent.epsilon
    agent.epsilon = 0.0 # Desactivar exploración para evaluación
    try:
        for ep in range(num_episodes):
            state, info = env.reset()
            done = False
            score = 0
            frames = 0
            
            while not done:
                action = agent.get_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                frames += 1
                
                if render:
                    time.sleep(1.0 / fps)
                    
                if done:
                    score = info.get('score', 0)
                    print(f"[Visual-Ep {ep+1}] Agente colisionó en el timestep {frames}. Puntuación final: {score}")
                    break
    finally:
        agent.epsilon = original_eps # Restaurar el valor original de epsilon después de la evaluación

def plot_robust_learning_curves(
        data_dict: Dict[str, np.ndarray], 
        window_size: int = 50, 
        title: str = "Rendimiento Promedio (Múltiples Semillas)", 
        ylabel: str = "Recompensa Acumulada"
    ):
    """
    Plottea la curva de aprendizaje mostrando la media y la desviación típica 
    para evaluar la robustez del algoritmo frente a distintas semillas.

    :param data_dict: Diccionario. Clave: nombre del agente. 
                      Valor: np.ndarray de forma (num_semillas, n_steps).
    :param window_size: Ventana para el suavizado de la media móvil.
    :param title: Título de la gráfica.
    :param ylabel: Etiqueta del eje Y.
    """
    plt.figure(figsize=(14, 7))
    colores = sns.color_palette("muted", n_colors=len(data_dict))
    
    for i, (agent_name, data_matrix) in enumerate(data_dict.items()):
        color = colores[i]
        
        # data_matrix.shape debe ser (semillas, steps)
        num_seeds, num_episodes = data_matrix.shape
        mean_rewards = np.mean(data_matrix, axis=0)
        std_rewards = np.std(data_matrix, axis=0)
        
        # Suavizado si la ventana es mayor a 1
        if window_size > 1 and num_episodes >= window_size:
            mean_rewards = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
            std_rewards = np.convolve(std_rewards, np.ones(window_size)/window_size, mode='valid')
            x_axis = range(window_size - 1, num_episodes)
        else:
            x_axis = range(num_episodes)
            
        # Dibujar la línea de la media
        plt.plot(x_axis, mean_rewards, linewidth=2, label=f"{agent_name} (Media de {num_seeds} semillas)", color=color)
        
        # Dibujar el área sombreada para la desviación típica (varianza)
        plt.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, color=color, alpha=0.2)

    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dqn_loss(
        losses: List[float], 
        window_size: int = 100, 
        title: str = "Evolución de la Pérdida (Loss) de la Red Neuronal"
    ):
    """
    Grafica la evolución del error MSE durante el entrenamiento de DQN.
    Crucial para diagnosticar la convergencia en métodos de aproximación.

    :param losses: Lista con los valores de loss devueltos por la red neuronal por paso de optimización.
    :param window_size: Tamaño de la ventana de suavizado (suele ser grande porque el loss por step es ruidoso).
    :param title: Título de la gráfica.
    """
    if not losses:
        print("No hay datos de pérdida para graficar.")
        return

    plt.figure(figsize=(14, 7))
    
    # El loss original suele tener muchísima varianza, lo dibujamos muy tenue
    plt.plot(losses, color='lightgray', alpha=0.4, label='Pérdida (Loss) por step')
    
    if window_size > 1 and len(losses) >= window_size:
        smoothed_loss = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size - 1, len(losses)), smoothed_loss, color='purple', linewidth=2, label=f'Media móvil (n={window_size})')

    plt.xlabel('Pasos de Optimización (Steps)', fontsize=14)
    plt.ylabel('Loss (Huber / SmoothL1)', fontsize=14)
    plt.title(title, fontsize=16)
    # Usamos escala logarítmica si hay picos muy altos al inicio
    if max(losses) > 10 * np.mean(losses):
        plt.yscale('log')
        plt.ylabel('Loss (Log Scale)')
        
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_average_q_values(
        avg_q_values: List[float], 
        title: str = "Evolución de las Estimaciones de Valor Q (Flappy Bird)"
    ):
    """
    Grafica el valor Q promedio máximo predicho por la red en cada episodio.
    Sirve para detectar la sobreestimación clásica de DQN.

    :param avg_q_values: Lista con el promedio de los Q-values máximos por episodio.
    :param title: Título de la gráfica.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(avg_q_values, color='teal', linewidth=2, label='Q-Value Máximo Promedio')
    
    plt.xlabel('Episodios', fontsize=14)
    plt.ylabel('Valor Q Estimado', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
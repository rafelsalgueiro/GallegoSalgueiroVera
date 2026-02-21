from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..agents import Agent

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def plot_average_rewards(
        steps: int, 
        rewards: np.ndarray, 
        Agents: List[Agent], 
        window_size: int = 1,
        optimal_value: float = None
    ):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param Agents: Lista de instancias de algoritmos comparados.
    :param window_size: Tamaño de la ventana para el suavizado de media móvil (opcional).
    :param optimal_value: Valor óptimo para graficar una línea de referencia (opcional).
    """
    if window_size is not None and (window_size <= 0 or window_size > steps):
        raise ValueError("El tamaño de la ventana debe ser un entero positivo menor o igual al número de pasos.")

    plt.figure(figsize=(14, 7))

    if optimal_value:
        plt.axhline(optimal_value, color='k', linestyle='--', label='Valor Óptimo')

    for idx, agent in enumerate(Agents):
        label = agent.label
        # Suavizado con media móvil
        smoothed_rewards = np.convolve(rewards[idx], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, steps), smoothed_rewards, label=label, linewidth=2)
    plt.xlabel('Pasos de Tiempo', fontsize=14)
    ylbl = 'Recompensa Promedio' + (f' (suavizado de tamaño {window_size})' if window_size else '')
    plt.ylabel(ylbl, fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_optimal_selections(
        steps: int, 
        optimal_selections: np.ndarray, 
        Agents: List[Agent],
        window_size: int = 1
    ):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param Agents: Lista de instancias de algoritmos comparados.
    :param window_size: Tamaño de la ventana para el suavizado de media móvil (opcional).
    """
    plt.figure(figsize=(14, 7))
    for i, algo in enumerate(Agents):
        # Suavizado con media móvil
        smoothed_selections = np.convolve(optimal_selections[i], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, steps), smoothed_selections * 100, label=algo.name)
    
    plt.xlabel('Pasos de Tiempo')
    ylbl = '% Brazo Óptimo' + (f' (suavizado de tamaño {window_size})' if window_size else '')
    plt.ylabel(ylbl)
    plt.title('Rendimiento: Selección del Brazo Óptimo')
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_arm_statistics(arm_stats: List[dict], Agents: List[Agent], *args):
    """
    Genera gráficas separadas de Selección de Arms:
                                            Ganancias vs Pérdidas para cada algoritmo.

    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo.
    :param Agents: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres

    """
    n_algos = len(Agents)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 5), sharey=True)
    if n_algos == 1: axes = [axes]

    for i, (algo, stats) in enumerate(zip(Agents, arm_stats)):
        arms = np.arange(len(stats['ganancias']))
        width = 0.35
        
        axes[i].bar(arms - width/2, stats['ganancias'], width, label='Ganancias', color='g')
        axes[i].bar(arms + width/2, stats['pérdidas'], width, label='Pérdidas', color='r')
        
        axes[i].set_title(f'Stats: {algo.name}')
        axes[i].set_xlabel('Brazo')
        axes[i].legend()

    axes[0].set_ylabel('Frecuencia / Valor')
    plt.tight_layout()
    plt.show()

def plot_regret(
        steps: int, 
        regret_accumulated: np.ndarray, 
        Agents: List[Agent],
        cte: float = None
    ):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param Agents: Lista de instancias de algoritmos comparados.
    :param cte: Constante teórica para la cota de regret (opcional).
    """
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, steps + 1)
    
    for i, agent in enumerate(Agents):
        plt.plot(time_steps, regret_accumulated[i], label=agent.label)
    
    # Si se pasa una constante en args, graficar la cota teórica logarítmica
    if cte is not None:
        theoretical_bound = cte * np.log(time_steps)
        plt.plot(time_steps, theoretical_bound, 'k--', label=f'Cota Teórica ({cte:.2f}*ln(T))')

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('Rechazo (Regret) Acumulado')
    plt.title('Evolución del Rechazo Acumulado')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
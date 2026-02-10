from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..agents import Agent, EpsilonGreedyAgent


def get_Agent_label(algo: Agent) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Agent
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedyAgent):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Agent o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, Agents: List[Agent]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param Agents: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(Agents):
        label = get_Agent_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, Agents: List[Agent]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param Agents: Lista de instancias de algoritmos comparados.
    """
    plt.figure(figsize=(10, 6))
    for i, algo in enumerate(Agents):
        plt.plot(range(steps), optimal_selections[i] * 100, label=algo.name)
    
    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('% Selección Brazo Óptimo')
    plt.title('Rendimiento: Selección del Brazo Óptimo')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_arm_statistics(arm_stats: List[dict], Agents: List, *args):
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

def plot_regret(steps: int, regret_accumulated: np.ndarray, Agents: List, *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param Agents: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T)
    """
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(1, steps + 1)
    
    for i, algo in enumerate(Agents):
        plt.plot(time_steps, regret_accumulated[i], label=algo.name)
    
    # Si se pasa una constante en args, graficar la cota teórica logarítmica
    if args:
        cte = args[0]
        theoretical_bound = cte * np.log(time_steps)
        plt.plot(time_steps, theoretical_bound, 'k--', label=f'Cota Teórica ({cte}*ln(T))')

    plt.xlabel('Pasos de Tiempo')
    plt.ylabel('Rechazo (Regret) Acumulado')
    plt.title('Evolución del Rechazo Acumulado')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
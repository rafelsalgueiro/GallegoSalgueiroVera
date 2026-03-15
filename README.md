# GallegoSalgueiroVera

## Información
- **Alumnos:** Gallego, Juan Diego; Salgueiro, Rafel; Vera, Oscar
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2025/2026
- **Grupo:** GallegoSalgueiroVera

## Descripción
En este trabajo se estudia el aprendizaje por refuerzo desde dos perspectivas complementarias.

En primer lugar, se aborda el problema del bandido de k-brazos, un escenario estático que permite analizar en profun-
didad el dilema exploración-explotación. Se comparan tres familias de algoritmos: ϵ-greedy (con y sin decaimiento), UCB y
Softmax; sobre bandidos con distribuciones de Bernoulli, binomial y normal. Para la evaluación se utilizan las métricas de
arrepentimiento acumulado y tasa de selección del brazo óptimo. Los resultados muestran que UCB destaca por su rapidez
de convergencia cuando se dispone de información sobre la varianza del problema, mientras que ϵ-greedy con decaimiento
ofrece un comportamiento más robusto y generalista.

En segundo lugar, se extiende el análisis a entornos con secuencialidad y estado, formalizados como Procesos de Decisión de
Markov. Se evalúan métodos tabulares clásicos (Monte Carlo y diferencias temporales) en un entorno discreto de cuadrícula,
y métodos de aproximación de función (SARSA semi-grediente y Deep Q-Learning) en el entorno continuo Flappy Bird. Se
estudia el impacto de técnicas como el replay buffer, la red objetivo y el reward shaping sobre la estabilidad y la calidad del
aprendizaje.

## Estructura
El repositorio está organizado de forma modular para separar los distintos ámbitos de estudio:

* `/k_brazos`: Directorio principal para la resolución de los problemas de bandidos.
    * `main.ipynb`: Notebook base que explica y redirige a los estudios específicos.
    * `estudio_bernoulli.ipynb`, `estudio_binomial.ipynb` y `estudio_normal.ipynb`: Notebooks de implementación y análisis de los algoritmos sobre cada tipo de distribución.
    * `/src`: Código fuente de los agentes, los brazos y las gráficas.
* `/Entornos_Complejos`: Directorio principal para la resolución de los MDPs.
    * `main.ipynb`: Notebook base que explica y redirige a los notebooks de implementación.
    * `estudio_tabulares.ipynb`: Entrenamientos y comparativas de los algoritmos tabulares sobre el entorno *SimpleGrid*.
    * `deep_q_learning.ipynb`, `flappy_bird.ipynb` y `sarsa_semi_gradient.ipynb`: Entrenamientos y evaluaciones de los agentes aproximados sobre el entorno *Flappy Bird*.
    * `/src`: Código fuente de los agentes, del entorno y de las utilidades de graficado.
* `/docs`: LaTeX para la elaboración de la memoria final.
* `informe.pdf`: Memoria final del trabajo de investigación.

## Instalación y Uso
Para ejecutar los *notebooks* y reproducir los experimentos, se recomienda utilizar Google Collab.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/rafelsalgueiro/GallegoSalgueiroVera.git
    cd GallegoSalgueiroVera
    ```

2. **Ejecutar notebooks:**
    - Navegar a los directorios correspondientes (`/k_brazos` o `/Entornos_Complejos`), abrir los notebooks con Google Collab y ejecutarlos (se instalan las librerias necesarias en cada notebook).

## Tecnologías Utilizadas

* **Lenguaje:** Python 3
* **Entornos de Simulación:** Gymnasium (para implementaciones base, *SimpleGrid* y wrappers de *Flappy Bird*)
* **Deep Learning:** PyTorch (para la construcción de las redes neuronales en DQN y SARSA Semi-Gradiente)
* **Computación y Visualización:** NumPy, Matplotlib, Seaborn, Pandas
* **Experimentación:** Jupyter Notebook
* Uso de agentes LLM para la asistencia en la elaboración de texto, toma de ideas y asistencia en la escritura de código.
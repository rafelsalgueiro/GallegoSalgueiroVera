import numpy as np
from .agent import Agent

class EpsilonDecayAgent(Agent):
    """
    Agente que implementa una política epsilon-greedy con decaimiento inversaente proporcional.
    f(t,e,lda) = max(e_end, e_start / (1 + lda * t))
    """

    def __init__(
            self, 
            k_arms: int,
            e_start: float = 1.0, 
            e_end: float = 0.01, 
            lda: float = 0.01
    ):
        """
        Inicializa los parámetros del agente

        :param k_arms: Número de brazos disponibles (acciones).
        :param e_start: Valor inicial de epsilon (probabilidad de exploración).
        :param e_end: Valor mínimo de epsilon (probabilidad de exploración).
        :param lda: Factor de decaimiento inverso para epsilon.
        """
        super().__init__(k_arms, name=f"EpsilonDecay (e_s={e_start}, e_e={e_end}, l={lda})")
        self.epsilon = e_start
        self.lda = lda
        self.epsilon_end = e_end

    def get_action(self) -> int:
        """
        Selecciona una acción siguiendo la política epsilon-greedy con decaimiento.
        """
        # Exploración: elegir brazo al azar
        eps = max(self.epsilon_end, self.epsilon / (1.0 + self.lda * self.n_steps))
        if np.random.random() < eps:
            return np.random.randint(self.k_arms)
        
        # Explotación: elegir el mejor brazo (valor Q más alto)
        # Se usa argmax con desempate aleatorio para evitar sesgos
        return np.random.choice(np.where(self.values == np.max(self.values))[0])
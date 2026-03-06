import gymnasium as gym
import flappy_bird_gymnasium

class FlappyBirdRewardWrapper(gym.Wrapper):
    """
    Custom Wrapper para modificar las recompensas del entorno FlappyBird-v0.
    Incentiva al agente a mantenerse en el centro del hueco de los próximos tubos.
    """
    def __init__(self, env, alpha=0.5, ignore_env_penalties=False):
        super().__init__(env)
        # Verificar que no se esté usando lidar
        if hasattr(env.unwrapped, 'use_lidar') and env.unwrapped.use_lidar:
            raise ValueError(
                "FlappyBirdRewardWrapper no es compatible con observaciones lidar. "
                "Los índices de observación usados para reward shaping solo son válidos con use_lidar=False."
            )
        self.alpha = alpha
        # Renombrado para mayor claridad semántica
        self.ignore_env_penalties = ignore_env_penalties

    def set_alpha(self, new_alpha):
        """Permite actualizar el valor de alpha durante el entrenamiento."""
        self.alpha = max(0.0, min(1.0, new_alpha))
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Gestionar las penalizaciones nativas del entorno (-0.5 por techo, -1.0 por morir)
        if reward < 0 and self.ignore_env_penalties:
            reward = 0.0
            
        # Aplicar Reward Shaping solo si el agente sigue vivo y no ha sido penalizado.
        # Evitamos dar un premio cuando choca
        if not terminated and reward >= 0:
            next_top_pipe_y = obs[4]
            next_bottom_pipe_y = obs[5]
            player_y = obs[9]
            
            # Calcular el centro del hueco
            gap_center = (next_top_pipe_y + next_bottom_pipe_y) / 2.0
            gap_size = abs(next_bottom_pipe_y - next_top_pipe_y)
            distance_to_center = abs(player_y - gap_center)
            
            # Normalización dinámica basada en el tamaño del hueco
            normalized_dist = distance_to_center / (gap_size / 2.0)
            proximity = max(0.0, 1.0 - normalized_dist)
            
            shaping_bonus = self.alpha * proximity
            reward += shaping_bonus
            
        return obs, reward, terminated, truncated, info

def make_flappy_bird_env(alpha=0.5, use_lidar=False, render_mode=None, ignore_env_penalties=False):
    """
    Crea el entorno FlappyBird-v0 y le aplica el wrapper de recompensas si alpha > 0.
    """
    env_base = gym.make("FlappyBird-v0", use_lidar=use_lidar, render_mode=render_mode)
    if alpha > 0:
        return FlappyBirdRewardWrapper(env_base, alpha=alpha, ignore_env_penalties=ignore_env_penalties)
    return env_base
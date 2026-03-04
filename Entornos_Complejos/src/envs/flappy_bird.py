import gymnasium as gym
import flappy_bird_gymnasium

class FlappyBirdRewardWrapper(gym.Wrapper):
    """
    Custom Wrapper para modificar las recompensas del entorno FlappyBird-v0.
    Incentiva al agente a mantenerse en el centro del hueco de los próximos tubos.
    """
    def __init__(self, env, alpha=0.5, count_negative_rewards=False):
        super().__init__(env)
        # Verificar que no se esté usando lidar, ya que los índices de observación
        # son específicos para el modo sin lidar (12-dim).
        if hasattr(env.unwrapped, 'use_lidar') and env.unwrapped.use_lidar:
            raise ValueError(
                "FlappyBirdRewardWrapper no es compatible con observaciones lidar. "
                "Los índices de observación usados para reward shaping solo son válidos con use_lidar=False."
            )
        self.alpha = alpha
        self.count_negative_rewards = count_negative_rewards

    def set_alpha(self, new_alpha):
        """
        Permite actualizar el valor de alpha durante el entrenamiento.
        """
        self.alpha = max(0.0, min(1.0, new_alpha))
        
    def step(self, action):
        # Ejecutar la acción en el entorno original
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Aplicar Reward Shaping solo si el agente sigue vivo
        if not terminated:
            # Gymnasium nativo penaliza con -0.5 si el pájaro roza el techo de la pantalla.
            # Clippeamos ANTES de aplicar el shaping para no enmascarar la penalización.
            if reward < 0 and not self.count_negative_rewards:
                reward = 0.0
            
            next_top_pipe_y = obs[4]
            next_bottom_pipe_y = obs[5]
            player_y = obs[9]
            
            gap_center = (next_top_pipe_y + next_bottom_pipe_y) / 2.0
            distance_to_center = abs(player_y - gap_center)
            
            # Normalizamos la cercanía usando max para evitar negativos
            proximity = max(0.0, 1.0 - (distance_to_center * 2.0))
            shaping_bonus = self.alpha * proximity
            
            reward += shaping_bonus
            
        return obs, reward, terminated, truncated, info

def make_flappy_bird_env(alpha=0.5, use_lidar=False, render_mode=None, count_negative_rewards=False):
    """
    Crea el entorno FlappyBird-v0 y le aplica el wrapper de recompensas si alpha > 0.
    """
    env_base = gym.make("FlappyBird-v0", use_lidar=use_lidar, render_mode=render_mode)
    if alpha > 0:
        return FlappyBirdRewardWrapper(env_base, alpha=alpha, count_negative_rewards=count_negative_rewards)
    return env_base

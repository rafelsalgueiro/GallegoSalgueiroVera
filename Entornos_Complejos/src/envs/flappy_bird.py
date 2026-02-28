import gymnasium as gym
import flappy_bird_gymnasium

class FlappyBirdRewardWrapper(gym.Wrapper):
    """
    Custom Wrapper para modificar las recompensas del entorno FlappyBird-v0.
    Incentiva al agente a mantenerse en el centro del hueco de los próximos tubos.
    """
    def __init__(self, env, alpha=0.5):
        super().__init__(env)
        self.alpha = alpha
        
    def step(self, action):
        # Ejecutar la acción en el entorno original
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Aplicar Reward Shaping solo si el agente sigue vivo
        if not terminated:
            next_top_pipe_y = obs[4]
            next_bottom_pipe_y = obs[5]
            player_y = obs[9]
            
            gap_center = (next_top_pipe_y + next_bottom_pipe_y) / 2.0
            distance_to_center = abs(player_y - gap_center)
            
            # Normalizamos la cercanía usando max para evitar negativos
            proximity = max(0.0, 1.0 - (distance_to_center * 2.0))
            shaping_bonus = self.alpha * proximity
            
            reward += shaping_bonus
            
            # Gymnasium nativo penaliza con -0.5 si el pájaro roza el techo de la pantalla.
            if reward < 0:
                reward = 0.0
            
        return obs, reward, terminated, truncated, info

def make_flappy_bird_env(alpha=0.5, use_lidar=False, render_mode=None):
    """
    Crea el entorno FlappyBird-v0 y le aplica el wrapper de recompensas si alpha > 0.
    """
    env_base = gym.make("FlappyBird-v0", use_lidar=use_lidar, render_mode=render_mode)
    if alpha > 0:
        return FlappyBirdRewardWrapper(env_base, alpha=alpha)
    return env_base

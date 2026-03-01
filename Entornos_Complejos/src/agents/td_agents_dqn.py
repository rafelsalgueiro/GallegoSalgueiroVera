import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy

from .td_agents import TDAgent


class DQNNet(nn.Module):
    """
    Neural network for approximating Q(s, a) in a Deep Q-Network.
    Uses a standard MLP with ReLU activations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNReplayBuffer:
    """
    Experience replay buffer for DQN (does not store next_action,
    since DQN is off-policy and uses max over all actions).
    """
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer, overwriting oldest if full."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Return a random batch of transitions as PyTorch tensors."""
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.LongTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.BoolTensor(self.dones[idxs]),
        )

    def __len__(self):
        return self.size


class DQNAgent(TDAgent):
    """
    Deep Q-Network (DQN) agent with optional target network.

    Key features:
      - Off-policy learning: uses max_a Q(s', a) as the bootstrap target.
      - Experience Replay: decorrelates training samples.
      - Target Network (optional): stabilises training by keeping a slowly-
        updated copy of the Q-network for computing TD targets.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to interact with.
    alpha : float
        Learning rate for the Adam optimiser.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate for epsilon-greedy policy.
    hidden_dim : int
        Number of units in each hidden layer of the Q-network.
    use_target_network : bool
        Whether to use a separate target network.
    target_update_freq : int
        How often (in optimisation steps) to hard-copy weights to target net.
    grad_clip : float or None
        If not None, clip gradient norms to this value.
    """

    def __init__(
        self,
        env,
        alpha=1e-3,
        gamma=0.99,
        epsilon=0.1,
        hidden_dim=128,
        use_target_network=True,
        target_update_freq=1000,
        grad_clip=1.0,
    ):
        super().__init__(env, alpha, gamma, epsilon)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Online (policy) network
        self.q_net = DQNNet(self.state_dim, self.action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.criterion = nn.SmoothL1Loss()  # Huber loss — more robust than MSE

        # Target network
        self.use_target_network = use_target_network
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self._optim_steps = 0

        if self.use_target_network:
            self.target_net = copy.deepcopy(self.q_net)
            self.target_net.eval()  # target net never trains directly
        else:
            self.target_net = self.q_net  # alias – same network

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def get_q_value(self, state, action=None):
        """Return Q-values from the *online* network (no gradient)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).numpy()[0]
        if action is not None:
            return q_values[action]
        return q_values

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        q_values = self.get_q_value(state)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return random.choice(best_actions)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """Single-transition DQN update (mainly for reference / debugging)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

        current_q = self.q_net(state_t)[0, action]

        with torch.no_grad():
            if done:
                target_q = torch.tensor(reward, dtype=torch.float32)
            else:
                next_q_max = self.target_net(next_state_t).max(1)[0]
                target_q = torch.tensor(reward, dtype=torch.float32) + self.gamma * next_q_max

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._optim_steps += 1
        self._maybe_update_target()

        return loss.item()

    def update_batch(self, states, actions, rewards, next_states, dones):
        """
        Update the Q-network using a mini-batch sampled from a replay buffer.
        This is the standard DQN training step.
        """
        # Q(s, a) from online network
        current_q_values = self.q_net(states)
        current_q = current_q_values.gather(1, actions).squeeze(1)

        # max_a' Q_target(s', a')  — no gradient
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_max = next_q_values.max(1)[0]

        # TD target: r + γ·max Q(s',a')  (0 when done)
        target_q = rewards.squeeze(1) + self.gamma * next_q_max * (~dones.squeeze(1))

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._optim_steps += 1
        self._maybe_update_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network management
    # ------------------------------------------------------------------
    def _maybe_update_target(self):
        """Hard-copy online weights → target network every N steps."""
        if self.use_target_network and self._optim_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def sync_target_network(self):
        """Manually synchronise the target network with the online network."""
        if self.use_target_network:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_weights(self, path):
        """Save the online Q-network weights."""
        torch.save(self.q_net.state_dict(), path)
        print(f"Weights saved to {path}")

    def load_weights(self, path):
        """Load weights into the online network (and target network if used)."""
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, weights_only=True))
            if self.use_target_network:
                self.target_net.load_state_dict(self.q_net.state_dict())
            self.q_net.eval()
            print(f"Weights loaded from {path}")
            return True
        else:
            print(f"Error: path {path} not found.")
            return False

from .agent import Agent
from .greedy import EpsilonGreedyAgent
from .UCB1 import UCBAgent
from .softmax import SoftmaxAgent
from .edecay import EpsilonDecayAgent

__all__ = ['Agent', 'EpsilonGreedyAgent', 'UCBAgent', 'SoftmaxAgent', 'EpsilonDecayAgent']
from .agent import Agent
from .edecay import EpsilonDecayAgent
from .greedy import EpsilonGreedyAgent
from .UCB1 import UCBAgent
from .softmax import SoftmaxAgent

__all__ = ['Agent',  'EpsilonDecayAgent', 'EpsilonGreedyAgent', 'UCBAgent', 'SoftmaxAgent']
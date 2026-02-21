from .agent import Agent
from .edecay import EpsilonDecayAgent
from .greedy import EpsilonGreedyAgent
from .UCB1 import UCBAgent
from .softmax import SoftmaxAgent
from .edecay import EpsilonDecayAgent

__all__ = ['Agent',  'EpsilonDecayAgent', 'EpsilonGreedyAgent', 'UCBAgent', 'SoftmaxAgent']

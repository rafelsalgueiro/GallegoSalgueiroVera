from abc import ABC, abstractmethod


class Arm(ABC):

    @classmethod
    def generate_arms(cls, k: int):
        """
        Generates a list of arms with random parameters.

        :param k: Number of arms to generate.
        :return: List of arms.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def pull(self):
        """
        Generates a reward based on the arm's distribution.

        This method must be implemented by derived classes.

        :raises NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def get_expected_value(self) -> float:
        """
        Calculates and returns the expected value of the arm's reward.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
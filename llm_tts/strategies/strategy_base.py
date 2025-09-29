from abc import ABC, abstractmethod
from typing import Dict


class StrategyBase(ABC):
    """Abstract base class for scoring step candidates in real-time"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        pass

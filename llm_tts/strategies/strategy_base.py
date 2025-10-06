from abc import ABC, abstractmethod
from typing import Dict, List


class StrategyBase(ABC):
    """Abstract base class for scoring step candidates in real-time"""

    @abstractmethod
    def generate_trajectory(self, input_chat: List[Dict[str, str]]) -> Dict[str, any]:
        pass

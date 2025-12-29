"""
Base class for step boundary detectors.
"""

from abc import ABC, abstractmethod
from typing import List


class StepBoundaryDetectorBase(ABC):
    """Abstract base class for step boundary detectors."""

    @abstractmethod
    def detect_steps(self, text: str, **kwargs) -> List[str]:
        """
        Detect and extract reasoning steps from text.

        Args:
            text: The text to analyze for step boundaries

        Returns:
            List of step strings
        """
        pass

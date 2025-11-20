from abc import ABC, abstractmethod
from typing import List

import numpy as np

# Classify in which step should be scaled based on signal (perplexity, entropy, etc.)

# 1. MUR: Momentum Uncertainty guided discriminator
# 2. Random: Random discriminator
# 3. Average: Average-based discriminator
# 4. Always: Always generate candidates discriminator
# 5. Threshold: Threshold-based discriminator


class DiscriminatorBase(ABC):
    """Abstract base class for discriminators"""

    @abstractmethod
    def should_scale(
        self, cur_signal: float, signal_history: List[float], step_num: int, **kwargs
    ) -> bool:
        pass


class MURDiscriminator(DiscriminatorBase):
    """Momentum Uncertainty guided discriminator with momentum uncertainty tracking"""

    def __init__(self, momentum_rate: float = 0.9, scaling_rate: float = 0.9):
        self.momentum_rate = momentum_rate
        self.scaling_rate = scaling_rate
        self.momentum_uncertainty = 0.0
        self.current_signal = 0.0
        self.step_num = 0

    def should_scale(self, cur_signal: float) -> bool:
        """Determine if candidates should be generated based on MUR criteria"""
        return self.step_num > 0 and np.exp(
            self.momentum_uncertainty
        ) / self.scaling_rate < np.exp(cur_signal)

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        self.current_signal = cur_signal
        self.step_num += 1
        self.momentum_uncertainty = (
            self.momentum_uncertainty * self.momentum_rate
            + (1 - self.momentum_rate) * self.current_signal
        )


class RandomDiscriminator(DiscriminatorBase):
    """Random discriminator for testing"""

    def __init__(self):
        self.step_num = 0

    def should_scale(self, cur_signal: float) -> bool:
        return self.step_num % 2 != 0

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        self.current_signal = cur_signal
        self.step_num += 1


class AverageDiscriminator(DiscriminatorBase):
    """Average-based discriminator"""

    def __init__(self):
        self.signal_history = []
        self.step_num = 0

    def should_scale(self, cur_signal: float) -> bool:
        if self.step_num == 0:
            return False
        return np.mean(self.signal_history) < cur_signal

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        self.step_num += 1
        self.signal_history.append(cur_signal)


class AlwaysDiscriminator(DiscriminatorBase):
    """Always generate candidates discriminator"""

    def __init__(self):
        self.step_num = 0
        self.current_signal = 0.0

    def should_scale(self, cur_signal: float) -> bool:
        return True

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        self.step_num += 1
        self.current_signal = cur_signal


class ThresholdDiscriminator(DiscriminatorBase):
    """Threshold-based discriminator"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.step_num = 0

    def should_scale(self, cur_signal: float) -> bool:
        return self.current_signal < self.threshold

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        self.current_signal = cur_signal
        self.step_num += 1


class ScaleDiscriminator:

    def __init__(self, criterion: str, **kwargs):
        self.criterion = criterion
        self.discriminator = None
        if self.criterion == "momentum":
            self.discriminator = MURDiscriminator(**kwargs)
        elif self.criterion == "random":
            self.discriminator = RandomDiscriminator(**kwargs)
        elif self.criterion == "average":
            self.discriminator = AverageDiscriminator(**kwargs)
        elif self.criterion == "always":
            self.discriminator = AlwaysDiscriminator(**kwargs)
        elif self.criterion == "threshold":
            self.discriminator = ThresholdDiscriminator(**kwargs)
        else:
            raise ValueError(
                f"Invalid criterion: {self.criterion}. Valid criteria are: 'momentum', 'random', 'average', 'always', 'threshold'"
            )

    def should_scale(self, cur_signal: float) -> bool:
        return self.discriminator.should_scale(cur_signal)

    def update(self, cur_signal: float) -> float:
        """Update state of the discriminator"""
        return self.discriminator.update(cur_signal)

    def reset(self):
        self.discriminator.__init__()

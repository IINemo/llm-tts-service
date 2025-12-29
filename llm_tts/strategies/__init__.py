from .adaptive_scaling_best_of_n import AdaptiveScalingBestOfN
from .deepconf import StrategyDeepConf
from .phi import PhiDecoding
from .strategy_base import StrategyBase
from .strategy_baseline import StrategyBaseline
from .strategy_beam_search import StrategyBeamSearch
from .strategy_chain_of_thought import StrategyChainOfThought
from .strategy_online_best_of_n import StrategyOnlineBestOfN
from .strategy_self_consistency import StrategySelfConsistency
from .strategy_uncertainty_cot import StrategyUncertaintyCoT
from .tree_of_thoughts import StrategyTreeOfThoughts

# vLLM-only strategies (optional - requires vllm package)
try:
    from .strategy_offline_best_of_n import StrategyOfflineBestOfN

    VLLM_STRATEGIES_AVAILABLE = True
except ImportError:
    StrategyOfflineBestOfN = None
    VLLM_STRATEGIES_AVAILABLE = False

__all__ = [
    "StrategyBaseline",
    "StrategyOnlineBestOfN",
    "StrategyOfflineBestOfN",
    "StrategyBeamSearch",
    "StrategyChainOfThought",
    "StrategySelfConsistency",
    "StrategyDeepConf",
    "AdaptiveScalingBestOfN",
    "StrategyUncertaintyCoT",
    "PhiDecoding",
    "StrategyTreeOfThoughts",
    "VLLM_STRATEGIES_AVAILABLE",
]

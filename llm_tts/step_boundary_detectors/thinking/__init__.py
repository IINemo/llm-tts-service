"""
Step boundary detectors for native thinking mode (<think> tags).

These detectors are designed for models with native thinking mode like
Qwen3, o1, DeepSeek-R1 that output reasoning in <think> tags.

Submodules:
- huggingface: HuggingFace-specific utilities (BatchStepStoppingCriteria)
- vllm: vLLM-specific utilities (stop token generation)
"""

from .hybrid import ThinkingAdaptiveDetector, ThinkingHybridDetector
from .llm import ThinkingLLMDetector, ThinkingLLMDetectorVLLM
from .marker import ThinkingMarkerDetector
from .sentence import ThinkingSentenceDetector

# Backend-specific submodules
from . import huggingface
from . import vllm

__all__ = [
    "ThinkingSentenceDetector",
    "ThinkingMarkerDetector",
    "ThinkingLLMDetector",
    "ThinkingLLMDetectorVLLM",
    "ThinkingHybridDetector",
    "ThinkingAdaptiveDetector",
    # Submodules
    "huggingface",
    "vllm",
]

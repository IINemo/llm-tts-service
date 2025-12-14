"""
Step boundary detectors for native thinking mode (<think> tags).

These detectors are designed for models with native thinking mode like
Qwen3, o1, DeepSeek-R1 that output reasoning in <think> tags.

Submodules:
- offline: Offline detectors for post-hoc analysis (sentence, llm, hybrid)
- huggingface: HuggingFace-specific utilities (BatchStepStoppingCriteria)
- vllm: vLLM-specific utilities (stop token generation)
"""

# Backend-specific submodules
from . import huggingface, offline, vllm
from .marker import ThinkingMarkerDetector

# Re-export offline detectors for backwards compatibility
from .offline import (
    ThinkingAdaptiveDetector,
    ThinkingHybridDetector,
    ThinkingLLMDetector,
    ThinkingLLMDetectorVLLM,
    ThinkingSentenceDetector,
)

__all__ = [
    # Online/offline marker detector
    "ThinkingMarkerDetector",
    # Offline detectors (also available via .offline submodule)
    "ThinkingSentenceDetector",
    "ThinkingLLMDetector",
    "ThinkingLLMDetectorVLLM",
    "ThinkingHybridDetector",
    "ThinkingAdaptiveDetector",
    # Submodules
    "offline",
    "huggingface",
    "vllm",
]

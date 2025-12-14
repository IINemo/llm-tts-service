"""
Step candidate generators for test-time scaling strategies.

This module provides generators that produce candidate next steps
for various reasoning strategies (beam search, best-of-n, etc.).

Structure:
- api/: API-based generators (OpenAI-compatible)
- huggingface/: HuggingFace transformers generators
- vllm/: vLLM generators for fast batched inference

Each backend has:
- structured.py: Generators for structured step patterns (- Step 1, etc.)
- thinking.py: Generators for thinking mode (<think> tags)
"""

# Backend submodules
from llm_tts.generators import api, huggingface, vllm

# Re-export commonly used classes for backward compatibility
from llm_tts.generators.api import StepCandidateGeneratorThroughAPI
from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    convert_trajectory_to_string,
)
from llm_tts.generators.huggingface import (
    BatchStepStoppingCriteria,
    StepCandidateGeneratorThroughHuggingface,
    ThinkingStepStoppingCriteria,
)

# vLLM generators (optional - requires vllm package)
try:
    from llm_tts.generators.vllm import StepCandidateGeneratorThroughVLLM
except ImportError:
    StepCandidateGeneratorThroughVLLM = None

__all__ = [
    # Base classes
    "StepCandidate",
    "StepCandidateGeneratorBase",
    "convert_trajectory_to_string",
    # Submodules
    "api",
    "huggingface",
    "vllm",
    # Backward-compatible exports
    "StepCandidateGeneratorThroughAPI",
    "StepCandidateGeneratorThroughHuggingface",
    "StepCandidateGeneratorThroughVLLM",
    "BatchStepStoppingCriteria",
    "ThinkingStepStoppingCriteria",
]

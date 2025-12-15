"""
Step candidate generators for test-time scaling strategies.

This module provides generators that produce candidate next steps
for various reasoning strategies (beam search, best-of-n, etc.).

Structure:
- api.py: API-based generators (OpenAI-compatible)
- huggingface.py: HuggingFace transformers generators
- vllm.py: Unified vLLM generator with thinking_mode parameter
"""

# Backend submodules (vllm is optional)
from llm_tts.generators import api, huggingface

# Re-export commonly used classes
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

# vLLM generator (optional - requires vllm package)
try:
    from llm_tts.generators.vllm import VLLMStepGenerator

    VLLM_AVAILABLE = True
except ImportError:
    VLLMStepGenerator = None
    VLLM_AVAILABLE = False

__all__ = [
    # Base classes
    "StepCandidate",
    "StepCandidateGeneratorBase",
    "convert_trajectory_to_string",
    # Submodules
    "api",
    "huggingface",
    "VLLM_AVAILABLE",
    # Exports
    "StepCandidateGeneratorThroughAPI",
    "StepCandidateGeneratorThroughHuggingface",
    "VLLMStepGenerator",
    "BatchStepStoppingCriteria",
    "ThinkingStepStoppingCriteria",
]

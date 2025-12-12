"""
Step candidate generators for test-time scaling strategies.

This module provides generators that produce candidate next steps
for various reasoning strategies (beam search, best-of-n, etc.).

Available generators:
- StepCandidateGeneratorThroughHuggingface: Local HuggingFace models
- StepCandidateGeneratorThroughAPI: OpenAI-compatible APIs
- StepCandidateGeneratorThroughVLLM: vLLM for fast batched inference

Also includes:
- StructuredStepDetector: Detects step and answer boundaries in generated text
- StepCandidate: Data class representing a candidate step
"""

from llm_tts.generators.api import StepCandidateGeneratorThroughAPI
from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    covert_trajectory_to_string,
)
from llm_tts.generators.huggingface import (
    BatchStepStoppingCriteria,
    StepCandidateGeneratorThroughHuggingface,
    ThinkingStepStoppingCriteria,
)

# vLLM generators (optional)
try:
    from llm_tts.generators.vllm import StepCandidateGeneratorThroughVLLM
except ImportError:
    pass

__all__ = [
    "StepCandidate",
    "StepCandidateGeneratorBase",
    "StepCandidateGeneratorThroughAPI",
    "StepCandidateGeneratorThroughHuggingface",
    "StepCandidateGeneratorThroughVLLM",
    "BatchStepStoppingCriteria",
    "ThinkingStepStoppingCriteria",
    "covert_trajectory_to_string",
]

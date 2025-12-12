"""
Step boundary detectors for reasoning traces.

This module provides different strategies for detecting step boundaries
in LLM reasoning outputs, supporting both structured responses and
native thinking mode.

Detectors:
---------
Structured Response Detectors (non-thinking mode):
    StructuredStepDetector: For explicit step markers (- Step 1:, - Step 2:, etc.)

Thinking Mode Detectors (native <think> tags):
    ThinkingSentenceDetector: Split by sentences/paragraphs
    ThinkingMarkerDetector: Split by linguistic markers (so, therefore, let me...)
    ThinkingLLMDetector: Use secondary LLM to parse steps
    ThinkingHybridDetector: Combine markers with fallback to sentences
    ThinkingAdaptiveDetector: Auto-select strategy based on content

Usage:
------
# For structured responses (non-thinking mode)
from llm_tts.step_boundary_detectors import StructuredStepDetector
detector = StructuredStepDetector(
    step_patterns=["- Step"],
    answer_patterns=["<Answer>:"],
    max_tokens_per_step=512,
)
steps = detector.detect_steps(response_text)

# For native thinking mode (recommended: marker_semantic_v2 config)
from llm_tts.step_boundary_detectors import ThinkingMarkerDetector
detector = ThinkingMarkerDetector(
    use_structure=False,
    use_reasoning=True,
    min_step_chars=100,
    max_step_chars=600,
)
steps = detector.detect_steps(thinking_content)
"""

from .base import StepBoundaryDetectorBase
from .non_thinking import StructuredStepDetector
from .thinking_hybrid import ThinkingAdaptiveDetector, ThinkingHybridDetector
from .thinking_llm import ThinkingLLMDetector, ThinkingLLMDetectorVLLM
from .thinking_marker import ThinkingMarkerDetector
from .thinking_sentence import ThinkingSentenceDetector

# Backward compatibility alias
StepBoundaryDetector = StructuredStepDetector

__all__ = [
    # Base
    "StepBoundaryDetectorBase",
    # Structured response detector (non-thinking mode)
    "StructuredStepDetector",
    "StepBoundaryDetector",  # Backward compatibility alias
    # Thinking mode detectors
    "ThinkingSentenceDetector",
    "ThinkingMarkerDetector",
    "ThinkingLLMDetector",
    "ThinkingLLMDetectorVLLM",
    "ThinkingHybridDetector",
    "ThinkingAdaptiveDetector",
]

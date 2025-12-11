"""
Step boundary detectors for reasoning traces.

This module provides different strategies for detecting step boundaries
in LLM reasoning outputs, supporting both structured responses and
native thinking mode.

Detectors:
---------
Structured Response Detectors:
    StepBoundaryDetector: For explicit step markers (- Step 1:, - Step 2:, etc.)

Thinking Mode Detectors:
    ThinkingSentenceDetector: Split by sentences/paragraphs
    ThinkingMarkerDetector: Split by linguistic markers (so, therefore, let me...)
    ThinkingLLMDetector: Use secondary LLM to parse steps
    ThinkingHybridDetector: Combine markers with fallback to sentences
    ThinkingAdaptiveDetector: Auto-select strategy based on content

Usage:
------
# For structured responses (non-thinking mode)
from llm_tts.step_boundary_detectors import StepBoundaryDetector
detector = StepBoundaryDetector(
    step_patterns=["- Step"],
    answer_patterns=["<Answer>:"],
    max_tokens_per_step=512,
)
steps = detector.detect_steps(response_text)

# For native thinking mode
from llm_tts.step_boundary_detectors import ThinkingHybridDetector
detector = ThinkingHybridDetector(
    min_steps=3,
    max_steps=15,
)
steps = detector.detect_steps(thinking_content)
"""

from .base import StepBoundaryDetector, StepBoundaryDetectorBase
from .thinking_hybrid import ThinkingAdaptiveDetector, ThinkingHybridDetector
from .thinking_llm import ThinkingLLMDetector, ThinkingLLMDetectorVLLM
from .thinking_marker import ThinkingMarkerDetector
from .thinking_sentence import ThinkingSentenceDetector

__all__ = [
    # Base
    "StepBoundaryDetectorBase",
    "StepBoundaryDetector",
    # Thinking mode detectors
    "ThinkingSentenceDetector",
    "ThinkingMarkerDetector",
    "ThinkingLLMDetector",
    "ThinkingLLMDetectorVLLM",
    "ThinkingHybridDetector",
    "ThinkingAdaptiveDetector",
]

"""
Step boundary detectors for reasoning traces.

This module provides different strategies for detecting step boundaries
in LLM reasoning outputs, supporting both structured responses and
native thinking mode.

Structure:
---------
step_boundary_detectors/
├── base.py                 # Abstract base class
├── non_thinking/           # For structured responses (- Step 1:, - Step 2:)
│   └── structured.py       # StructuredStepDetector
└── thinking/               # For native thinking mode (<think> tags)
    ├── sentence.py         # ThinkingSentenceDetector
    ├── marker.py           # ThinkingMarkerDetector
    ├── hybrid.py           # ThinkingHybridDetector, ThinkingAdaptiveDetector
    └── llm.py              # ThinkingLLMDetector, ThinkingLLMDetectorVLLM

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

# For native thinking mode (recommended: marker with v2 config)
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
from .thinking import (
    ThinkingAdaptiveDetector,
    ThinkingHybridDetector,
    ThinkingLLMDetector,
    ThinkingLLMDetectorVLLM,
    ThinkingMarkerDetector,
    ThinkingSentenceDetector,
)

__all__ = [
    # Base
    "StepBoundaryDetectorBase",
    # Non-thinking mode (structured responses)
    "StructuredStepDetector",
    # Thinking mode detectors
    "ThinkingSentenceDetector",
    "ThinkingMarkerDetector",
    "ThinkingLLMDetector",
    "ThinkingLLMDetectorVLLM",
    "ThinkingHybridDetector",
    "ThinkingAdaptiveDetector",
]

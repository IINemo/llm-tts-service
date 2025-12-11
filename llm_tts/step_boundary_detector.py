"""
Backward compatibility module.

This module re-exports StepBoundaryDetector from the new location
for backward compatibility with existing code.

New code should import from llm_tts.step_boundary_detectors instead.
"""

from llm_tts.step_boundary_detectors import StepBoundaryDetector

__all__ = ["StepBoundaryDetector"]

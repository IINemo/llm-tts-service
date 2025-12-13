"""
Step boundary detectors for non-thinking mode (structured responses).

These detectors are designed for responses with explicit step markers
like "- Step 1:", "- Step 2:", etc.
"""

from .structured import StructuredStepDetector

__all__ = [
    "StructuredStepDetector",
]

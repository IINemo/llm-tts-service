"""HuggingFace-specific step boundary detection utilities.

The main HuggingFace integration is in:
- llm_tts.early_stopping.BatchStepStoppingCriteria
- llm_tts.generators.huggingface.StepCandidateGeneratorThroughHuggingface

This module can be extended with HF-specific utilities as needed.
"""

# Re-export from early_stopping for convenience
try:
    from llm_tts.early_stopping import BatchStepStoppingCriteria
except ImportError:
    BatchStepStoppingCriteria = None

__all__ = [
    "BatchStepStoppingCriteria",
]

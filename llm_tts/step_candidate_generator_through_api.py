"""
Candidate step generation system for online best-of-n using API models
"""

import time
from typing import List
import logging

from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    StepCandidateGeneratorBase,
)
from llm_tts.step_detection import StepBoundaryDetector

from lm_polygraph import BlackboxModel

log = logging.getLogger(__name__)


class StepCandidateGeneratorThroughAPI(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n using API models"""

    def __init__(
        self,
        model: BlackboxModel,
        detector: StepBoundaryDetector,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ):
        self.model = model
        self.detector = detector or StepBoundaryDetector()
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def generate_candidates(
        self, request, candidates_per_step: int
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        log.info(f"Generating {candidates_per_step} candidates from trajectory")

        candidates = []

        start_time = time.time()

        # Generate multiple candidates by making multiple API calls
        for i in range(candidates_per_step):
            log.info(f"Generating candidate {i+1}/{candidates_per_step}")

            # Use the model's generate_texts method which handles streaming
            results = self.model.generate_texts([request])

            if results and len(results) > 0:
                result = results[0]

                # Extract step using detector
                step_text = self.detector.extract_step_text(
                    result.get("raw_collected", "")
                )
                is_complete = self.detector.is_step_complete(
                    result.get("raw_collected", "")
                )
                is_trajectory_complete = self.detector.is_trajectory_complete(
                    result.get("raw_collected", "")
                )

                # Create StepCandidate object
                candidate = StepCandidate(
                    text=step_text,
                    token_ids=[],  # API models don't provide token IDs
                    is_complete=is_complete,
                    is_trajectory_complete=is_trajectory_complete,
                    generation_scores=None,  # API models don't provide generation scores
                    raw_text=result.get("raw_collected", ""),
                )
                candidates.append(candidate)
            else:
                log.warning(f"No result returned for candidate {i+1}")
                # Create empty candidate
                candidate = StepCandidate(
                    text="",
                    token_ids=[],
                    is_complete=False,
                    is_trajectory_complete=False,
                    generation_scores=None,
                    raw_text="",
                )
                candidates.append(candidate)

        generation_time = time.time() - start_time
        log.info(f"Generated {len(candidates)} candidates in {generation_time:.2f}s")

        return candidates

    def generate_answer(self, request, candidates_per_step: int) -> str:
        """Generate and select best final answer based on criterion"""

        return self.generate_candidates(request, candidates_per_step)

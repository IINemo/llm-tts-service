"""
Candidate step generation system for online best-of-n using API models
"""

import logging
import time
import copy
from typing import List, Dict

from lm_polygraph import BlackboxModel

from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    StepCandidateGeneratorBase,
    covert_trajectory_to_string
)
from llm_tts.step_boundary_detector import StepBoundaryDetector

log = logging.getLogger(__name__)


class StepCandidateGeneratorThroughAPI(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n using API models"""

    def __init__(
        self,
        model: BlackboxModel,
        detector: StepBoundaryDetector,
        prefill_mode: bool,
    ):
        self.model = model
        self.detector = detector or StepBoundaryDetector()
        self.prefill_mode = prefill_mode

    def generate_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        log.info(f"Generating {candidates_per_step} candidates from trajectory")

        start_time = time.time()
        
        candidates = []
        request_with_trajectory = self._prepare_request(request, trajectory)

        # Generate multiple candidates by making multiple API calls
        for i in range(candidates_per_step):
            log.info(f"Generating candidate {i+1}/{candidates_per_step}")

            # Use the model's generate_texts method which handles streaming
            results = self.model.generate_texts([request_with_trajectory])

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
                raise ValueError(f"No result returned for candidate {i+1}")

        generation_time = time.time() - start_time
        log.info(f"Generated {len(candidates)} candidates in {generation_time:.2f}s")

        return candidates

    def generate_answer_candidates(
        self,
        request: List[Dict[str, str]],
        trajectory: List[StepCandidate],
        candidates_per_step: int,
    ) -> List[StepCandidate]:
        """Generate and select best final answer based on criterion"""

        final_trajectory = [e for e in trajectory]
        final_trajectory.append(StepCandidate(
            text="\n<Answer>:\n", # TODO: get configuration from the step boundary detector
            token_ids=[],
            is_complete=False,
            is_trajectory_complete=False,
            generation_scores=None,
            raw_text="\n<Answer>:\n",
        ))

        candidates = self.generate_candidates(
            request, final_trajectory, candidates_per_step
        )
        for cand in candidates:
            cand.is_trajectory_complete = True
            cand.text = "\n<Answer>:\n" + cand.text
            cand.raw_text = "\n<Answer>:\n" + cand.raw_text

        return candidates

    def _prepare_request(
        self, request: List[Dict[str, str]], trajectory: List[StepCandidate]
    ):
        request_with_trajectory = copy.deepcopy(request)

        if not trajectory:
            request_with_trajectory = request

        else:
            if not self.prefill_mode:
                request_with_trajectory = self._add_prefix_to_request(
                    request_with_trajectory, trajectory
                )
            else:
                request_with_trajectory.append(
                    {
                        "role": "assistant",
                        "content": covert_trajectory_to_string(trajectory),
                        "prefix": True,
                    }
                )

        return request_with_trajectory

    def _add_prefix_to_request(
        self, request: List[Dict[str, str]], trajectory: List[StepCandidate]
    ):
        continuation_request = request
        prefix = covert_trajectory_to_string(trajectory)
        continuation_promt = (
            "Continue the assistant message from the EXACT prefix below. "
            "Begin immediately after the last character of the prefix. "
            "Output ONLY the final answer text; do NOT include steps, chain-of-thought, or any preface. "
            "Do NOT repeat the prefix.\n"
            "----- PREFIX START -----\n"
            f"{prefix}\n"
            "----- PREFIX END -----"
        )

        log.debug(f"Continuation prompt: {continuation_promt}")
        continuation_request.append({"role": "user", "content": continuation_promt})

        return continuation_request

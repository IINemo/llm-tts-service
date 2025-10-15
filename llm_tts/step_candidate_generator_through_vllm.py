import logging
from typing import List

import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput

from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.step_candidate_generator_base import (
    StepCandidate,
    StepCandidateGeneratorBase,
)

log = logging.getLogger(__name__)


class StepCandidateGeneratorThroughVLLM(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n using API models"""

    def __init__(
        self,
        model: LLM,
        detector: StepBoundaryDetector,
        sampling_params: SamplingParams,
    ):
        self.model = model
        self.detector = detector or StepBoundaryDetector()
        self.sampling_params = sampling_params

    def calculate_perplexity(self, candidate: CompletionOutput) -> float:
        """Calculate perplexity of the response"""
        res = 0
        for token, logprob in zip(candidate.token_ids, candidate.logprobs):
            res += logprob[token].logprob
        return -1.0 * res / len(candidate.token_ids)

    def calculate_mean_token_entropy(self, candidate: CompletionOutput) -> float:
        """Calculate mean token entropy of the response"""
        res = 0
        for logprob in candidate.logprobs:
            for k, v in logprob.items():
                res += v.logprob * np.exp(v.logprob)
        return -1.0 * res / len(candidate.token_ids)

    def generate_candidates(
        self, request, candidates_per_step: int
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        log.info(f"Generating {candidates_per_step} candidates from trajectory")

        candidates = []

        self.sampling_params.n = candidates_per_step
        answers = self.model.generate(request, sampling_params=self.sampling_params)[
            0
        ].outputs
        # return answers
        for i in range(candidates_per_step):
            # Extract step using detector
            step_text = answers[i].text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(
                answers[i].text
            )
            token_ids = answers[i].token_ids
            generation_scores = answers[i].logprobs
            generation_scores = {
                "perplexity": self.calculate_perplexity(answers[i]),
                "mean_entropy": self.calculate_mean_token_entropy(answers[i]),
            }
            # Create StepCandidate object
            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=generation_scores,
                raw_text=answers[i].text,
            )
            candidates.append(candidate)

        return candidates

    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> str:
        """Generate and select best final answer based on criterion"""
        if not more_information:
            return self.generate_candidates(request, candidates_per_step)

        return self.generate_candidates(request, candidates_per_step)

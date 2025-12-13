from typing import List

import numpy as np
from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput

from llm_tts.generators.base import (
    StepCandidate,
    StepCandidateGeneratorBase,
)
from llm_tts.step_boundary_detectors import StructuredStepDetector

# log = logging.getLogger(__name__)


class StepCandidateGeneratorThroughVLLM(StepCandidateGeneratorBase):
    """Generates N candidate next steps for online best-of-n using API models"""

    def __init__(
        self,
        model: LLM,
        detector: StructuredStepDetector,
        sampling_params: SamplingParams,
    ):
        self.model = model
        self.detector = detector or StructuredStepDetector()
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
        self, request, candidates_per_step: int = 1
    ) -> List[StepCandidate]:
        """Generate N candidate next steps from current trajectory"""

        # log.info(f"Generating {candidates_per_step} candidates from trajectory")

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

    # Single request
    def generate_answer(
        self, request, candidates_per_step: int, more_information=False
    ) -> str:
        """Generate and select best final answer based on criterion"""
        if not more_information:
            return self.generate_candidates(request, candidates_per_step)

        return self.generate_candidates(request, candidates_per_step)

    # Batch request
    def generate_batch(
        self, requests: List[str], candidates_per_step: int = 1
    ) -> List[StepCandidate]:
        """Generate batch of completions from requests"""
        self.sampling_params.n = candidates_per_step
        vllm_outputs = self.model.generate(
            requests, sampling_params=self.sampling_params
        )
        formated_vllm_output = [
            vllm_outputs[i].outputs[0] for i in range(len(requests))
        ]
        result = []
        for i in range(len(requests)):
            step_text = formated_vllm_output[i].text
            is_complete = True

            is_trajectory_complete = self.detector.is_trajectory_complete(
                formated_vllm_output[i].text
            )
            token_ids = formated_vllm_output[i].token_ids
            generation_scores = formated_vllm_output[i].logprobs
            generation_scores = {
                "perplexity": self.calculate_perplexity(formated_vllm_output[i]),
                "mean_entropy": self.calculate_mean_token_entropy(
                    formated_vllm_output[i]
                ),
            }
            # Create StepCandidate object
            candidate = StepCandidate(
                text=step_text,
                token_ids=token_ids,
                is_complete=is_complete,
                is_trajectory_complete=is_trajectory_complete,
                generation_scores=generation_scores,
                raw_text=formated_vllm_output[i].text,
            )
            result.append(candidate)

        return result


if __name__ == "__main__":
    model = LLM(
        model="Qwen/Qwen3-8B",
        gpu_memory_utilization=0.9,
        max_model_len=32768,
    )
    answer_patterns = [
        "<Answer>:",
        "\n<Answer>:",
        "\n\nAnswer:",
        "Final Answer:",
        "The answer is",
    ]
    step_patterns = [
        "\n",
        "\n- Step",
        "- Step",
        "\nStep",
        "\n\nStep",
        "## Step",
    ]

    detector = StructuredStepDetector(
        answer_patterns=answer_patterns,
        step_patterns=step_patterns,
        max_tokens_per_step=2048,
    )

    step_candidate_generator = StepCandidateGeneratorThroughVLLM(
        model=model,
        detector=detector,
        sampling_params=SamplingParams(
            min_tokens=100,
            max_tokens=2048,
            logprobs=20,
            stop=step_patterns,
        ),
    )

    print(step_candidate_generator.generate_candidates("What is 2+2?"))

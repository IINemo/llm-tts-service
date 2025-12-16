"""
vLLM wrapper with uncertainty estimation, similar to CausalLMWithUncertainty.

Reference implementation (HuggingFace):
    lm_polygraph/utils/causal_lm_with_uncertainty.py
    https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/utils/causal_lm_with_uncertainty.py

Usage:
    from lm_polygraph.estimators import Perplexity
    from lm_polygraph.stat_calculators import VLLMLogprobsCalculator, EntropyCalculator
    from vllm import LLM, SamplingParams

    llm = LLM(model="model_path")

    # For Perplexity
    stat_calculators = [VLLMLogprobsCalculator()]
    estimator = Perplexity()

    # For MeanTokenEntropy
    stat_calculators = [VLLMLogprobsCalculator(), EntropyCalculator()]
    estimator = MeanTokenEntropy()

    llm_with_uncertainty = VLLMWithUncertainty(llm, stat_calculators, estimator)
    outputs = llm_with_uncertainty.generate(prompts, sampling_params)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from lm_polygraph.estimators import Estimator
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


@dataclass
class RequestOutputWithUncertainty:
    """Extends vLLM RequestOutput to include uncertainty scores."""

    request_output: RequestOutput
    uncertainty_scores: List[float]  # One score per output sequence
    deps: Optional[List[Dict]] = None  # Dependencies from stat calculators (if keep_deps=True)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped RequestOutput."""
        return getattr(self.request_output, name)


class VLLMWithUncertainty:
    """
    Wraps vLLM LLM with uncertainty estimation using lm-polygraph estimators.

    Similar to CausalLMWithUncertainty but for vLLM backend.

    Args:
        llm: vLLM LLM instance
        stat_calculators: List of stat calculators (e.g., [VLLMLogprobsCalculator(), EntropyCalculator()])
        estimator: lm-polygraph Estimator (e.g., Perplexity, MeanTokenEntropy)
        keep_deps: If True, include computed dependencies (stats) in output for debugging/analysis
    """

    def __init__(
        self,
        llm: LLM,
        stat_calculators: List,
        estimator: Estimator,
        keep_deps: bool = False,
    ):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.stat_calculators = stat_calculators
        self.estimator = estimator
        self.keep_deps = keep_deps

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutputWithUncertainty]:
        """
        Generate completions with uncertainty scores.

        Args:
            prompts: Input prompts (single string or list)
            sampling_params: SamplingParams (must have logprobs > 0)

        Returns:
            List of RequestOutputWithUncertainty with uncertainty_scores
        """
        # Ensure logprobs enabled
        if sampling_params and (sampling_params.logprobs is None or sampling_params.logprobs == 0):
            sampling_params.logprobs = 20

        # Generate with vLLM
        outputs = self.llm.generate(prompts, sampling_params)

        # Compute uncertainty for each request
        results = []
        for request_output in outputs:
            request_scores = []
            request_deps = [] if self.keep_deps else None

            for output in request_output.outputs:
                # Build deps dict with vLLM output for stat_calculators
                deps = {"vllm_output": output}

                # Run stat calculators
                for calc in self.stat_calculators:
                    deps.update(calc(deps))

                # Call estimator
                uncertainty = self.estimator(deps)
                request_scores.append(float(uncertainty[0]))

                # Store deps if requested (remove vllm_output to avoid serialization issues)
                if self.keep_deps:
                    deps_copy = {k: v for k, v in deps.items() if k != "vllm_output"}
                    request_deps.append(deps_copy)

            results.append(
                RequestOutputWithUncertainty(
                    request_output=request_output,
                    uncertainty_scores=request_scores,
                    deps=request_deps,
                )
            )

        return results

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped LLM."""
        return getattr(self.llm, name)

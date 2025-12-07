import sys

import torch
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.generators import StepCandidateGeneratorThroughHuggingface
from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.strategies.strategy_uncertainty_cot import StrategyUncertaintyCoT

sys.path.insert(0, ".")
from omegaconf import OmegaConf  # noqa: E402

from config.scorer.uncertainty_pd import create_uncertainty_model  # noqa: E402

prompt_template = """You will be presented with a <Question>. Before providing the [Answer], you should first think step-by-step carefully.

Your response format:
<start of response>
Reasoning Steps:
- Step 1: Your reasoning step 1
- Step 2: Your reasoning step 2
- Step 3: Your reasoning step 3
...
- Step N: Your reasoning step N
<Answer>: Your final answer
<end of response>

Follow the above output format STRICTLY! Do not add any other additional texts outside the template.
Keep each reasoning step concise (single steps should not be too long).
Each reasoning step must be on a single line (no line breaks within a step).

Now answer:
<Question>: {question}
"""


def create_request(question):
    request = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt_template.format(question=question)},
    ]
    return request


def test_uncertainty_guided_cot_with_whitebox():
    max_new_tokens = 64
    max_length = 512
    step_patterns = ["Step:"]
    answer_patterns = ["Answer:"]
    candidates_per_step = 2
    max_steps = 2
    generation_batch_size = 2

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen3-0.6B"
    omega_config = OmegaConf.create(
        {"model": {"model_path": model_path, "device": device}}
    )
    polygraph_model = create_uncertainty_model(omega_config)
    # Attach generation params for the whitebox generator
    polygraph_model.generation_parameters = GenerationParameters()

    step_boundary_detector = StepBoundaryDetector(
        step_patterns=step_patterns,
        answer_patterns=answer_patterns,
        max_tokens_per_step=max_new_tokens,
    )
    step_generator = StepCandidateGeneratorThroughHuggingface(
        polygraph_model,
        step_boundary_detector,
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        generation_batch_size=generation_batch_size,
        disable_thinking_mode=True,
    )

    # Use the step generator inside uncertainty-guided CoT
    strategy = StrategyUncertaintyCoT(
        step_generator=step_generator,
        model_name="local",
        candidates_per_step=candidates_per_step,
        max_steps=max_steps,
        max_empty_steps=1,
        uncertainty_threshold=0.3,
        uncertainty_sampling="token",
    )

    question = "3 + 4 = ?"
    request = create_request(question)
    result = strategy.generate_trajectory(request)

    assert "trajectory" in result and isinstance(result["trajectory"], str)
    assert "uncertainties" in result and isinstance(result["uncertainties"], list)
    # Ensure we probed uncertainty at least once
    assert len(result["uncertainties"]) >= 1
    # Ensure steps were collected
    assert len(result["steps"]) >= 1

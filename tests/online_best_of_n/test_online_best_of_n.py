import sys

import torch
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.scorers.step_scorer_uncertainty import StepScorerUncertainty
from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.step_candidate_generator_through_huggingface import (
    StepCandidateGeneratorThroughHuggingface,
)
from llm_tts.strategies import StrategyOnlineBestOfN

sys.path.insert(0, ".")
from omegaconf import OmegaConf

from config.scorer.uncertainty_entropy import create_uncertainty_model

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


def test_online_gest_of_n():
    max_new_tokens = 100
    step_patterns = ["- Step", "<Answer>:", "\n<Answer>:"]
    answer_patterns = ["<Answer>:", "\n<Answer>:"]
    candidates_per_step = 3
    max_steps = 3
    generation_batch_size = 2

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen3-0.6B"
    omega_config = OmegaConf.create(
        {"model": {"model_path": model_path, "device": device}}
    )
    polygraph_model = create_uncertainty_model(omega_config)
    polygraph_model.generation_parameters = GenerationParameters()

    step_boundary_detector = StepBoundaryDetector(
        step_patterns=step_patterns,
        answer_patterns=answer_patterns,
        max_tokens_per_step=max_new_tokens,
    )
    step_generator = StepCandidateGeneratorThroughHuggingface(
        polygraph_model,
        step_boundary_detector,
        temperature=0.5,
        top_p=1.0,
        top_k=50,
        max_new_tokens=max_new_tokens,
        generation_batch_size=generation_batch_size,
        disable_thinking_mode=True,
    )
    scorer = StepScorerUncertainty()
    strategy = StrategyOnlineBestOfN(
        step_generator=step_generator,
        scorer=scorer,
        candidates_per_step=candidates_per_step,
        max_steps=max_steps,
    )

    question = "Tom had 8 apples. He gave 3 to his friend and bought 5 more. How many apples does Tom have now?"
    request = create_request(question)
    result = strategy.generate_trajectory(request)
    print(result)

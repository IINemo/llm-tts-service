import sys

import torch
from lm_polygraph.utils.generation_parameters import GenerationParameters

from llm_tts.strategies import StrategyOfflineBestOfN

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


def test_offline_best_of_n():
    """Test offline best-of-n strategy that generates complete trajectories directly."""
    trajectories = 3  # Generate 3 complete trajectories
    max_tokens = 200
    temperature = 0.7
    top_p = 1.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen/Qwen3-0.6B"
    omega_config = OmegaConf.create(
        {"model": {"model_path": model_path, "device": device}}
    )
    polygraph_model = create_uncertainty_model(omega_config)
    polygraph_model.generation_parameters = GenerationParameters()

    strategy = StrategyOfflineBestOfN(
        model=polygraph_model,
        trajectories=trajectories,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    question = "Tom had 8 apples. He gave 3 to his friend and bought 5 more. How many apples does Tom have now?"
    request = create_request(question)
    result = strategy.generate_trajectory(request)
    
    # Verify the result structure
    assert "trajectory" in result
    assert "steps" in result
    assert "validity_scores" in result
    assert "completed" in result
    
    # Verify we have the expected number of trajectories
    assert len(result["validity_scores"]) == trajectories
    
    # Verify the trajectory is not empty
    assert result["completed"] == True
    assert len(result["trajectory"]) > 0
    
    print("Offline Best-of-N Test Results:")
    print(f"Generated {len(result['validity_scores'])} trajectories")
    print(f"Best trajectory score: {max(result['validity_scores'])}")
    print(f"Trajectory: {result['trajectory'][:200]}...")
    
    return result


if __name__ == "__main__":
    test_offline_best_of_n()
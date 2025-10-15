import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

# initialize policy model
model_name_or_path = "Qwen/Qwen3-1.7B"
vllm = LLM(model=model_name_or_path, gpu_memory_utilization=0.45, max_model_len=8192 * 2)

# define critimic model to select best candidate. If critic_model = None -> select best candidate by estimator score.
critic_model = None
# critic_model = LLM(model="GenPRM/GenPRM-1.5B", gpu_memory_utilization=0.45, max_model_len=8192 *2)

from llm_tts.step_boundary_detector import StepBoundaryDetector
from llm_tts.step_candidate_generator_through_vllm import (
    StepCandidateGeneratorThroughVLLM,
)
from llm_tts.strategies import MUR

answer_patterns = [
    "<Answer>:",
    "\n<Answer>:",
    "\n\nAnswer:",
    "Final Answer:",
    "The answer is",
]
step_patterns = [
    "\n- Step",
    "- Step",
    "\nStep",
    "\n\nStep",
    "## Step",
]

detector = StepBoundaryDetector(
    answer_patterns=answer_patterns,
    step_patterns=step_patterns,
    max_tokens_per_step=2048,
)

step_candidate_generator = StepCandidateGeneratorThroughVLLM(
    model=vllm,
    detector=detector,
    sampling_params=SamplingParams(
        max_tokens=2048, logprobs=20, stop=step_patterns, temperature=0.6
    ),
)

mur = MUR(
    step_candidate_generator=step_candidate_generator,
    max_steps=5,
    temperature=0.6,
    candidate_num=4,
    select_best="entropy",
)

prompt = """Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"""
result = mur.generate_trajectory(prompt)

print(result)

import logging
import re
from typing import Dict, List, Union

import numpy as np
from lm_polygraph.model_adapters import WhiteboxModelvLLM
from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator
from lm_polygraph.utils.generation_parameters import GenerationParameters
from vllm import SamplingParams

log = logging.getLogger(__name__)


def build_policy_input(tokenizer, question, traj, step_idx):
    chat = [
        {
            "role": "user",
            "content": f"Q: {question}\n Break problem and solve it step by step (Step0, Step1, Step2,...). Always end your solution with the phrase 'the answer is' followed by your final answer. Start your solution with 'Step{step_idx}:'\n",
        }
    ]
    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, enable_thinking=False, add_generation_prompt=True
    )
    input_text = input_text.replace(tokenizer.eos_token, "").strip()
    input_text += (
        "\n".join(traj) + f"\nStep{step_idx}:" if step_idx > 0 else f"\nStep0:"
    )
    return input_text


def ciritique_last_generation_math(problem, solution):

    system_prompt = "You are a math teacher. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{Yes}` if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect."

    user_prompt = f"""[Math Problem]

    {problem}

    [Solution]

    """

    if "the reasoning steps are:" in solution[0].lower():
        solution = solution[1:]
    for i, para in enumerate(solution):
        user_prompt += f"<paragraph_{i}>\n{para}\n</paragraph_{i}>\n\n"

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


class MUR:
    """
    MUR: Momentum Uncertainty guided Reasoning for Large Language Models
    Reimplement from https://github.com/yayayacc/MUR/blob/main/guided_search-mur.py
    """

    def __init__(
        self,
        vllm_model,
        estimators,
        max_steps: int,
        temperature: float,
        candidate_num: int = 4,
        max_tokens: int = 512,
        momentum_rate: float = 0.9,
        scaling_rate: float = 0.9,
        logprobs: int = 1,
        device: str = "cuda:0",
        critic_model: str = None,
    ):
        self.max_steps = max_steps
        self.temperature = temperature
        self.momentum_rate = momentum_rate
        self.scaling_rate = scaling_rate
        self.calc_infer_llm = GreedyProbsCalculator()
        self.estimators = estimators
        self.candidate_num = candidate_num
        self.max_tokens = max_tokens
        self.answer_patterns = [
            "<Answer>:",
            "\n<Answer>:",
            "\n\nAnswer:",
            "Final Answer:",
            "The answer is",
        ]
        self.step_patterns = [
            "\n- Step",
            "- Step",
            "\nStep",
            "\n\nStep",
            "\n\*\*Step",
            "**Step",
            "## Step",
        ]
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            logprobs=logprobs,
            stop=self.step_patterns,
            temperature=self.temperature,
        )
        self.generation_parameters = GenerationParameters(
            stop_strings=self.step_patterns
        )
        self.model = WhiteboxModelvLLM(
            vllm_model,
            sampling_params=self.sampling_params,
            generation_parameters=self.generation_parameters,
            device=device,
        )
        self.critic_model = critic_model
        if critic_model is not None:
            self.critic_tokenizer = critic_model.get_tokenizer()

    def generate_trajectory(
        self, input: Union[str, List[Dict[str, str]]]
    ) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.

        Args:
            input: Initial prompt/question or a conversation

        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, list) and all(isinstance(m, dict) for m in input):
            prompt = input[-1]["content"]
        else:
            raise ValueError(
                "Input must be a string or a list of role-content dictionaries."
            )

        trajectory = []
        selected_steps = []
        validity_scores = []
        momentum_uncertainty, get_answer = 0.0, False
        all_policy_output_tokens = 0

        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===")
            input_prompt = build_policy_input(
                self.model.tokenizer, prompt, trajectory, step_num
            )
            # print(input_prompt)
            deps = {"input_texts": [input_prompt]}
            deps.update(
                self.calc_infer_llm(deps, texts=[input_prompt], model=self.model)
            )
            all_policy_output_tokens += len(deps["greedy_log_likelihoods"][0])
            perplexity = self.estimators(deps)[0]

            step_text = deps["greedy_texts"][0]
            if not step_text:
                log.info("No output generated, stopping")
                break

            cur_signal = -1.0 * perplexity
            log.info(f"Current signal: {cur_signal:.3f}")
            log.info(f"Momentum uncertainty: {momentum_uncertainty:.3f}")

            if (
                step_num > 0
                and np.exp(cur_signal)
                < np.exp(momentum_uncertainty) * self.scaling_rate
            ):
                # log.info("Generating candidates MUR ....")
                deps = {"input_texts": [input_prompt]}
                deps.update(
                    self.calc_infer_llm(
                        deps,
                        texts=[input_prompt] * self.candidate_num,
                        model=self.model,
                    )
                )
                candidates = deps["greedy_texts"]

                # count real tokens
                for row in deps["greedy_log_likelihoods"]:
                    count = sum(
                        1 for x in row if x not in (float("inf"), float("-inf"))
                    )
                    all_policy_output_tokens += count

                # replace inf with 0
                deps["greedy_log_likelihoods"] = [
                    [0 if x in (float("inf"), float("-inf")) else x for x in row]
                    for row in deps["greedy_log_likelihoods"]
                ]
                candidate_validity_scores = self.estimators(deps)

                # Select best candidate
                if self.critic_model is not None:
                    best_idx, step_text = self._select_best_candidate_with_critic(
                        candidates, prompt, trajectory, step_num
                    )
                else:
                    best_idx, step_text = self._select_best_candidate(
                        candidates, candidate_validity_scores
                    )
                cur_signal = candidate_validity_scores[best_idx] * -1.0

            validity_scores.append(cur_signal)

            # filter step text and add to trajectory
            step_text = self.normalize_step_text(step_text)
            trajectory.append(f"Step{str(step_num)}: {step_text}")
            selected_steps.append(step_text)
            log.info(f"Step {step_num}: {step_text}")
            # Update momentum uncertainty
            momentum_uncertainty = (
                momentum_uncertainty * self.momentum_rate
                + (1 - self.momentum_rate) * cur_signal
            )

            # Check if trajectory is complete
            if self.is_complete(selected_steps[-1]):
                log.info("Answer pattern detected - generating final answer")
                get_answer = True
                break

        # Generate final answer
        if not get_answer:
            try:
                input_prompt = build_policy_input(
                    self.model.tokenizer, prompt, trajectory, step_num
                )
                deps = {"input_texts": [input_prompt]}
                deps.update(
                    self.calc_infer_llm(deps, texts=[input_prompt], model=self.model)
                )
                all_policy_output_tokens += len(deps["greedy_log_likelihoods"][0])
                final_answer = deps["greedy_texts"][0]
                trajectory.append(f"Step{str(step_num)}: {final_answer}")
                selected_steps.append(final_answer)
            except Exception as e:
                log.error(f"Error generating final answer: {e}")

        return {
            "trajectory": "\n".join(trajectory),
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
            "all_policy_output_tokens": all_policy_output_tokens,
        }

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""
        # Min perplexity
        best_idx = min(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]

    def _select_best_candidate_with_critic(
        self, candidates: List, question: str, traj: List, step_idx: int
    ) -> tuple:
        """Select the best candidate based on logits of Yes token in Critic's model response"""
        analyze_inputs = []
        for candidate in candidates:
            critic_prompt_dict = ciritique_last_generation_math(
                question, traj + [candidate]
            )

            critic_input = self.critic_tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": critic_prompt_dict["system_prompt"]},
                    {"role": "user", "content": critic_prompt_dict["user_prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            analyze_start = (
                f"<analyze>\nLet's analyze the paragraph {step_idx} step by step: "
            )
            analyze_inputs.append(
                critic_input.replace(self.critic_tokenizer.eos_token, "").strip()
                + analyze_start
            )
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens // 2,
            temperature=self.temperature,
            stop=["</analyze>\n", "```python"],
            include_stop_str_in_output=True,
            n=1,
        )
        analyze_outputs = self.critic_model.generate(analyze_inputs, sampling_params)

        output_inputs = []
        output_start = "<output>\n**Judgement**: $\\boxed"
        for idx, out in enumerate(analyze_outputs):
            for result in out.outputs:
                analyze_text = result.text.strip()
                output_inputs.append(analyze_inputs[idx] + analyze_text + output_start)

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</output>\n", "</think>\n", "```python"],
            include_stop_str_in_output=True,
            logprobs=1,
        )
        output_outputs = self.critic_model.generate(output_inputs, sampling_params)

        yes_logps = [0.0 for _ in range(len(candidates))]
        for idx, critic_output in enumerate(output_outputs):
            for result in critic_output.outputs:
                for token_logprobs in result.logprobs:
                    for token, info in token_logprobs.items():
                        if info.decoded_token == "Yes":
                            yes_logps[idx] += np.exp(info.logprob)
                            break

        return int(np.argmax(yes_logps)), candidates[int(np.argmax(yes_logps))]

    def cleanup(self):
        """Clean up resources"""
        self.model.cleanup()

    def is_complete(self, generated_text: str) -> bool:
        """Check if current generation represents a complete step"""
        for pattern in self.answer_patterns:
            if pattern.lower() in generated_text.lower():
                return True

    def normalize_step_text(self, step_text: str) -> str:
        step_text = step_text.replace("<think>", "").strip()
        step_text = step_text.replace("</think>", "").strip()
        step_text = step_text.replace("---", "").strip()
        for pattern in self.step_patterns:
            try:
                step_text = re.sub(f"{pattern}$", "", step_text)
            except:
                pass
        return step_text.strip()

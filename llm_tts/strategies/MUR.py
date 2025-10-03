import re
import numpy as np
from typing import List, Dict
from lm_polygraph.model_adapters import WhiteboxModelvLLM
from vllm import SamplingParams

from lm_polygraph.stat_calculators.greedy_probs import GreedyProbsCalculator
from lm_polygraph.utils.generation_parameters import GenerationParameters

import logging

log = logging.getLogger(__name__)


def build_policy_input(tokenizer, question, traj, step_idx):
    chat = [{'role': 'user', 'content': f'Q: {question}\n Break problem and solve it step by step (Step0, Step1, Step2,...). Always end your solution with the phrase \'the answer is\' followed by your final answer. Start your solution with \'Step{step_idx}:\'\n'}]
    input_text = tokenizer.apply_chat_template(
        chat, tokenize=False, enable_thinking=False, add_generation_prompt=True)
    input_text = input_text.replace(tokenizer.eos_token, "").strip()
    input_text += '\n'.join(traj) + f'\nStep{step_idx}:' if step_idx > 0 else f'\nStep0:'
    return input_text
    

class MUR_No_Critic:
    """
    MUR without critic model. Use uncertainty_scores to select best candidate.
    """
    
    def __init__(
        self, 
        vllm_model,
        estimators,
        max_steps: int,
        max_new_tokens: int,
        temperature: float,
        generation_batch_size: int,
        candidate_num: int = 4,
        max_tokens: int = 256,
        momentum_rate: float = 0.9,
        scaling_rate: float = 0.8,
        logprobs: int = 1,
        device: str = "cuda:0",
    ):
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or candidate_num
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
        self.sampling_params = SamplingParams(max_tokens=self.max_tokens, logprobs=logprobs, stop=self.step_patterns, temperature=self.temperature)
        self.generation_parameters = GenerationParameters(stop_strings=self.step_patterns)
        self.model = WhiteboxModelvLLM(vllm_model, sampling_params=self.sampling_params, generation_parameters=self.generation_parameters, device=device)

    
    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.
        
        Args:
            prompt: Initial prompt/question
            
        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """

        trajectory = []
        selected_steps = []
        validity_scores = []
        momentum_uncertainty = 0.0

        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===") 
            input_prompt = build_policy_input(self.model.tokenizer, prompt, trajectory, step_num)
            # print(input_prompt)
            deps = {"input_texts": [input_prompt]}
            deps.update(self.calc_infer_llm(deps, texts=[input_prompt], model=self.model))
            perplexity = self.estimators(deps)[0]
            
            step_text = deps['greedy_texts'][0]
            if not step_text:
                log.info("No output generated, stopping")
                break
            
            cur_signal = -1.0 * perplexity
            log.info(f"Current signal: {cur_signal:.3f}")
            log.info(f"Momentum uncertainty: {momentum_uncertainty:.3f}")
            
            # print(cur_signal, momentum_uncertainty)

            if step_num > 0 and np.exp(cur_signal) < np.exp(momentum_uncertainty) * self.scaling_rate: 
                log.info("Generating candidates MUR ....")
                deps = {"input_texts": [input_prompt]}
                deps.update(self.calc_infer_llm(deps, texts=[input_prompt] * self.candidate_num, model=self.model))
                candidates = deps['greedy_texts']

                # replace inf with 0
                deps['greedy_log_likelihoods'] = [[0 if x in (float('inf'), float('-inf')) else x for x in row] 
                                                for row in deps['greedy_log_likelihoods']]
                candidate_validity_scores = self.estimators(deps)
            
                # Select best candidate 
                best_idx, step_text = self._select_best_candidate(candidates, candidate_validity_scores)
                cur_signal = candidate_validity_scores[best_idx]
            
            validity_scores.append(cur_signal)

            # filter step text and add to trajectory
            step_text = self.normalize_step_text(step_text)
            trajectory.append(f"Step{str(step_num)}: {step_text}")
            selected_steps.append(step_text)
            log.info(f"Step {step_num}: {step_text}")
            # Update momentum uncertainty
            momentum_uncertainty = momentum_uncertainty * self.momentum_rate + (1 - self.momentum_rate) * cur_signal

            # Check if trajectory is complete
            if self.is_complete(selected_steps[-1]):
                log.info("Answer pattern detected - generating final answer")
                break

        return {
            "trajectory": trajectory,
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0
        }
    

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""
        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]
        
    def cleanup(self):
        """Clean up resources"""
        self.model.cleanup()

    def is_complete(self, generated_text: str) -> bool:
        """Check if current generation represents a complete step"""
        for pattern in self.answer_patterns:
            if pattern.lower() in generated_text.lower():
                return True
                
    def normalize_step_text(self, step_text: str) -> str:
        step_text = step_text.replace("<think>","").strip()
        step_text = step_text.replace("</think>","").strip()
        step_text = step_text.replace("---", "").strip()
        for pattern in self.step_patterns: 
            try:
                step_text = re.sub(f"{pattern}$", "", step_text)
            except:
                pass
        return step_text.strip()

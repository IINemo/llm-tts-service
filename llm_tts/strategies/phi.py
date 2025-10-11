from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from vllm import LLM, SamplingParams

from .strategy_base import StrategyBase


def softmax(x):
    """
    Compute softmax values for the input array
    Args:
        x: Input array of values
    Returns:
        Softmax probabilities
    """
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


class PhiBase:
    """
    Main class for phi-decoding algorithm implementation.
    Combines clustering and sampling strategies for response selection.
    """

    def __init__(
        self,
        # Model configuration
        vllm_model: LLM = None,
        max_model_len: int = 16384,
        # Algorithm parameters
        step_beam_size: int = 2,
        num_rollout: int = 2,
        num_foresight: int = 8,
        strategy: str = "cluster",
        width_pruning_strategy: str = "low_sigma",
        depth_pruning_strategy: str = "cluster",
        cluster_num: int = 3,
        threshold: float = 0.69,
        least_foresight_num: int = 4,
        sigma_rate: float = 1.0,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        system_prompt: str = None,
        # other config
        TEMPERATURE: float = 0.1,
    ):
        """
        Initialize the decoder
        Args:
            model_path: Model path
            gpus: Number of GPUs to use
            step_beam_size: Beam size for each step
            num_rollout: Number of rollouts
            num_foresight: Number of foresight steps
            strategy: Response selection strategy
            width_pruning_strategy: Width pruning strategy
            depth_pruning_strategy: Depth pruning strategy
            cluster_num: Number of clusters for clustering strategy
            threshold: Threshold for early stopping
            least_foresight_num: Minimum number of foresight steps
            sigma_rate: Sigma rate for width pruning
            temperature: Temperature for softmax sampling
            max_tokens: Maximum number of tokens for generation
        """
        # Store all parameters
        self.max_model_len = max_model_len
        self.step_beam_size = step_beam_size
        self.num_rollout = num_rollout
        self.num_foresight = num_foresight
        self.strategy = strategy
        self.width_pruning_strategy = width_pruning_strategy
        self.depth_pruning_strategy = depth_pruning_strategy
        self.cluster_num = cluster_num
        self.threshold = threshold
        self.least_foresight_num = least_foresight_num
        self.sigma_rate = sigma_rate
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.model = vllm_model
        self.tokenizer = vllm_model.get_tokenizer()
        self.answer_patterns = [
            "<Answer>",
            "\n<Answer>:",
            "\n\nAnswer",
            "Final Answer",
            "The answer is",
        ]
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else """Break problem and solve it step by step. 
Example:
Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Step: Michael started with 58 golf balls.
Step: After losing 23 on tuesday, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.
So the answer is: 33.
NOTE: Always end your solution with the phrase 'the answer is' followed by your final answer.
""".strip()
        )
        self.TEMPERATURE = TEMPERATURE

    def get_system_prompt(self):
        return self.system_prompt

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def cluster_responses(self, responses, advantages):
        """
        Cluster responses using TF-IDF and K-means
        Args:
            responses: List of response texts
            advantages: List of advantage values for each response
        Returns:
            Tuple of (clusters, cluster_info)
        """
        # Filter out empty responses
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.step_beam_size:
            return None, {"state": "cannot cluster"}

        try:
            valid_responses = [responses[i] for i in valid_indices]

            # Vectorize responses
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(valid_responses)

            # Perform clustering
            kmeans = KMeans(n_clusters=2)  # Using 2 clusters as per paper
            kmeans.fit(X)

            # Group responses by cluster
            clusters = [[] for _ in range(2)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "cluster_sizes": [len(c) for c in clusters],
            }

        except Exception as e:
            return None, {"state": "fail", "error": str(e)}

    def select_response(self, responses, logprobs, advantages):
        """Select final response based on strategy"""
        if self.strategy == "cluster":
            # filter out empty responses
            valid_indices = []
            for idx, response in enumerate(responses):
                if response.strip() != "":
                    valid_indices.append(idx)

            if len(valid_indices) == 0:
                print(
                    "all responses in the final generation are empty, use -adv no replace"
                )
                # beacuse current responses are empty, add '-' to maximize the advantage
                weights = softmax([-adv / self.TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.step_beam_size:
                # if the number of valid responses is less than step_beam_size, use adv no replace
                print(
                    "valid responses are less than step_beam_size, use adv no replace"
                )
                weights = softmax([adv / self.TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            try:
                # prepare cluster data
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                # execute TF-IDF vectorization and clustering
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(valid_responses)
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X)
                cluster_labels = kmeans.labels_

                # build cluster list
                cluster_list = [[] for _ in range(2)]
                for idx, label in enumerate(cluster_labels):
                    cluster_list[label].append(idx)
                cluster_list = [sorted(cluster) for cluster in cluster_list]

                cluster0 = cluster_list[0]
                cluster1 = cluster_list[1]
                if len(cluster0) > len(cluster1):
                    cluster_adv_list = [valid_advantages[ddi] for ddi in cluster0]
                    weights = softmax(
                        [adv / self.TEMPERATURE for adv in cluster_adv_list]
                    )
                    selected_index_in_cluster = np.random.choice(
                        len(weights), p=weights
                    )
                    selected_index_in_tem = cluster0[selected_index_in_cluster]
                    selected_index_final = valid_indices[selected_index_in_tem]
                    return selected_index_final
                else:
                    cluster_adv_list = [valid_advantages[ddi] for ddi in cluster1]
                    weights = softmax(
                        [adv / self.TEMPERATURE for adv in cluster_adv_list]
                    )
                    selected_index_in_cluster = np.random.choice(
                        len(weights), p=weights
                    )
                    selected_index_in_tem = cluster1[selected_index_in_cluster]
                    selected_index_final = valid_indices[selected_index_in_tem]
                    return selected_index_final

            except Exception as e:
                print("cannot select response based on cluster, use adv no replace")
                weights = softmax([adv / self.TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def process_example(self, example: str):
        """
        Process a single example through the phi-decoding pipeline
        Args:
            example: Input example containing question and other fields
            system_prompt: System prompt for the model
        Returns:
            Dictionary containing results and statistics
        """
        # Initialize tracking variables
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # Initialize trajectory pools
        traj_pool = [[] for _ in range(self.num_foresight)]
        step_pool = [[] for _ in range(self.num_foresight)]
        prob_pool = [[] for _ in range(self.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.num_foresight + 1)]

        # Initialize beam states
        previous_steps = ["Step: " for _ in range(self.step_beam_size)]
        previous_values = [0.0 for _ in range(self.step_beam_size)]

        traj_info = {
            "question": example,
            "foresight_part": [],  # Will be filled during each step
            "final_part": {},  # Will be filled during final generation
            "config": {  # Add configuration information
                "num_rollout": self.num_rollout,
                "num_foresight": self.num_foresight,
                "step_beam_size": self.step_beam_size,
                "strategy": self.strategy,
                "width_pruning_strategy": self.width_pruning_strategy,
                "depth_pruning_strategy": self.depth_pruning_strategy,
                "threshold": self.threshold,
                "sigma_rate": self.sigma_rate,
                "cluster_num": self.cluster_num,
            },
        }

        # Multi-step reasoning
        for step in range(self.num_foresight):
            step_results = self._process_step(
                example,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info,  # Pass trajectory information
            )

            # Check early stopping condition
            if self._should_stop_early(step_results, step):
                break

            # Update state for next step
            previous_steps = step_results["next_steps"]
            previous_values = step_results["next_values"]

            # Record step results
            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]

        # Generate final response
        final_result = self._generate_final_response(
            example,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info,  # Pass trajectory information
        )

        # Record token statistics
        traj_info["token_num"] = token_stats["input"] + token_stats["output"]

        return {
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"],
            },
            "traj_info": traj_info,  # Add trajectory information to return result
        }

    def _process_step(
        self,
        example,
        previous_steps,
        previous_values,
        token_stats,
        rollout_stats,
        traj_info,
    ):
        """Process a single reasoning step"""
        stop_foresight = False
        # first stage: generate incomplete responses
        all_inputs = []
        for beam_idx in range(self.step_beam_size):
            chat = self._prepare_chat_template(example)
            inputs = (
                self.tokenizer.apply_chat_template(
                    chat, tokenize=False, enable_thinking=False
                )
                .rstrip(self.tokenizer.eos_token)
                .rstrip()
            )
            inputs = inputs + previous_steps[beam_idx]
            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            n=self.num_rollout,
            logprobs=0,
            temperature=self.temperature,
            stop=["\nStep:", "<END>", "</think>"],
        )

        outputs = self.model.generate(all_inputs, sampling_params)

        rollout_stats["total"] += self.num_rollout * self.step_beam_size

        # collect the results of the first stage
        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []

        for beam_idx, beam_outputs in enumerate(outputs):
            for output in beam_outputs.outputs:
                response = output.text.strip()
                logprob = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
                advantage = logprob - previous_values[beam_idx]

                all_responses_first_stage.append(response)
                all_logprobs_first_stage.append(logprob)
                all_advantages_first_stage.append(advantage)
                all_token_nums_first_stage.append(len(output.token_ids))
                token_stats["output"] += len(output.token_ids)

        # prune the responses based on width pruning_strategy
        if self.width_pruning_strategy != "none" and self.width_pruning_strategy != "":
            keep_foresight_list = []
            if self.width_pruning_strategy == "low_sigma":
                # calculate the mean and standard deviation of logprobs
                mean = np.mean(all_logprobs_first_stage)
                std = np.std(all_logprobs_first_stage)

                # keep the samples with logprob higher than mean - sigma_rate * std
                for idx, logp in enumerate(all_logprobs_first_stage):
                    if logp > mean - self.sigma_rate * std:
                        keep_foresight_list.append(idx)

            # if the number of kept samples is less than step_beam_size, then supplement
            if len(keep_foresight_list) < self.step_beam_size:
                weights = softmax(
                    [logp / self.TEMPERATURE for logp in all_logprobs_first_stage]
                )
                num_to_add = self.step_beam_size - len(keep_foresight_list)
                available_indices = [
                    i
                    for i in range(len(all_logprobs_first_stage))
                    if i not in keep_foresight_list
                ]
                if available_indices:
                    available_weights = [weights[i] for i in available_indices]
                    available_weights = [
                        w / sum(available_weights) for w in available_weights
                    ]
                    additional_indices = np.random.choice(
                        available_indices,
                        # size=min(num_to_add, len(available_indices)),
                        size=num_to_add,
                        p=available_weights,
                        replace=False,
                    ).tolist()
                    keep_foresight_list.extend(additional_indices)

            keep_foresight_list.sort()

            # update the statistics
            rollout_stats["saved"] += self.step_beam_size * self.num_rollout - len(
                keep_foresight_list
            )

            # only keep the selected samples
            filtered_responses = [
                all_responses_first_stage[i] for i in keep_foresight_list
            ]
            filtered_logprobs = [
                all_logprobs_first_stage[i] for i in keep_foresight_list
            ]
            filtered_advantages = [
                all_advantages_first_stage[i] for i in keep_foresight_list
            ]
            filtered_beam_indices = [i // self.num_rollout for i in keep_foresight_list]

            all_responses = filtered_responses

        # second stage: complete the responses
        completion_inputs = []
        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx] + "\nStep: "
            # if response.strip() != '':
            chat = self._prepare_chat_template(example)
            beam_idx = keep_foresight_list[idx] // self.num_rollout
            chat[-1]["content"] = previous_steps[beam_idx] + response

            inputs = (
                self.tokenizer.apply_chat_template(
                    chat, tokenize=False, enable_thinking=False
                )
                .rstrip(self.tokenizer.eos_token)
                .rstrip()
            )

            completion_inputs.append(inputs)
            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])

        # generate the completed responses
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            n=1,
            logprobs=0,
            stop=["\nStep:", "<END>", "</think>"],
        )

        completion_outputs = self.model.generate(completion_inputs, sampling_params)
        rollout_stats["total"] += len(completion_inputs)

        # collect the results of the second stage
        completed_responses = []
        completed_logprobs = []
        completed_advantages = []

        for idx, outputs in enumerate(completion_outputs):
            output = outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
            beam_idx = keep_foresight_list[idx] // self.num_rollout
            advantage = logprob - previous_values[beam_idx]

            completed_responses.append(response)
            completed_logprobs.append(logprob)
            completed_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # third stage: cluster and select the completed responses
        try:
            # execute TF-IDF vectorization and clustering
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(completed_responses)
            kmeans = KMeans(n_clusters=self.cluster_num)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_

            # build the cluster list
            cluster_list = [[] for _ in range(self.cluster_num)]
            for idx, label in enumerate(cluster_labels):
                cluster_list[label].append(idx)
            cluster_list = [sorted(cluster) for cluster in cluster_list]

            # calculate the cluster weights and advantage weights
            cluster_len_ratio = [
                len(cluster) / len(completed_responses) for cluster in cluster_list
            ]
            per_sample_cluster_len_ratio = [
                cluster_len_ratio[cluster_labels[i]]
                for i in range(len(completed_responses))
            ]
            cluster_weights = softmax(per_sample_cluster_len_ratio)
            adv_weights = softmax(
                [adv / self.TEMPERATURE for adv in completed_advantages]
            )

            # combine the weights
            weights = [
                (cluster_weights[ii] + adv_weights[ii]) / 2
                for ii in range(len(completed_responses))
            ]

            # select the samples
            selected = np.random.choice(
                len(weights), size=self.step_beam_size, p=weights, replace=False
            ).tolist()

            sizes = np.bincount(cluster_labels)
            largest_ratio = max(sizes) / len(completed_responses)

            if largest_ratio >= self.threshold:
                stop_foresight = True

            # Record information after generating first stage responses
            step_info = {
                "first_stage": {
                    "responses": all_responses_first_stage,
                    "logprobs": all_logprobs_first_stage,
                    "advantages": all_advantages_first_stage,
                    "token_nums": all_token_nums_first_stage,
                }
            }

            # Record information after width pruning
            if (
                self.width_pruning_strategy != "none"
                and self.width_pruning_strategy != ""
            ):
                step_info["width_pruning"] = {
                    "keep_indices": keep_foresight_list,
                    "filtered_responses": filtered_responses,
                    "filtered_logprobs": filtered_logprobs,
                    "filtered_advantages": filtered_advantages,
                    "filtered_beam_indices": filtered_beam_indices,
                }

            # Record information after second stage completion
            step_info["second_stage"] = {
                "responses": completed_responses,
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
            }

            # Record information after clustering and selection
            step_info["clustering"] = {
                "cluster_labels": cluster_labels.tolist(),
                "cluster_sizes": [len(cluster) for cluster in cluster_list],
                "cluster_weights": cluster_weights.tolist(),
                "adv_weights": adv_weights.tolist(),
                "combined_weights": weights,
                "selected_indices": selected,
            }

            # Record final selection results
            step_info["final"] = {
                "selected_steps": [
                    previous_steps[keep_foresight_list[idx] // self.num_rollout]
                    + all_responses_first_stage[keep_foresight_list[idx]]
                    + "\n"
                    for idx in selected
                ],
                "selected_values": [completed_logprobs[idx] for idx in selected],
                "selected_indices": selected,
            }

            # Add current step information to trajectory information
            traj_info["foresight_part"].append(step_info)

            return {
                "next_steps": [
                    previous_steps[keep_foresight_list[idx] // self.num_rollout]
                    + all_responses_first_stage[keep_foresight_list[idx]]
                    + "\n"
                    for idx in selected
                ],
                "next_values": [completed_logprobs[idx] for idx in selected],
                "trajectories": completed_responses,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
                "stop_foresight": stop_foresight,
            }

        except Exception as e:
            print(
                "when cluster during intermediate steps, error occurs, use adv no replace"
            )
            weights = softmax([adv / self.TEMPERATURE for adv in completed_advantages])
            selected = np.random.choice(
                len(weights), size=self.step_beam_size, p=weights, replace=False
            ).tolist()

            return {
                "next_steps": [
                    previous_steps[keep_foresight_list[idx] // self.num_rollout]
                    + all_responses_first_stage[keep_foresight_list[idx]]
                    + "\n"
                    for idx in selected
                ],
                "next_values": [all_logprobs_first_stage[idx] for idx in selected],
                "trajectories": all_responses_first_stage,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": all_logprobs_first_stage,
                "advantages": all_advantages_first_stage,
                "stop_foresight": stop_foresight,
            }

    def _is_answer_pattern(self, response):
        for pattern in self.answer_patterns:
            if pattern.lower() in response.lower():
                return True
        return False

    def _should_stop_early(self, step_results, current_step):
        """Check if early stopping conditions are met"""
        if current_step < self.least_foresight_num:
            return False

        just_stop = True
        first_response = step_results["trajectories"][0]
        for response in step_results["trajectories"][1:]:
            if response != first_response:
                just_stop = False
                break

        if just_stop:
            print(
                f"Early stopping at depth {current_step} (all responses are the same)"
            )
            return True

        if self.depth_pruning_strategy == "cluster":
            # Check if responses are becoming similar
            if step_results["stop_foresight"]:
                print(
                    f"Early stopping at depth {current_step} (max cluster ratio >= args.threshold)"
                )
                return True

        return False

    def trim_final_pattern_answer(self, response):
        """Trim the response by removing the lines that contain the answer patterns"""
        """ Example: 
        Step: A,B,C
        Step: D,E,F
        So the answer is ..., 
        The answer is .... and so on.
        ----> 
        We want to remove the lines that contain the answer patterns, so the result should be:
        Step: A,B,C
        Step: D,E,F
        """
        results = []
        for line in response.split("\n"):
            has_pattern = False
            for pattern in self.answer_patterns:
                if pattern.lower() in line.lower():
                    has_pattern = True
                    break
            if not has_pattern:
                results.append(line)
        return "\n".join(results).strip()

    def _generate_final_response(
        self,
        example,
        previous_steps,
        previous_values,
        token_stats,
        rollout_stats,
        traj_info,
    ):
        """Generate final response after multi-step reasoning"""
        # Prepare input for each beam
        all_inputs = []
        for beam_idx in range(self.step_beam_size):
            chat = self._prepare_chat_template(example)
            previous_steps[beam_idx] = self.trim_final_pattern_answer(
                previous_steps[beam_idx].strip()
            )
            chat[-1]["content"] = previous_steps[beam_idx]

            inputs = (
                self.tokenizer.apply_chat_template(
                    chat, tokenize=False, enable_thinking=False
                )
                .rstrip(self.tokenizer.eos_token)
                .rstrip()
            )

            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        # parallel generate all beam responses
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens, n=1, logprobs=0, stop=["</think>", "<END>"]
        )
        outputs = self.model.generate(all_inputs, sampling_params)

        rollout_stats["total"] += self.step_beam_size

        # Collect all response results
        all_responses = []
        all_logprobs = []
        all_advantages = []
        all_combined_responses = []

        for beam_idx, beam_outputs in enumerate(outputs):
            output = beam_outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / (len(output.token_ids) + 1e-8)
            advantage = logprob - previous_values[beam_idx]

            # Combine previous_steps and new response
            combined_response = previous_steps[beam_idx] + response
            all_combined_responses.append(combined_response)
            all_responses.append(response)
            all_logprobs.append(logprob)
            all_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # Select final response
        selected_idx = self.select_response(all_responses, all_logprobs, all_advantages)

        # Record final results
        traj_info["final_part"]["responses"] = all_combined_responses
        traj_info["final_part"]["responses_in_the_final_generation"] = all_responses
        traj_info["final_part"]["logprobs"] = all_logprobs
        traj_info["final_part"]["advantages"] = all_advantages
        traj_info["final_part"]["selected_idx"] = selected_idx

        return {
            "response": previous_steps[selected_idx] + all_responses[selected_idx],
            # "response_in_the_final_generation": all_responses[selected_idx],
            "trajectories": {
                "responses": all_responses,
                "logprobs": all_logprobs,
                "advantages": all_advantages,
                "selected_idx": selected_idx,
            },
        }

    def _prepare_chat_template(self, example: str):
        """
        Prepare chat template based on dataset type
        Args:
            example: Input example
        Returns:
            List of chat messages
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {example}"},
            {"role": "assistant", "content": ""},
        ]


class PhiDecoder(StrategyBase):
    def __init__(self, *args, **kwargs):
        self.decoder = PhiBase(*args, **kwargs)

    def cleanup(self):
        self.decoder.cleanup()

    def generate_trajectory(self, input: Union[str, List[Dict[str, str]]]):
        if isinstance(input, str):
            prompt = input
        elif isinstance(input, list) and all(isinstance(m, dict) for m in input):
            prompt = input[-1]["content"]
        else:
            raise ValueError(
                "Input must be a string or a list of role-content dictionaries."
            )
        result = self.decoder.process_example(prompt)
        return {
            "trajectory": result["response"],
            "steps": result["response"].split("Step:"),
            "validity_scores": [0],
            "completed": True,
            "all_policy_output_tokens": result["traj_info"]["token_num"],
        }


def main():
    """Main execution function"""
    vllm_model = LLM(
        model="Qwen/Qwen3-1.7B", trust_remote_code=True, max_model_len=8192
    )
    decoder = PhiDecoder(vllm_model=vllm_model)
    result = decoder.generate_trajectory(
        "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
    )
    print(result)


if __name__ == "__main__":
    main()

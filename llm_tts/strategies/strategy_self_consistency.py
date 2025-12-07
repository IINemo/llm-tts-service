"""
Self-consistency strategy for LLM reasoning.

Based on "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022). Generates multiple diverse reasoning paths and selects
the most consistent answer via majority voting.

Supports multiple backends:
- API-based models (OpenAI, OpenRouter) via BlackboxModelWithStreaming
- Local HuggingFace models via WhiteboxModel
- vLLM for efficient batched generation (recommended for local models)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming
from llm_tts.scorers.majority_voting import ChainMajorityVotingScorer

from .metadata_builder import StrategyMetadataBuilder
from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategySelfConsistency(StrategyBase):
    """
    Self-consistency strategy that generates multiple reasoning paths
    and selects the most consistent answer via majority voting.
    """

    def __init__(
        self,
        model,
        num_paths: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        generation_batch_size: int = None,
        scorer: Optional[Any] = None,
        n_threads: int = None,
        disable_thinking_mode: bool = True,
        seed: int = 42,
    ):
        """
        Initialize self-consistency strategy.

        Args:
            model: Language model for generation
            num_paths: Number of reasoning paths to generate
            max_new_tokens: Maximum tokens per reasoning path
            temperature: Sampling temperature (> 0 for diversity)
            generation_batch_size: Batch size for generation (None = all at once)
            scorer: Custom scorer for answer selection (defaults to majority voting)
            n_threads: Number of parallel threads for API calls (None = defaults to 4)
            disable_thinking_mode: Disable Qwen3 thinking mode (default True)
            seed: Random seed for reproducibility (default 42)
        """
        self.model = model
        self.num_paths = num_paths
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or num_paths
        # Default to 4 threads (conservative to avoid API overload/deadlock)
        self.n_threads = n_threads if n_threads is not None else 4
        self.disable_thinking_mode = disable_thinking_mode
        self.seed = seed

        # Use majority voting scorer by default
        self.scorer = scorer or ChainMajorityVotingScorer()
        if hasattr(self.scorer, "prepare_model"):
            self.scorer.prepare_model()

    def _generate_single_path(self, args) -> Optional[str]:
        """
        Generate a single reasoning path (for multithreading).

        Args:
            args: Tuple of (prompt_or_messages, path_index, total_paths)
                  where prompt_or_messages can be a string or list of message dicts

        Returns:
            Generated reasoning text, or None if error
        """
        prompt_or_messages, i, total = args

        try:
            # Check if this is an API-based model
            if isinstance(self.model, BlackboxModelWithStreaming):
                # Convert prompt to chat format if needed
                if isinstance(prompt_or_messages, list):
                    messages = prompt_or_messages
                else:
                    messages = [{"role": "user", "content": prompt_or_messages}]

                results = self.model.generate_texts(
                    chats=[messages],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

                if results and results[0] and results[0].get("text"):
                    generated_text = results[0]["text"]
                    # Return just the generated reasoning (not prompt + generation)
                    # The scorer extracts answers from this text
                    log.info(f"  Generated path {i+1}/{total}")
                    return generated_text
                else:
                    log.warning(f"  Empty generation for path {i+1}/{total}")
                    return None

            else:
                # Local model generation
                # For local models, prompt_or_messages should be a string
                prompt = (
                    prompt_or_messages
                    if isinstance(prompt_or_messages, str)
                    else prompt_or_messages[0].get("content", "")
                )
                inputs = self.model.tokenize([prompt])
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=1,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                    )

                # Decode generated path
                output_seq = outputs[0]
                new_tokens = output_seq[input_ids.shape[1] :]
                generated_text = self.model.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )

                log.info(f"  Generated path {i+1}/{total}")

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Return just the generated reasoning
                return generated_text

        except Exception as e:
            log.error(f"  Error generating path {i+1}/{total}: {e}")
            return None

    def _generate_paths_vllm(self, prompt: str) -> List[str]:
        """
        Generate multiple reasoning paths using vLLM batched generation.

        vLLM generates all paths in a single batched call with:
        - PagedAttention for memory efficiency (no OOM on long sequences)
        - Continuous batching for high throughput
        - Prefix caching for shared prompts

        Args:
            prompt: Input prompt (string or list of messages)

        Returns:
            List of generated reasoning path texts
        """
        from vllm import SamplingParams

        log.info(f"🚀 vLLM batch generation: {self.num_paths} paths...")

        # Get the vLLM engine
        llm = self.model.vllm_engine
        tokenizer = llm.get_tokenizer()

        # Prepare the prompt with chat template
        if isinstance(prompt, list):
            # Already in message format
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        # Apply chat template with enable_thinking parameter for Qwen3 models
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not self.disable_thinking_mode,
        )

        # Create sampling params for batch generation
        sampling_params = SamplingParams(
            n=self.num_paths,  # Generate all paths in ONE call
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=self.max_new_tokens,
            stop=["<end of response>", "</end of response>"],  # Early stopping
            include_stop_str_in_output=True,
            seed=self.seed,  # Reproducibility
        )

        log.info(
            f"  Generating with params: n={self.num_paths}, "
            f"temp={self.temperature}, max_tokens={self.max_new_tokens}"
        )

        # Single call generates all paths in parallel
        outputs = llm.generate([formatted_prompt], sampling_params)

        # Import answer extraction for logging
        from llm_tts.utils import extract_answer

        # Extract generated texts with token counts
        paths = []
        total_tokens = 0
        for i, output in enumerate(outputs[0].outputs):
            text = output.text
            if text:
                num_tokens = len(output.token_ids)
                total_tokens += num_tokens
                paths.append({"text": text, "num_tokens": num_tokens})
                # Extract answer for logging (like DeepConf)
                answer = extract_answer(text, answer_format="auto") or "no_answer"
                log.info(
                    f"  Path {i+1}/{self.num_paths}: "
                    f"tokens={num_tokens}, answer={answer}"
                )
                # Log full trace text (like DeepConf)
                log.info(f"{text}")
            else:
                log.warning(f"  Empty generation for path {i+1}/{self.num_paths}")

        log.info(f"✅ vLLM generated {len(paths)}/{self.num_paths} paths successfully")
        log.info(
            f"  Total tokens: {total_tokens}, Average: {total_tokens/len(paths):.0f} tokens/path"
        )
        return paths

    def generate_reasoning_paths(self, prompt: str) -> List[str]:
        """
        Generate multiple diverse reasoning paths for the given prompt.

        Args:
            prompt: Input prompt/question

        Returns:
            List of complete reasoning paths (prompt + generated reasoning)
        """
        log.info(
            f"Generating {self.num_paths} reasoning paths "
            f"with temperature {self.temperature}"
        )

        # Check if this is a vLLM model (preferred for local inference)
        if hasattr(self.model, "is_vllm") and self.model.is_vllm:
            return self._generate_paths_vllm(prompt)

        # Check if this is an API-based model
        if isinstance(self.model, BlackboxModelWithStreaming):
            # Convert prompt to chat format
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": prompt}]

            # ==================================================================================
            # CRITICAL FIX: OpenRouter Batched Generation Bug
            # ==================================================================================
            # IMPORTANT: Skip batched generation (n parameter) entirely for OpenRouter.
            #
            # BUG DESCRIPTION:
            # ----------------
            # OpenRouter does not support the OpenAI API's 'n' parameter (e.g., n=16 to generate
            # 16 completions in a single request). When attempted, it silently returns only 1
            # completion instead of n completions, AND corrupts the HTTP connection pool in the
            # shared OpenAI client instance (self.model.client).
            #
            # This corruption causes INTERMITTENT DEADLOCKS in subsequent parallel generation
            # attempts using ThreadPoolExecutor. The deadlock manifests as:
            # - Only 1 of 16 tasks executing (instead of all 16)
            # - Only 1 HTTP request sent to the API (instead of 16)
            # - The executor hangs indefinitely waiting for the remaining 15 futures
            #
            # SYMPTOMS OBSERVED:
            # ------------------
            # - Sample 20: Works perfectly (16/16 paths generated)
            # - Sample 21: Gets stuck (only 1/16 paths generated, then hangs forever)
            # - The failure is INTERMITTENT and unpredictable
            #
            # EVIDENCE FROM LOGS:
            # -------------------
            # [20:26:53] Batched generation returned 1 results instead of 16.
            #            Falling back to parallel generation with 16 threads.
            # [20:26:53] Generating 16 reasoning paths with 16 parallel threads
            # [20:26:53] HTTP Request: POST (only ONE request logged instead of 16)
            # [20:27:01] Generated path 1/16 (then hangs - no more paths generated)
            #
            # ROOT CAUSE ANALYSIS:
            # --------------------
            # 1. OpenRouter receives request with n=16 parameter
            # 2. OpenRouter ignores 'n' and returns only 1 completion (API limitation)
            # 3. Parent class's generate_texts() method expects 16 results, gets 1
            # 4. During this failed request, the shared self.model.client (OpenAI client)
            #    HTTP connection pool enters a corrupted state
            # 5. When fallback parallel generation starts, it submits 16 tasks to ThreadPoolExecutor
            # 6. Due to corrupted client state, only 1 task actually executes (sends HTTP request)
            # 7. The executor waits for remaining 15 futures that never complete -> deadlock
            #
            # WHY THIS FIX WORKS:
            # -------------------
            # By skipping the batched generation attempt entirely, we prevent the HTTP connection
            # pool corruption from ever occurring. The shared client remains in a clean state,
            # and parallel generation with ThreadPoolExecutor works reliably:
            # - All 16 tasks execute correctly
            # - All 16 HTTP requests are sent
            # - All 16 futures complete successfully
            #
            # PERFORMANCE IMPLICATIONS:
            # -------------------------
            # Direct parallel generation is actually FASTER than batched+fallback because:
            # 1. No time wasted on failed batched attempt
            # 2. No HTTP connection pool corruption overhead
            # 3. Parallel requests can be processed concurrently by OpenRouter
            #
            # VERIFIED FIX:
            # -------------
            # Tested with 30-sample GSM8K evaluation using num_paths=16:
            # - Before fix: Intermittent deadlocks (only 1/16 paths generated on random samples)
            # - After fix: 100% success rate (all samples generate 16/16 paths reliably)
            # ==================================================================================
            log.info(
                f"Using parallel generation with {self.num_paths} threads (batched generation disabled for OpenRouter)"
            )

            # Generate all paths in parallel (no queueing)
            path_args = [(messages, i, self.num_paths) for i in range(self.num_paths)]
            paths = self._parallel_generate(
                worker_func=self._generate_single_path,
                task_args=path_args,
                n_threads=self.num_paths,  # Match thread count to path count to avoid queueing
                desc=f"Generating {self.num_paths} reasoning paths",
            )

            # Filter out None values (failed generations)
            valid_paths = [p for p in paths if p]
            log.info(
                f"Successfully generated {len(valid_paths)}/{self.num_paths} paths via parallel generation"
            )
            return valid_paths
        else:
            # Fallback: Use parallel generation for local models
            log.info(
                f"Generating {self.num_paths} paths in parallel "
                f"with {self.n_threads} threads"
            )

            path_args = [(prompt, i, self.num_paths) for i in range(self.num_paths)]
            paths = self._parallel_generate(
                worker_func=self._generate_single_path,
                task_args=path_args,
                n_threads=self.n_threads,
                desc=f"Generating {self.num_paths} reasoning paths",
            )

            # Filter out None values (failed generations)
            valid_paths = [p for p in paths if p]
            log.info(
                f"Successfully generated {len(valid_paths)}/{self.num_paths} paths"
            )

            return valid_paths

    def select_best_answer(self, reasoning_paths: List) -> Dict[str, Any]:
        """
        Select the best answer using majority voting across reasoning paths.

        Args:
            reasoning_paths: List of reasoning paths (strings or dicts with 'text' and 'num_tokens')

        Returns:
            Dictionary containing:
                - best_path: The reasoning path with the most consistent answer
                - best_answer: The extracted answer
                - consensus_score: Confidence based on answer frequency
                - all_answers: All extracted answers for debugging
                - answer_distribution: Answer frequency distribution
                - all_traces: List of dicts with text, num_tokens, answer for each path
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_answer": "no_answer",
                "consensus_score": 0.0,
                "all_answers": [],
                "answer_distribution": {},
                "all_traces": [],
            }

        # Handle both string and dict formats (vLLM returns dicts with token counts)
        path_texts = []
        path_tokens = []
        for p in reasoning_paths:
            if isinstance(p, dict):
                path_texts.append(p.get("text", ""))
                path_tokens.append(p.get("num_tokens", 0))
            else:
                path_texts.append(p)
                path_tokens.append(0)  # No token info for non-vLLM paths

        # Use the scorer to get consensus scores
        scores = self.scorer.score_complete_chains(path_texts)

        # Find the path with highest consensus
        best_idx = np.argmax(scores)
        best_path = path_texts[best_idx]
        best_score = scores[best_idx]

        # Extract the best answer
        best_answer = self.scorer.extract_answer(best_path)

        # Get all answers for analysis
        all_answers = [self.scorer.extract_answer(path) for path in path_texts]

        # Calculate answer distribution
        from collections import Counter

        answer_counts = Counter(all_answers)

        log.info(
            f"Selected reasoning path {best_idx} with consensus score {best_score:.3f}"
        )
        log.info(f"Best answer: {best_answer}")
        log.info(f"Answer distribution: {dict(answer_counts)}")

        # Build all_traces with token info
        all_traces = []
        for i, (text, tokens, answer) in enumerate(
            zip(path_texts, path_tokens, all_answers)
        ):
            all_traces.append(
                {
                    "text": text,
                    "num_tokens": tokens,
                    "answer": answer,
                    "score": float(scores[i]),
                    "selected": i == best_idx,
                }
            )

        total_tokens = sum(path_tokens)
        log.info(f"Total tokens across all paths: {total_tokens}")

        return {
            "best_path": best_path,
            "best_answer": best_answer,
            "consensus_score": best_score,
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "all_paths": path_texts,
            "all_scores": scores,
            "all_traces": all_traces,
            "total_tokens": total_tokens,
        }

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point for self-consistency reasoning.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information compatible with evaluation framework
        """

        log.info(f"Starting self-consistency reasoning for prompt: {prompt[:100]}...")

        # Generate multiple reasoning paths
        reasoning_paths = self.generate_reasoning_paths(prompt)

        # Select best answer via majority voting
        result = self.select_best_answer(reasoning_paths)

        # Build metadata using StrategyMetadataBuilder
        builder = StrategyMetadataBuilder("self_consistency")

        # Add configuration
        builder.add_config(
            num_paths=len(reasoning_paths),
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            n_threads=self.n_threads,
        )

        # Add results
        builder.add_results(
            selected_answer=result["best_answer"],
            consensus_score=result["consensus_score"],
            answer_distribution=result["answer_distribution"],
        )

        # Check if we have valid paths (generation might have failed)
        if reasoning_paths and "all_paths" in result and "all_scores" in result:
            # Create path summaries for detailed analysis
            selected_idx = int(
                result["all_scores"].argmax()
                if hasattr(result["all_scores"], "argmax")
                else result["all_scores"].index(max(result["all_scores"]))
            )
            path_summaries = builder.create_path_summaries(
                paths=result["all_paths"],
                scores=result["all_scores"],
                answers=result["all_answers"],
                selected_index=selected_idx,
            )

            # Add generation details
            builder.add_generation_details(
                all_paths=result["all_paths"],
                all_scores=result["all_scores"],
                all_answers=result["all_answers"],
                path_summaries=path_summaries,
            )
        else:
            # No valid paths generated - log error
            log.error(
                f"Failed to generate any valid reasoning paths "
                f"({len(reasoning_paths)} successful out of {self.num_paths})"
            )

        # Log summary to console
        builder.log_summary(log)

        # Format output to match expected interface
        return {
            "trajectory": result["best_path"],
            "steps": [result["best_path"]],  # Single step containing full reasoning
            "validity_scores": [result["consensus_score"]],  # Consensus as validity
            "completed": bool(reasoning_paths),
            "strategy": "self_consistency",
            "extracted_answer": result[
                "best_answer"
            ],  # For run_tts_eval.py compatibility
            "metadata": builder.build(),
            "all_traces": result.get("all_traces", []),  # Token info per path
            "total_tokens": result.get("total_tokens", 0),  # Total tokens for sample
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.scorer, "cleanup"):
            self.scorer.cleanup()

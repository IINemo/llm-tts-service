"""
Direct PRM scorer that bypasses the stat calculator pipeline for efficient stepwise scoring.

Supports both HuggingFace and vLLM backends for the PRM model.
"""

import logging
import statistics
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.stat_calculators.extract_claims import Claim
from tqdm import tqdm

from llm_tts.utils import get_torch_dtype
from llm_tts.utils.flops import FLOPCalculator

from .step_scorer_reward_base import StepScorerRewardBase

# Optional vLLM import
try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# HuggingFace imports (always available as fallback)
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


class StepsExtractor(StatCalculator):
    def __init__(
        self,
        sent_separators: str = "\n",
        skip_starts: list[str] = [
            "Reasoning Steps:",
            "SOLUTION:",
            "<start of response>",
            "<end of response>",
        ],
        progress_bar: bool = True,
    ):
        super().__init__()
        self.sent_separators = sent_separators
        self.skip_starts = skip_starts
        self.progress_bar = progress_bar

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
            [
                "claims",
                "claim_texts_concatenated",
                "claim_input_texts_concatenated",
            ],
            [
                "greedy_texts",
                "greedy_tokens",
            ],
        )

    def __call__(
        self,
        dependencies: Dict[str, object],
        texts: List[str],
        model: WhiteboxModel,
        *args,
        **kwargs,
    ) -> Dict[str, List]:
        claims: list[list[Claim]] = []
        claim_texts_concatenated: list[str] = []
        claim_input_texts_concatenated: list[str] = []

        data = zip(
            texts,
            dependencies["greedy_texts"],
            dependencies["greedy_tokens"],
        )
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc="Extracting steps")
        for input_text, greedy_text, greedy_tokens in data:
            steps: list[Claim] = self.split_to_steps(
                greedy_text, greedy_tokens, model.tokenizer
            )
            claims.append(steps)
            claim_texts_concatenated += [c.claim_text for c in steps]
            claim_input_texts_concatenated += [input_text for c in steps]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def filter_claim_texts(self, claim_text: str) -> bool:
        claim_text = claim_text.strip()
        return len(claim_text) > 0 and not any(
            claim_text.lower().startswith(b.lower()) for b in self.skip_starts
        )

    def split_to_steps(
        self,
        text: str,
        tokens: list[int],
        tokenizer,
    ) -> list[Claim]:
        if not tokenizer.decode(tokens).startswith(text):
            return []

        prev_token_i, token_i = 0, 0
        prev_text_i = 0
        claims: list[Claim] = []
        for text_i in range(len(text)):
            if text[text_i] in self.sent_separators and self.filter_claim_texts(
                text[prev_text_i : text_i + 1]
            ):
                claims.append(
                    Claim(
                        claim_text=text[prev_text_i : text_i + 1].strip(),
                        sentence=text[prev_text_i : text_i + 1],
                        aligned_token_ids=list(
                            range(prev_token_i, min(token_i + 1, len(tokens) - 1))
                        ),
                    )
                )

            while (
                token_i < len(tokens)
                and tokenizer.decode(tokens[: token_i + 1]) in text[: text_i + 1]
            ):
                token_i += 1

            if text[text_i] in self.sent_separators:
                prev_text_i = text_i + 1
                prev_token_i = token_i

        if self.filter_claim_texts(text[prev_text_i:]):
            claims.append(
                Claim(
                    claim_text=text[prev_text_i:].strip(),
                    sentence=text[prev_text_i:],
                    aligned_token_ids=list(
                        range(prev_token_i, min(token_i + 1, len(tokens) - 1))
                    ),
                )
            )

        return claims


class StepScorerPRM(StepScorerRewardBase):
    """
    Direct PRM scorer that applies Process Reward Model without stat calculator pipeline.

    This implementation:
    1. Extracts claims/steps from candidates
    2. Formats them for PRM evaluation
    3. Computes step rewards directly
    4. Returns reward scores (higher = better)

    Supports both vLLM (preferred for efficiency) and HuggingFace backends.

    Args:
        prm_model_path: Path to the PRM model (e.g., "Qwen/Qwen2.5-Math-PRM-7B")
        device: Device for HuggingFace backend (e.g., "cuda:1")
        batch_size: Batch size for scoring
        torch_dtype: Torch dtype string (e.g., "bfloat16")
        use_vllm: If True, use vLLM backend (default: True if available)
        gpu_memory_utilization: GPU memory fraction for vLLM (default: 0.9)
    """

    def __init__(
        self,
        prm_model_path: str,
        device: str,
        batch_size: int,
        torch_dtype: str,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.9,
        prm_max_tokens: int = 4000,
    ):
        self.prm_model_path = prm_model_path
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.prm_max_tokens = prm_max_tokens
        self.prm_model = None
        self.prm_tokenizer = None
        self.steps_extractor = StepsExtractor(progress_bar=False)

        # PRM token/FLOP tracking
        self.flop_calculator: Optional[FLOPCalculator] = None
        self._total_prm_tokens: int = 0
        self._per_sample_prm_tokens: Dict[Any, int] = {}

        if use_vllm and not VLLM_AVAILABLE:
            log.warning("vLLM requested but not available, falling back to HuggingFace")

        self.prepare_model()

    def prepare_model(self):
        """Load PRM model and tokenizer using selected backend."""
        if self.use_vllm:
            self._prepare_vllm_model()
        else:
            self._prepare_hf_model()

    def _prepare_vllm_model(self):
        """Load PRM model using vLLM backend."""
        import os

        # Parse device to get GPU ID
        if "cuda:" in self.device:
            gpu_id = int(self.device.split(":")[1])
        else:
            gpu_id = 0

        # Get current CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        visible_gpus = [int(x) for x in cuda_visible.split(",") if x.strip()]

        # Determine the actual physical GPU ID
        if visible_gpus:
            # gpu_id is relative to visible GPUs, map to physical GPU
            if gpu_id < len(visible_gpus):
                physical_gpu = visible_gpus[gpu_id]
            else:
                physical_gpu = visible_gpus[0]
                log.warning(
                    f"Requested GPU {gpu_id} not in CUDA_VISIBLE_DEVICES={cuda_visible}, "
                    f"using GPU {physical_gpu}"
                )
        else:
            physical_gpu = gpu_id

        log.info(
            f"Loading PRM model from {self.prm_model_path} (vLLM backend) on GPU {physical_gpu}"
        )

        # Temporarily set CUDA_VISIBLE_DEVICES to only the target GPU
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)

        try:
            self.prm_model = LLM(
                model=self.prm_model_path,
                task="reward",
                trust_remote_code=True,
                dtype=self.torch_dtype,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=True,  # More stable for reward models
                max_model_len=4096,
                enable_prefix_caching=True,
            )
            self.prm_tokenizer = self.prm_model.get_tokenizer()
            log.info("vLLM PRM model loaded successfully")
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def _prepare_hf_model(self):
        """Load PRM model using HuggingFace backend."""
        log.info(f"Loading PRM model from {self.prm_model_path} (HuggingFace backend)")
        self.prm_tokenizer = AutoTokenizer.from_pretrained(
            self.prm_model_path, trust_remote_code=True
        )
        self.prm_model = AutoModel.from_pretrained(
            self.prm_model_path,
            device_map=self.device,
            torch_dtype=get_torch_dtype(self.torch_dtype),
            trust_remote_code=True,
        ).eval()

    def cleanup(self):
        """Free PRM model memory."""
        if self.prm_model is not None:
            del self.prm_model
            self.prm_model = None
        if self.prm_tokenizer is not None:
            del self.prm_tokenizer
            self.prm_tokenizer = None
        torch.cuda.empty_cache()

    def init_flop_calculator(self, model_name: str):
        """Initialize FLOP calculator for PRM token/compute tracking."""
        self.flop_calculator = FLOPCalculator(model_name, method="simple")
        log.info(
            f"PRM FLOP calculator initialized: "
            f"{self.flop_calculator.tflops_per_1k_tokens:.3f} TFLOPs/1k tokens"
        )

    def _record_prm_tokens(self, num_tokens: int, sample_id: Any = None):
        """Record PRM input tokens for tracking."""
        self._total_prm_tokens += num_tokens
        if sample_id is not None:
            self._per_sample_prm_tokens[sample_id] = (
                self._per_sample_prm_tokens.get(sample_id, 0) + num_tokens
            )

    def reset_prm_stats(self):
        """Clear per-sample PRM stats (call before each batch)."""
        self._per_sample_prm_tokens.clear()
        self._total_prm_tokens = 0

    def get_prm_stats_for(self, sample_id: Any) -> Dict[str, Any]:
        """Get PRM stats for a specific sample."""
        tokens = self._per_sample_prm_tokens.get(sample_id, 0)
        tflops = (
            self.flop_calculator.compute_tflops(tokens)
            if self.flop_calculator
            else None
        )
        return {"prm_input_tokens": tokens, "prm_tflops": tflops}

    def get_prm_total_stats(self) -> Dict[str, Any]:
        """Get aggregate PRM stats across all samples."""
        tflops = (
            self.flop_calculator.compute_tflops(self._total_prm_tokens)
            if self.flop_calculator
            else None
        )
        return {"prm_input_tokens": self._total_prm_tokens, "prm_tflops": tflops}

    def compute_claim_rewards(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Compute reward scores for claims in each candidate.

        Args:
            chat: Current chat
            candidates: List of candidate next steps
            trajectory: List of previous selected steps (StepCandidate objects)

        Returns:
            List of claim reward lists (one per candidate)
        """
        if not candidates:
            return []

        trajectory = kwargs.get("trajectory", None)

        if self.use_vllm:
            return self._compute_rewards_vllm(chat, candidates, trajectory=trajectory)
        else:
            return self._compute_rewards_hf(chat, candidates)

    def _compute_rewards_hf(
        self, chat: List[Dict[str, str]], candidates: List[str]
    ) -> List[List[float]]:
        """Compute rewards using HuggingFace backend (original implementation)."""
        all_rewards = []

        for candidate in candidates:
            # Handle both StepCandidate objects and plain strings
            candidate_text = candidate.text if hasattr(candidate, "text") else candidate
            rewards = self._score_single_candidate_hf(chat, candidate_text)
            all_rewards.append(rewards)

            # Clean up memory after each candidate
            torch.cuda.empty_cache()

        return all_rewards

    def _compute_rewards_vllm(
        self, chat: List[Dict[str, str]], candidates: List[str], trajectory: List = None
    ) -> List[List[float]]:
        """
        Compute rewards using vLLM backend with batched scoring.

        Each candidate is treated as a single step. The PRM prompt includes:
        - Previous trajectory steps (passed directly, not split)
        - The candidate as the new step to score

        Returns single reward per candidate (positive class probability).
        """
        if not candidates:
            return []

        # Extract question from chat
        question = None
        for msg in chat:
            if msg["role"] == "user":
                question = msg["content"]
                break

        if question is None:
            question = chat[-1]["content"]

        # Get trajectory steps directly (each step.text is one PRM step)
        trajectory_steps = []
        if trajectory:
            trajectory_steps = [
                step.text if hasattr(step, "text") else str(step) for step in trajectory
            ]

        # Build prompts: trajectory steps + candidate as single new step
        truncated_prompts = []
        total_prompt_tokens = 0
        num_traj_steps = len(trajectory_steps)
        any_truncated = False
        for cand_idx, candidate in enumerate(candidates):
            candidate_text = candidate.text if hasattr(candidate, "text") else candidate
            all_steps = trajectory_steps + [candidate_text]
            prompt, num_skipped = self._truncate_steps_from_tail(
                question, all_steps, self.prm_max_tokens
            )
            tokens = self.prm_tokenizer.encode(prompt)
            total_prompt_tokens += len(tokens)
            truncated_prompts.append(prompt)
            if num_skipped > 0:
                any_truncated = True
                log.info(
                    f"  Candidate {cand_idx}: {len(all_steps)} total steps -> "
                    f"skipped {num_skipped}, included {len(all_steps) - num_skipped}, "
                    f"prompt_tokens={len(tokens)}"
                )

        # Track PRM tokens
        self._record_prm_tokens(total_prompt_tokens)

        log.info(
            f"PRM scoring {len(truncated_prompts)} candidates "
            f"(trajectory={num_traj_steps} steps, "
            f"total_prompt_tokens={total_prompt_tokens}, "
            f"max_tokens={self.prm_max_tokens}"
            f"{', SOME TRUNCATED' if any_truncated else ''})"
        )
        outputs = self.prm_model.reward(truncated_prompts, use_tqdm=True)

        # Extract reward for the last step (the candidate) from each output
        all_rewards = []
        for cand_idx, output in enumerate(outputs):
            reward = 0.0
            all_step_rewards = []
            if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
                data = output.outputs.data
                if hasattr(data, "tolist"):
                    step_scores = data.tolist()
                elif isinstance(data, list):
                    step_scores = data
                else:
                    step_scores = [data]

                # Extract positive probability from each [neg, pos] pair
                for score in step_scores:
                    if isinstance(score, (list, tuple)) and len(score) == 2:
                        all_step_rewards.append(score[1])
                    else:
                        all_step_rewards.append(float(score))

                # Last score is the candidate's score
                if all_step_rewards:
                    reward = all_step_rewards[-1]

            all_rewards.append((reward, all_step_rewards))
            # Log all step scores: preceding steps + candidate (last)
            preceding_scores = (
                all_step_rewards[:-1] if len(all_step_rewards) > 1 else []
            )
            cand_score = all_step_rewards[-1:] if all_step_rewards else []
            log.info(
                f"Candidate {cand_idx}: preceding_scores={[f'{s:.3f}' for s in preceding_scores]}, cand_score={[f'{s:.3f}' for s in cand_score]}"
            )

        # Return only last step scores for backward compatibility
        return [[r[0]] for r in all_rewards]

    def score_trajectory(
        self, chat: List[Dict[str, str]], trajectory: List, **kwargs
    ) -> List[Optional[float]]:
        """
        Score a complete trajectory and return scores for ALL steps in a single forward pass.

        Args:
            chat: Chat messages (contains the question)
            trajectory: List of StepCandidate objects representing the full trajectory

        Returns:
            List of scores, one per step. None for steps skipped by tail truncation.
        """
        if not trajectory:
            return []

        # Extract question from chat
        question = None
        for msg in chat:
            if msg["role"] == "user":
                question = msg["content"]
                break
        if question is None:
            question = chat[-1]["content"]

        # Get all step texts
        step_texts = [
            step.text if hasattr(step, "text") else str(step) for step in trajectory
        ]

        # Build prompt with tail truncation
        prompt, num_skipped = self._truncate_steps_from_tail(
            question, step_texts, self.prm_max_tokens
        )

        # Track PRM tokens
        num_prompt_tokens = len(self.prm_tokenizer.encode(prompt))
        self._record_prm_tokens(num_prompt_tokens)

        num_included = len(step_texts) - num_skipped
        log.info(
            f"PRM scoring trajectory: {len(step_texts)} total steps, "
            f"{num_skipped} skipped, {num_included} scored, "
            f"prompt_tokens={num_prompt_tokens}, max_tokens={self.prm_max_tokens}"
        )

        # Log per-step text lengths for diagnostics
        step_char_lens = [len(s) for s in step_texts]
        log.debug(
            f"Step char lengths: {step_char_lens}, "
            f"total_chars={sum(step_char_lens)}"
        )

        # Single forward pass
        outputs = self.prm_model.reward([prompt], use_tqdm=True)

        # Extract step scores for included steps
        scored_step_scores = []
        raw_score_count = 0
        if (
            outputs
            and hasattr(outputs[0], "outputs")
            and hasattr(outputs[0].outputs, "data")
        ):
            data = outputs[0].outputs.data
            if hasattr(data, "tolist"):
                step_scores = data.tolist()
            elif isinstance(data, list):
                step_scores = data
            else:
                step_scores = [data]

            raw_score_count = len(step_scores)
            for score in step_scores:
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    scored_step_scores.append(score[1])
                else:
                    scored_step_scores.append(float(score))

        if raw_score_count != num_included:
            log.warning(
                f"PRM model returned {raw_score_count} scores but {num_included} steps were sent "
                f"(total={len(step_texts)}, skipped={num_skipped})"
            )

        log.info(
            f"PRM trajectory scores (scored only): "
            f"{[f'{s:.3f}' for s in scored_step_scores]}"
        )

        # Prepend None for skipped steps, use PRM scores as-is
        full_scores = [None] * num_skipped + scored_step_scores

        # Log final scores summary
        score_strs = [
            f"{s:.3f}" if s is not None else "null"
            for s in full_scores[: len(step_texts)]
        ]
        log.info(
            f"PRM final scores ({len(step_texts)} steps): [{', '.join(score_strs)}]"
        )

        return full_scores[: len(step_texts)]

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List],
        sample_ids: List[int] = None,
        trajectory_ids: List[int] = None,
        **kwargs,
    ) -> List[List[Optional[float]]]:
        """
        Score multiple trajectories in a single batched vLLM call.

        This is significantly faster than sequential scoring because vLLM
        can process all prompts together with continuous batching.

        Args:
            chats: List of chat messages (one per trajectory)
            trajectories: List of trajectories (each is list of steps)
            sample_ids: Optional list of sample indices for logging
            trajectory_ids: Optional list of trajectory indices within each sample

        Returns:
            List of score lists, one per trajectory
        """
        if not self.use_vllm:
            # Fall back to sequential for HuggingFace backend
            log.info("HuggingFace backend: falling back to sequential scoring")
            return [
                self.score_trajectory(chat, traj, **kwargs)
                for chat, traj in zip(chats, trajectories)
            ]

        if not trajectories:
            return []

        # Log all trajectories before scoring
        log.info(f"--- Preparing {len(trajectories)} trajectories for PRM scoring ---")
        for traj_idx, trajectory in enumerate(trajectories):
            sample_id = sample_ids[traj_idx] if sample_ids else "?"
            traj_id = trajectory_ids[traj_idx] if trajectory_ids else traj_idx
            num_steps = len(trajectory) if trajectory else 0
            log.info(f"Sample {sample_id}, Traj {traj_id}: {num_steps} steps")
            # Log each step content (full text)
            for step_idx, step in enumerate(trajectory or []):
                step_text = step.text if hasattr(step, "text") else str(step)
                log.info(f"  Step {step_idx}:\n{step_text}")

        # Build all prompts and track metadata for result mapping
        all_prompts = []
        trajectory_metadata = (
            []
        )  # (traj_idx, num_steps, sample_id, traj_id, num_skipped, num_included) for each prompt

        for traj_idx, (chat, trajectory) in enumerate(zip(chats, trajectories)):
            if not trajectory:
                trajectory_metadata.append((traj_idx, 0, None, traj_idx, 0, 0))
                all_prompts.append("")  # Placeholder
                continue

            # Extract question
            question = None
            for msg in chat:
                if msg["role"] == "user":
                    question = msg["content"]
                    break
            if question is None:
                question = chat[-1]["content"]

            # Get step texts
            step_texts = [
                step.text if hasattr(step, "text") else str(step) for step in trajectory
            ]

            # Build prompt with tail truncation
            prompt, num_skipped = self._truncate_steps_from_tail(
                question, step_texts, self.prm_max_tokens
            )

            # Track PRM tokens per sample
            num_prompt_tokens = len(self.prm_tokenizer.encode(prompt))
            sample_id = sample_ids[traj_idx] if sample_ids else None
            self._record_prm_tokens(num_prompt_tokens, sample_id=sample_id)

            all_prompts.append(prompt)
            traj_id = trajectory_ids[traj_idx] if trajectory_ids else traj_idx
            num_included = len(step_texts) - num_skipped
            trajectory_metadata.append(
                (
                    traj_idx,
                    len(step_texts),
                    sample_id,
                    traj_id,
                    num_skipped,
                    num_included,
                )
            )

        # Filter out empty prompts for scoring
        valid_indices = [i for i, p in enumerate(all_prompts) if p]
        valid_prompts = [all_prompts[i] for i in valid_indices]

        # Log truncation summary table
        num_truncated = sum(1 for _, _, _, _, ns, _ in trajectory_metadata if ns > 0)
        total_prompt_tokens = sum(
            len(self.prm_tokenizer.encode(p)) for p in valid_prompts
        )
        prompt_token_counts = [len(self.prm_tokenizer.encode(p)) for p in valid_prompts]
        log.info(
            f"--- PRM Batch Prompt Summary ---\n"
            f"  Trajectories: {len(trajectories)} total, {len(valid_prompts)} valid, "
            f"{len(trajectories) - len(valid_prompts)} empty\n"
            f"  Truncated: {num_truncated}/{len(valid_prompts)} trajectories had steps skipped\n"
            f"  Prompt tokens: total={total_prompt_tokens}, "
            f"min={min(prompt_token_counts) if prompt_token_counts else 0}, "
            f"max={max(prompt_token_counts) if prompt_token_counts else 0}, "
            f"mean={total_prompt_tokens / len(prompt_token_counts) if prompt_token_counts else 0:.0f}\n"
            f"  Token limit: {self.prm_max_tokens}"
        )
        # Per-trajectory detail at debug level
        for vi, pi in enumerate(valid_indices):
            meta = trajectory_metadata[pi]
            traj_idx, num_steps, sample_id, traj_id, num_skipped, num_included = meta
            log.debug(
                f"  Traj {traj_id} (sample={sample_id}): {num_steps} steps, "
                f"skipped={num_skipped}, included={num_included}, "
                f"prompt_tokens={prompt_token_counts[vi]}"
            )

        log.info(f"PRM batch scoring {len(valid_prompts)} trajectories in single call")

        # Single batched vLLM call
        if valid_prompts:
            outputs = self.prm_model.reward(valid_prompts, use_tqdm=True)
        else:
            outputs = []

        # Parse results and map back to trajectories
        log.info("--- PRM Scoring Results ---")
        results = [[] for _ in trajectories]
        output_idx = 0

        for i, (
            traj_idx,
            num_steps,
            sample_id,
            traj_id,
            num_skipped,
            num_included,
        ) in enumerate(trajectory_metadata):
            if i not in valid_indices:
                results[traj_idx] = []
                continue

            output = outputs[output_idx]
            output_idx += 1

            scored_step_scores = []
            raw_score_count = 0
            if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
                data = output.outputs.data
                if hasattr(data, "tolist"):
                    raw_scores = data.tolist()
                elif isinstance(data, list):
                    raw_scores = data
                else:
                    raw_scores = [data]

                raw_score_count = len(raw_scores)
                for score in raw_scores:
                    if isinstance(score, (list, tuple)) and len(score) == 2:
                        scored_step_scores.append(
                            score[1]
                        )  # Positive class probability
                    else:
                        scored_step_scores.append(float(score))

            if raw_score_count != num_included:
                log.warning(
                    f"Traj {traj_id} (sample={sample_id}): PRM model returned "
                    f"{raw_score_count} scores but {num_included} steps were sent "
                    f"(num_steps={num_steps}, skipped={num_skipped})"
                )

            # Prepend None for skipped steps, use PRM scores as-is
            final_scores = [None] * num_skipped + scored_step_scores
            results[traj_idx] = final_scores

            # Log detailed scores for this trajectory
            sample_str = f"Sample {sample_id}" if sample_id is not None else ""
            score_strs = [f"{s:.3f}" if s is not None else "null" for s in final_scores]
            log.info(
                f"{sample_str} Traj {traj_id}: {num_steps} steps "
                f"({num_skipped} skipped, {num_included} scored, "
                f"model_returned={raw_score_count}), "
                f"scores=[{', '.join(score_strs)}]"
            )

        # Batch scoring summary
        total_steps = sum(len(r) for r in results)
        total_nulls = sum(1 for r in results for s in r if s is None)
        total_scored = total_steps - total_nulls
        all_scored_values = [s for r in results for s in r if s is not None]
        if all_scored_values:
            log.info(
                f"--- PRM Batch Scoring Summary ---\n"
                f"  Trajectories scored: {len(results)}\n"
                f"  Total steps: {total_steps} ({total_scored} scored, {total_nulls} skipped/null)\n"
                f"  Score stats: min={min(all_scored_values):.3f}, "
                f"max={max(all_scored_values):.3f}, "
                f"mean={statistics.mean(all_scored_values):.3f}, "
                f"median={statistics.median(all_scored_values):.3f}"
            )
        else:
            log.info(
                f"PRM batch scoring complete: {len(results)} trajectories, no scores produced"
            )
        return results

    def _format_prm_prompt(self, question: str, step_texts: List[str]) -> str:
        """Format a prompt for PRM scoring with <extra_0> step separators."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Please reason step by step, and put your final "
                    "answer within \\boxed{}."
                ),
            },
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "<extra_0>".join(step_texts) + "<extra_0>",
            },
        ]
        return self.prm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _truncate_steps_from_tail(
        self, question: str, step_texts: List[str], max_tokens: int
    ) -> tuple:
        """Build PRM prompt keeping as many steps from the tail as fit.

        Walks backward from the last step, accumulating token costs, and
        includes only the steps that fit within the budget. Earlier steps
        that don't fit are dropped.

        Args:
            question: The question text.
            step_texts: All step texts (trajectory steps + candidate).
            max_tokens: Maximum token budget for the prompt.

        Returns:
            (prompt, num_skipped): formatted prompt string and count of
            dropped leading steps.
        """
        # Compute frame overhead (system + user + assistant wrapper without steps)
        frame_prompt = self._format_prm_prompt(question, [])
        frame_tokens = (
            len(self.prm_tokenizer.encode(frame_prompt)) - 1
        )  # subtract trailing <extra_0>

        # Per-step token costs (step text tokens + 1 for <extra_0> separator)
        step_costs = []
        for step_text in step_texts:
            step_token_count = len(self.prm_tokenizer.encode(step_text))
            step_costs.append(step_token_count + 1)  # +1 for <extra_0>

        total_step_tokens = sum(step_costs)
        budget = max_tokens - frame_tokens - 10  # safety margin

        log.debug(
            f"Tail truncation budget: max_tokens={max_tokens}, "
            f"frame_tokens={frame_tokens + 1} (adjusted={frame_tokens}), "
            f"budget={budget}, total_step_tokens={total_step_tokens}, "
            f"num_steps={len(step_texts)}, "
            f"fits_without_truncation={total_step_tokens <= budget}"
        )

        # Walk from last step backward, accumulating costs
        total_cost = 0
        first_included_idx = len(step_texts)  # will move backward
        for i in range(len(step_texts) - 1, -1, -1):
            if total_cost + step_costs[i] <= budget:
                total_cost += step_costs[i]
                first_included_idx = i
            else:
                break

        # Edge case: if even the last step alone exceeds budget, force-include it
        if first_included_idx == len(step_texts) and step_texts:
            first_included_idx = len(step_texts) - 1
            log.warning(
                f"Last step alone ({step_costs[-1]} tokens) exceeds budget "
                f"({budget} tokens), force-including it"
            )

        num_skipped = first_included_idx
        included_steps = step_texts[first_included_idx:]

        if num_skipped > 0:
            skipped_tokens = sum(step_costs[:num_skipped])
            included_tokens = sum(step_costs[num_skipped:])
            log.warning(
                f"Tail truncation: skipped {num_skipped}/{len(step_texts)} leading steps "
                f"({skipped_tokens} tokens dropped, {included_tokens} tokens kept) "
                f"to fit within {max_tokens} token limit (budget={budget})"
            )

        prompt = self._format_prm_prompt(question, included_steps)
        prompt_tokens = self.prm_tokenizer.encode(prompt)
        actual_prompt_tokens = len(prompt_tokens)

        # Hard-truncate safety net: if the final prompt still exceeds the model's
        # max_position_embeddings (4096), truncate token sequence directly.
        # This handles BPE boundary effects and the force-include edge case.
        model_max_len = 4096
        if actual_prompt_tokens > model_max_len:
            log.warning(
                f"Hard-truncating prompt from {actual_prompt_tokens} to {model_max_len} tokens "
                f"(exceeded model max_position_embeddings)"
            )
            prompt_tokens = prompt_tokens[:model_max_len]
            prompt = self.prm_tokenizer.decode(prompt_tokens)
            actual_prompt_tokens = model_max_len

        if num_skipped > 0 or actual_prompt_tokens > max_tokens:
            log.info(
                f"Tail truncation result: {len(included_steps)} steps included, "
                f"actual_prompt_tokens={actual_prompt_tokens}, limit={max_tokens}"
            )
        return prompt, num_skipped

    def _score_single_candidate_hf(
        self, chat: List[Dict[str, str]], candidate: str
    ) -> List[float]:
        """Score a single candidate using HuggingFace PRM backend."""
        # Extract claims from candidate
        candidate_tokens = self.prm_tokenizer(candidate, return_tensors="pt")

        claims = self.steps_extractor.split_to_steps(
            candidate, candidate_tokens["input_ids"][0], self.prm_tokenizer
        )

        if not claims:
            log.debug(f"No claims extracted from candidate: {candidate[:50]}...")
            return [0.0]

        # Get PRM rewards
        rewards = self._compute_prm_rewards_hf(chat, claims)
        log.info(f"PRM rewards for {len(claims)} claims: {rewards}")
        return rewards if rewards else [0.0]

    def _compute_prm_rewards_hf(
        self, chat: List[Dict[str, str]], claims: List[Any]
    ) -> List[float]:
        """Compute PRM rewards for claims using HuggingFace backend.

        TODO: Apply tail truncation (_truncate_steps_from_tail) here too.
        """
        if not claims:
            return []

        # Format conversation for PRM
        question = chat[-1]["content"]
        log.debug(f"Question: {question[:100]}...")
        messages = [
            {
                "role": "system",
                "content": (
                    "Please reason step by step, and put your final "
                    "answer within \\boxed{}."
                ),
            },
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "<extra_0>".join([c.claim_text for c in claims])
                + "<extra_0>",
            },
        ]

        conversation_str = self.prm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        input_ids = self.prm_tokenizer.encode(conversation_str, return_tensors="pt").to(
            self.prm_model.device
        )

        # Get model outputs (disable cache to avoid version compatibility issues)
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids, use_cache=False)

        # Extract step rewards
        step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        token_masks = input_ids == step_sep_id

        # Compute rewards
        rewards = self._extract_step_rewards_hf(outputs[0], token_masks)

        return rewards[0] if rewards else []

    def _extract_step_rewards_hf(self, logits, token_masks):
        """Extract reward scores from PRM logits (HuggingFace backend)."""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)

        all_scores = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            # Get positive class probabilities where mask is non-zero
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            scores = positive_probs.cpu().tolist()
            all_scores.append(scores)

        return all_scores

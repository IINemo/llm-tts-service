"""
Direct PRM scorer that bypasses the stat calculator pipeline for efficient stepwise scoring.

Supports both HuggingFace and vLLM backends for the PRM model.
"""

import logging
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
    ):
        self.prm_model_path = prm_model_path
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_vllm = use_vllm and VLLM_AVAILABLE
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
        all_prompts = []
        for candidate in candidates:
            candidate_text = candidate.text if hasattr(candidate, "text") else candidate
            # Each trajectory step is one PRM step, candidate is one PRM step
            all_steps = trajectory_steps + [candidate_text]
            prompt = self._format_prm_prompt(question, all_steps)
            all_prompts.append(prompt)

        # Truncate prompts that exceed max length
        max_tokens = 4000
        truncated_prompts = []
        total_prompt_tokens = 0
        for prompt in all_prompts:
            tokens = self.prm_tokenizer.encode(prompt)
            total_prompt_tokens += len(tokens)
            if len(tokens) > max_tokens:
                log.warning(
                    f"Truncating PRM prompt from {len(tokens)} to {max_tokens} tokens"
                )
                tokens = tokens[:max_tokens]
                prompt = self.prm_tokenizer.decode(tokens)
            truncated_prompts.append(prompt)

        # Track PRM tokens
        self._record_prm_tokens(total_prompt_tokens)

        # Batch score all prompts
        num_traj_steps = len(trajectory_steps)
        log.info(
            f"PRM scoring {len(truncated_prompts)} candidates (trajectory has {num_traj_steps} steps)"
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
            # Log all step scores: trajectory steps + candidate step
            traj_scores = (
                all_step_rewards[:num_traj_steps] if num_traj_steps > 0 else []
            )
            cand_score = (
                all_step_rewards[num_traj_steps:]
                if len(all_step_rewards) > num_traj_steps
                else all_step_rewards
            )
            log.info(
                f"Candidate {cand_idx}: traj_scores={[f'{s:.3f}' for s in traj_scores]}, cand_score={[f'{s:.3f}' for s in cand_score]}"
            )

        # Return only last step scores for backward compatibility
        return [[r[0]] for r in all_rewards]

    def score_trajectory(
        self, chat: List[Dict[str, str]], trajectory: List, **kwargs
    ) -> List[float]:
        """
        Score a complete trajectory and return scores for ALL steps in a single forward pass.

        Args:
            chat: Chat messages (contains the question)
            trajectory: List of StepCandidate objects representing the full trajectory

        Returns:
            List of scores, one per step in the trajectory
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

        # Build single prompt with all steps
        prompt = self._format_prm_prompt(question, step_texts)

        # Truncate if needed
        max_tokens = 4000
        tokens = self.prm_tokenizer.encode(prompt)
        num_prompt_tokens = len(tokens)
        if len(tokens) > max_tokens:
            log.warning(
                f"Truncating PRM prompt from {len(tokens)} to {max_tokens} tokens"
            )
            tokens = tokens[:max_tokens]
            prompt = self.prm_tokenizer.decode(tokens)

        # Track PRM tokens
        self._record_prm_tokens(num_prompt_tokens)

        log.info(f"PRM scoring trajectory with {len(step_texts)} steps")

        # Single forward pass
        outputs = self.prm_model.reward([prompt], use_tqdm=True)

        # Extract all step scores
        all_step_scores = []
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

            for score in step_scores:
                if isinstance(score, (list, tuple)) and len(score) == 2:
                    all_step_scores.append(score[1])
                else:
                    all_step_scores.append(float(score))

        log.info(f"PRM trajectory scores: {[f'{s:.3f}' for s in all_step_scores]}")

        while len(all_step_scores) < len(step_texts):
            all_step_scores.append(0.0)

        return all_step_scores[: len(step_texts)]

    def score_trajectories_batch(
        self,
        chats: List[List[Dict[str, str]]],
        trajectories: List[List],
        sample_ids: List[int] = None,
        trajectory_ids: List[int] = None,
        **kwargs,
    ) -> List[List[float]]:
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
        )  # (traj_idx, num_steps, sample_id, traj_id) for each prompt

        for traj_idx, (chat, trajectory) in enumerate(zip(chats, trajectories)):
            if not trajectory:
                trajectory_metadata.append((traj_idx, 0))
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

            # Build prompt
            prompt = self._format_prm_prompt(question, step_texts)

            # Truncate if needed
            max_tokens = 4000
            tokens = self.prm_tokenizer.encode(prompt)
            num_prompt_tokens = len(tokens)
            if len(tokens) > max_tokens:
                log.warning(
                    f"Truncating PRM prompt {traj_idx} from {len(tokens)} to {max_tokens} tokens"
                )
                tokens = tokens[:max_tokens]
                prompt = self.prm_tokenizer.decode(tokens)

            # Track PRM tokens per sample
            sample_id = sample_ids[traj_idx] if sample_ids else None
            self._record_prm_tokens(num_prompt_tokens, sample_id=sample_id)

            all_prompts.append(prompt)
            traj_id = trajectory_ids[traj_idx] if trajectory_ids else traj_idx
            trajectory_metadata.append((traj_idx, len(step_texts), sample_id, traj_id))

        # Filter out empty prompts for scoring
        valid_indices = [i for i, p in enumerate(all_prompts) if p]
        valid_prompts = [all_prompts[i] for i in valid_indices]

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

        for i, (traj_idx, num_steps, sample_id, traj_id) in enumerate(
            trajectory_metadata
        ):
            if i not in valid_indices:
                results[traj_idx] = []
                continue

            output = outputs[output_idx]
            output_idx += 1

            step_scores = []
            if hasattr(output, "outputs") and hasattr(output.outputs, "data"):
                data = output.outputs.data
                if hasattr(data, "tolist"):
                    raw_scores = data.tolist()
                elif isinstance(data, list):
                    raw_scores = data
                else:
                    raw_scores = [data]

                for score in raw_scores:
                    if isinstance(score, (list, tuple)) and len(score) == 2:
                        step_scores.append(score[1])  # Positive class probability
                    else:
                        step_scores.append(float(score))

            # Pad if needed
            while len(step_scores) < num_steps:
                step_scores.append(0.0)

            final_scores = step_scores[:num_steps]
            results[traj_idx] = final_scores

            # Log detailed scores for this trajectory
            sample_str = f"Sample {sample_id}" if sample_id is not None else ""
            log.info(
                f"{sample_str} Traj {traj_id}: {num_steps} steps, "
                f"scores={[f'{s:.3f}' for s in final_scores]}"
            )

        log.info(f"PRM batch scoring complete: {len(results)} trajectories scored")
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
        """Compute PRM rewards for claims using HuggingFace backend."""
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

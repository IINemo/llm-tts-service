"""
Direct PRM scorer that bypasses the stat calculator pipeline for efficient stepwise scoring
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.stat_calculators.extract_claims import Claim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from llm_tts.utils import get_torch_dtype

from .step_scorer_reward_base import StepScorerRewardBase

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

    Much cleaner and more efficient than going through the full pipeline.
    """

    def __init__(
        self, prm_model_path: str, device: str, batch_size: int, torch_dtype: str
    ):
        self.prm_model_path = prm_model_path
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.prm_model = None
        self.prm_tokenizer = None
        self.steps_extractor = StepsExtractor(progress_bar=False)

        self.prepare_model()

    def prepare_model(self):
        """Load PRM model and tokenizer"""

        log.info(f"Loading PRM model from {self.prm_model_path}")
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
        """Free PRM model memory"""

        if self.prm_model is not None:
            del self.prm_model
            self.prm_model = None
            del self.prm_tokenizer
            self.prm_tokenizer = None
            torch.cuda.empty_cache()

    def compute_claim_rewards(
        self, chat: List[Dict[str, str]], candidates: List[str], **kwargs
    ) -> List[List[float]]:
        """
        Compute reward scores for claims in each candidate.

        Args:
            chat: Current chat
            candidates: List of candidate next steps

        Returns:
            List of claim reward lists (one per candidate)
        """

        if not candidates:
            return []

        # Score all candidates
        all_rewards = []

        for candidate in candidates:
            rewards = self._score_single_candidate(chat, candidate)
            all_rewards.append(rewards)

            # Clean up memory after each candidate
            torch.cuda.empty_cache()

        return all_rewards

    def _score_single_candidate(
        self, chat: List[Dict[str, str]], candidate: str
    ) -> List[float]:
        """Score a single candidate using PRM"""

        # Extract claims from candidate
        candidate_tokens = self.prm_tokenizer(candidate.text, return_tensors="pt")

        claims = self.steps_extractor.split_to_steps(
            candidate.text, candidate_tokens["input_ids"][0], self.prm_tokenizer
        )

        if not claims:
            log.debug(f"No claims extracted from candidate: {candidate.text[:50]}...")
            return [0.0]

        # Get PRM rewards
        rewards = self._compute_prm_rewards(chat, claims)
        return rewards if rewards else [0.0]

    def _compute_prm_rewards(
        self, chat: List[Dict[str, str]], claims: List[Any]
    ) -> List[float]:
        """Compute PRM rewards for claims"""

        if not claims:
            return []

        # Format conversation for PRM
        question = chat[-1]["content"]
        log.info(f"Question: {question}")
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

        # Get model outputs
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids)

        # Extract step rewards
        step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        token_masks = input_ids == step_sep_id

        # Compute rewards
        rewards = self._extract_step_rewards(outputs[0], token_masks)

        return rewards[0] if rewards else []

    def _extract_step_rewards(self, logits, token_masks):
        """Extract reward scores from PRM logits"""

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

"""Adapter to add token-level logprob generation to local WhiteboxModel instances.

This wrapper delegates attribute access to the wrapped WhiteboxModel so the
rest of the code (generators, tokenizers, generation parameters) continue to
work, but also exposes `supports_logprobs = True` and a
`generate_with_logprobs(request, ...)` method expected by CoT-UQ.

The implementation uses the underlying transformers `AutoModelForCausalLM`
(`base_model`) and `tokenizer` to run `generate(..., output_scores=True,
return_dict_in_generate=True)` and extracts per-token probabilities from the
returned `scores` logits.
"""
from types import SimpleNamespace
import inspect
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict

log = logging.getLogger(__name__)


class LocalWhiteboxLogprobAdapter:
    """Wrap a WhiteboxModel to provide `generate_with_logprobs` and advertise
    `supports_logprobs = True` while delegating other attributes.

    Args:
        wb_model: the WhiteboxModel instance (kept for delegation)
        base_model: underlying transformers model (AutoModelForCausalLM)
        tokenizer: corresponding tokenizer
        device: device string or torch.device where tensors should be placed
    """

    def __init__(self, wb_model, base_model, tokenizer, device="cpu", disable_thinking_mode: bool = False):
        self._wb = wb_model
        self._base = base_model
        self.tokenizer = tokenizer
        self.device_str = device
        self.supports_logprobs = True
        self.disable_thinking_mode = disable_thinking_mode

    def __getattr__(self, name):
        # Delegate unknown attributes to the wrapped WhiteboxModel
        return getattr(self._wb, name)

    def _build_prompt(self, request: List[Dict[str, str]]) -> str:
        # Prefer tokenizer's chat template helper if available
        try:
            if hasattr(self._wb.tokenizer, "apply_chat_template"):
                sig = inspect.signature(self._wb.tokenizer.apply_chat_template)
                has_enable_thinking = "enable_thinking" in sig.parameters

                inputs = self._wb.tokenizer.apply_chat_template(
                    [request],
                    tokenize=False,
                    add_generation_prompt=True,
                    **(
                        {"enable_thinking": (not self.disable_thinking_mode)}
                        if has_enable_thinking
                        else {}
                    ),
                )
                if isinstance(inputs, list):
                    return inputs[0]
                prompt = inputs
                if self.disable_thinking_mode and not has_enable_thinking:
                    # For tokenizers without `enable_thinking`, explicitly close any
                    # thinking block to discourage thought emissions.
                    prompt += "\n<think>\n\n</think>\n\n"
                return prompt
        except Exception:
            # Fall back to concatenating contents
            log.debug("tokenizer.apply_chat_template failed, falling back")

        return "".join([m.get("content", "") for m in request])

    def generate_with_logprobs(
        self,
        request: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 128,
        top_logprobs: int = 10,
    ) -> SimpleNamespace:
        """Generate a response and return an object with `.text` and
        `.token_probs` (list of (token_text, prob)).

        Note: this implementation computes the probability of each generated
        token by applying softmax to the logits produced at each generation
        step and selecting the probability corresponding to the generated
        token id. This returns full-vocab probabilities (may be slower), but
        is robust and deterministic.
        """
        prompt = self._build_prompt(request)

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = {k: v.to(self._base.device) for k, v in inputs.items()}

        gen_params = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        with torch.no_grad():
            outputs = self._base.generate(**inputs, **gen_params)

        # Decode generated text
        # outputs.sequences includes the input prompt tokens + generated tokens
        seq = outputs.sequences[0]

        # Determine where generated tokens start
        input_len = inputs["input_ids"].shape[1]
        gen_ids = seq[input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        token_probs: List[Tuple[str, float]] = []

        # Extract per-step logits (list length == num_generated_tokens), each
        # element is (batch_size, vocab_size). We compute softmax and get the
        # probability of the generated token id at that step.
        try:
            scores = outputs.scores  # list[Tensor(batch, vocab)]
            if scores is None:
                scores = []
        except Exception:
            scores = []

        if scores:
            # For each generated token i, corresponding logits are scores[i]
            for i, token_id in enumerate(gen_ids.tolist()):
                if i < len(scores):
                    logits = scores[i][0]
                    probs = F.softmax(logits, dim=-1)
                    prob = float(probs[token_id].cpu().item())
                    token_text = self.tokenizer.decode([token_id])
                    token_probs.append((token_text, prob))
                else:
                    # Missing score for this token; append None prob
                    token_text = self.tokenizer.decode([token_id])
                    token_probs.append((token_text, None))
        else:
            # No scores available; return empty token_probs
            token_probs = []

        return SimpleNamespace(text=text, token_probs=token_probs)

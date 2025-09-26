import os
import math
import time
import logging
from typing import List, Optional, Dict, Any

from together import Together


log = logging.getLogger(__name__)


class TogetherChat:
    """
    Together.ai chat client wrapper supporting logprobs for uncertainty estimation.

    Supports:
      - batched generation (n return sequences)
      - temperature control
      - logprobs with top_k for PD uncertainty estimation (up to 20)
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        wait_times: tuple = (5, 10, 20, 40, 80),
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not provided")
        self.wait_times = wait_times
        self.client = Together(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        logprobs: Optional[int] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Generate completions using Together.ai API"""
        completion_args = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "n": n,
            "max_tokens": max_new_tokens,
        }
        
        if logprobs is not None:
            completion_args["logprobs"] = min(max(logprobs, 0), 20)
            completion_args["top_logprobs"] = min(max(logprobs, 0), 20)
        if stop:
            completion_args["stop"] = stop

        try:
            return self.client.completions.create(**completion_args)
        except Exception as e:
            raise RuntimeError(f"Together.ai request failed: {e}") from e

    def generate_texts(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate text completions and return as list of strings"""
        resp = self.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            logprobs=None,
            stop=stop,
        )
        texts = []
        for choice in resp.choices:
            texts.append(choice.text or "")
        return texts

    def generate_one_with_top_logprobs(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate exactly one token and return its text and a mapping of top-K alternatives to probabilities.
        Returns {"text": str, "top_logprobs": {token: prob, ...}}
        """
        # Prefer chat.completions for reliable top_logprobs
        k = min(max(top_k, 0), 5)
        try:
            chat_resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=temperature,
                logprobs=k,
                top_logprobs=k,
            )
            choice = chat_resp.choices[0]
            text = choice.message.content or ""
            logprobs_data = getattr(choice, "logprobs", None)
            if not logprobs_data:
                raise RuntimeError("chat.completions missing logprobs")
        except Exception:
            # Fallback to completions API
            resp = self.generate(
                prompt=prompt,
                n=1,
                temperature=temperature,
                max_new_tokens=1,
                logprobs=k,
            )
            choice = resp.choices[0]
            text = choice.text or ""
            logprobs_data = getattr(choice, "logprobs", None)
            if not logprobs_data:
                raise RuntimeError("Together.ai response missing logprobs data")

        token_to_prob: Dict[str, float] = {}
        try:
            top_logprobs = getattr(logprobs_data, 'top_logprobs', None)
            if top_logprobs:
                last_top = top_logprobs[-1]
                if isinstance(last_top, dict):
                    pairs = list(last_top.items())
                    try:
                        pairs.sort(key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
                    except Exception:
                        pass
                    for tok, lp in pairs[:k]:
                        if lp is not None:
                            token_to_prob[tok] = math.exp(lp)
                elif isinstance(last_top, list):
                    try:
                        sorted_top = sorted(
                            last_top,
                            key=lambda e: (e.get('logprob') if isinstance(e, dict) else getattr(e, 'logprob', None)) or -1e9,
                            reverse=True,
                        )
                    except Exception:
                        sorted_top = last_top
                    for entry in sorted_top[:k]:
                        if isinstance(entry, dict):
                            tok = entry.get('token')
                            lp = entry.get('logprob')
                        else:
                            tok = getattr(entry, 'token', None)
                            lp = getattr(entry, 'logprob', None)
                        if tok is not None and lp is not None:
                            token_to_prob[tok] = math.exp(lp)
            if not token_to_prob and hasattr(logprobs_data, 'tokens') and hasattr(logprobs_data, 'token_logprobs'):
                last_tok = logprobs_data.tokens[-1]
                last_lp = logprobs_data.token_logprobs[-1]
                if last_lp is not None:
                    token_to_prob[last_tok] = math.exp(last_lp)
        except Exception as e:
            log.warning(f"Failed to parse top_logprobs for single-token gen: {e}")

        if not token_to_prob:
            raise RuntimeError("Together.ai returned no usable logprob data for single token")
        return {"text": text, "top_logprobs": token_to_prob}

    def generate_texts_with_logprobs(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        top_k: int = 5,
        stop: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate sequences and return per-token logprob details for confidence scoring.

        Returns a list of dicts, each containing:
          - text: the completion text
          - tokens: list[str]
          - token_logprobs: list[float]
          - top_logprobs: list[Union[dict, list]] (as provided by API)
        """
        k = min(max(top_k, 0), 20)
        resp = self.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            logprobs=k,
            stop=stop,
        )
        outputs: List[Dict[str, Any]] = []
        for choice in resp.choices:
            item: Dict[str, Any] = {"text": choice.text or ""}
            lp = getattr(choice, 'logprobs', None)
            if lp is not None:
                item["tokens"] = getattr(lp, 'tokens', []) or []
                item["token_logprobs"] = getattr(lp, 'token_logprobs', []) or []
                item["top_logprobs"] = getattr(lp, 'top_logprobs', []) or []
            else:
                item["tokens"] = []
                item["token_logprobs"] = []
                item["top_logprobs"] = []
            outputs.append(item)
        return outputs


class TogetherChatCompat:
    """
    Compatibility wrapper to match OpenRouter interface for easy replacement.
    """
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.together = TogetherChat(model=model, api_key=api_key)
    
    def generate_texts(self, prompt: str, n: int = 1, temperature: float = 0.7, max_new_tokens: int = 128, stop: Optional[List[str]] = None) -> List[str]:
        return self.together.generate_texts(prompt, n, temperature, max_new_tokens, stop)

    def generate_one_with_top_logprobs(self, prompt: str, temperature: float = 0.0, top_k: int = 5) -> Dict[str, Any]:
        return self.together.generate_one_with_top_logprobs(prompt, temperature, top_k)

    def generate_texts_with_logprobs(self, prompt: str, n: int = 1, temperature: float = 0.7, max_new_tokens: int = 128, top_k: int = 5, stop: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.together.generate_texts_with_logprobs(prompt, n, temperature, max_new_tokens, top_k, stop)

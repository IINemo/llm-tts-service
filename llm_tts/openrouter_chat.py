import os
import math
import time
import logging
from typing import List, Optional, Dict, Any

import openai


log = logging.getLogger(__name__)


class OpenRouterChat:
    """
    Minimal OpenRouter chat client wrapper using OpenAI-compatible SDK.

    Supports:
      - batched generation (n return sequences)
      - temperature control
      - logprobs with top_logprobs for PD uncertainty estimation
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: str = "https://openrouter.ai/api/v1",
        wait_times: tuple = (5, 10, 20, 40, 80),
        use_chat: Optional[bool] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not provided")
        self.api_base = api_base
        self.wait_times = wait_times
        # If None, will auto-detect based on prompt content
        self.use_chat = use_chat

    def _client(self):
        return openai.OpenAI(base_url=self.api_base, api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        logprobs: bool = False,
        top_logprobs: int = 0,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
    ):
        # Decide endpoint: explicit flag or auto-detect ChatML markers
        def _looks_like_chat(s: str) -> bool:
            if s is None:
                return False
            low = s.lower()
            return ("<|im_start|>" in low) or ("<|assistant|>" in low) or ("<|user|>" in low)

        use_chat = self.use_chat if self.use_chat is not None else _looks_like_chat(prompt)

        if use_chat:
            chat_args = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "n": n,
                "max_tokens": max_new_tokens,
            }
            if logprobs:
                chat_args["logprobs"] = True
                if top_logprobs:
                    chat_args["top_logprobs"] = max(1, min(int(top_logprobs), 20))
            if response_format is not None:
                chat_args["response_format"] = response_format
            if stop:
                chat_args["stop"] = stop

            last_exc = None
            for i, wt in enumerate(self.wait_times):
                try:
                    return self._client().chat.completions.create(**chat_args)
                except Exception as e:
                    last_exc = e
                    log.warning(f"OpenRouter request failed (attempt {i+1}): {e}")
                    time.sleep(wt)
            try:
                return self._client().chat.completions.create(**chat_args)
            except Exception as e:
                raise RuntimeError(f"OpenRouter request failed: {e}") from last_exc
        else:
            completion_args = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "n": n,
                "max_tokens": max_new_tokens,
            }
            if logprobs:
                # Completions API expects 'logprobs' to be an integer: number of alternatives.
                k = 5
                try:
                    if top_logprobs:
                        k = max(1, min(int(top_logprobs), 5))
                except Exception:
                    k = 5
                completion_args["logprobs"] = k
            if stop:
                completion_args["stop"] = stop

            last_exc = None
            for i, wt in enumerate(self.wait_times):
                try:
                    return self._client().completions.create(**completion_args)
                except Exception as e:
                    last_exc = e
                    log.warning(f"OpenRouter request failed (attempt {i+1}): {e}")
                    time.sleep(wt)
            try:
                return self._client().completions.create(**completion_args)
            except Exception as e:
                raise RuntimeError(f"OpenRouter request failed: {e}") from last_exc

    def generate_texts(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_new_tokens: int = 128,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        resp = self.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            logprobs=False,
            stop=stop,
        )
        texts = []
        for ch in resp.choices:
            # Prefer base completions field
            text_val = getattr(ch, 'text', None)
            if not text_val:
                # Fallback to chat-style message content if provided
                content = getattr(getattr(ch, 'message', None), 'content', None)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(str(part.get('text') or part.get('content') or ''))
                        else:
                            parts.append(str(part))
                    text_val = ''.join(parts)
                else:
                    text_val = content if isinstance(content, str) else ''
            texts.append(text_val)
        return texts

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
        Generate sequences with per-token logprob details suitable for PD uncertainty.

        Returns a list of dicts, each containing:
          - text: str
          - tokens: List[str]
          - token_logprobs: List[float]
          - top_logprobs: List[List[Dict[str, Any]]]  # per-position alternatives
        """
        k = max(0, min(int(top_k), 20))
        resp = self.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            logprobs=True,
            top_logprobs=k,
            stop=stop,
        )

        outputs: List[Dict[str, Any]] = []
        for choice in resp.choices:
            # Prefer base completions text
            text = getattr(choice, 'text', None) or ""
            if not text:
                # Fallback to chat-style if present
                msg = getattr(choice, 'message', None)
                content = getattr(msg, 'content', None)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            parts.append(str(part.get('text') or part.get('content') or ''))
                        else:
                            parts.append(str(part))
                    text = ''.join(parts)
                elif isinstance(content, str):
                    text = content
            item: Dict[str, Any] = {
                "text": text,
                "tokens": [],
                "token_logprobs": [],
                "top_logprobs": [],
            }

            # Prefer completion-style logprobs
            lp = getattr(choice, "logprobs", None)
            tokens = getattr(lp, 'tokens', None) if lp is not None else None
            token_lps = getattr(lp, 'token_logprobs', None) if lp is not None else None
            top_lps = getattr(lp, 'top_logprobs', None) if lp is not None else None
            if isinstance(tokens, list) and isinstance(token_lps, list):
                item["tokens"] = list(tokens)
                item["token_logprobs"] = list(token_lps)
                if isinstance(top_lps, list):
                    norm_positions = []
                    for pos in top_lps:
                        if isinstance(pos, dict):
                            pairs = list(pos.items())
                            try:
                                pairs.sort(key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
                            except Exception:
                                pass
                            norm_positions.append([
                                {"token": t, "logprob": lpv} for t, lpv in (pairs[:k] if k > 0 else pairs)
                            ])
                        elif isinstance(pos, list):
                            norm_positions.append([
                                {"token": getattr(e, 'token', None) if not isinstance(e, dict) else e.get('token'),
                                 "logprob": getattr(e, 'logprob', None) if not isinstance(e, dict) else e.get('logprob')}
                                for e in (pos[:k] if k > 0 else pos)
                            ])
                        else:
                            norm_positions.append([])
                    item["top_logprobs"] = norm_positions
            else:
                # Fallback to chat-style logprobs.content if present
                content_lp = getattr(lp, "content", None) if lp is not None else None
                if isinstance(content_lp, list):
                    for token_entry in content_lp:
                        token_str = token_entry.get("token") if isinstance(token_entry, dict) else None
                        token_lp = token_entry.get("logprob") if isinstance(token_entry, dict) else None
                        if token_str is not None:
                            item["tokens"].append(token_str)
                            item["token_logprobs"].append(token_lp if token_lp is not None else None)
                            tlp = token_entry.get("top_logprobs") if isinstance(token_entry, dict) else None
                            if isinstance(tlp, list):
                                norm = []
                                for alt in tlp[:k] if k > 0 else tlp:
                                    if isinstance(alt, dict):
                                        norm.append({
                                            "token": alt.get("token"),
                                            "logprob": alt.get("logprob"),
                                        })
                                item["top_logprobs"].append(norm)
                            else:
                                item["top_logprobs"].append([])
            outputs.append(item)

        return outputs

    def generate_one_with_top_logprobs(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate exactly one token and return its text and a mapping of top-K
        alternatives to probabilities: {"text": str, "top_logprobs": {token: prob}}
        """
        k = max(0, min(int(top_k), 20))
        resp = self.generate(
            prompt=prompt,
            n=1,
            temperature=temperature,
            max_new_tokens=1,
            logprobs=True,
            top_logprobs=k,
        )
        choice = resp.choices[0]
        # Prefer completions text
        text = getattr(choice, 'text', None) or ""
        if not text:
            # Fallback to chat-style
            msg = getattr(choice, 'message', None)
            content = getattr(msg, 'content', None)
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get('text') or part.get('content') or ''))
                    else:
                        parts.append(str(part))
                text = ''.join(parts)
            elif isinstance(content, str):
                text = content
        token_to_prob: Dict[str, float] = {}

        lp = getattr(choice, "logprobs", None)
        # Completions-style: use tokens/token_logprobs last position
        tokens = getattr(lp, 'tokens', None)
        token_lps = getattr(lp, 'token_logprobs', None)
        top_lps = getattr(lp, 'top_logprobs', None)
        if isinstance(top_lps, list) and len(top_lps) > 0:
            last_top = top_lps[-1]
            if isinstance(last_top, dict):
                pairs = list(last_top.items())
                try:
                    pairs.sort(key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
                except Exception:
                    pass
                for t, lpv in pairs[:k] if k > 0 else pairs:
                    if t is not None and lpv is not None:
                        try:
                            token_to_prob[str(t)] = math.exp(lpv)
                        except Exception:
                            token_to_prob[str(t)] = float(lpv) if isinstance(lpv, (int, float)) else 0.0
        if not token_to_prob and isinstance(tokens, list) and isinstance(token_lps, list) and len(tokens) > 0 and len(token_lps) > 0:
            tok = tokens[-1]
            lpv = token_lps[-1]
            if tok is not None and lpv is not None:
                try:
                    token_to_prob[str(tok)] = math.exp(lpv)
                except Exception:
                    token_to_prob[str(tok)] = float(lpv) if isinstance(lpv, (int, float)) else 0.0

        if not token_to_prob:
            raise RuntimeError("OpenRouter returned no usable logprob data for single token")

        return {"text": text, "top_logprobs": token_to_prob}


class OpenRouterChatCompat:
    """
    Compatibility wrapper to match Together interface for easy replacement.
    """
    def __init__(self, model: str, api_key: Optional[str] = None, api_base: str = "https://openrouter.ai/api/v1"):
        self.client = OpenRouterChat(model=model, api_key=api_key, api_base=api_base)

    def generate_texts(self, prompt: str, n: int = 1, temperature: float = 0.7, max_new_tokens: int = 128, stop: Optional[List[str]] = None) -> List[str]:
        return self.client.generate_texts(prompt, n, temperature, max_new_tokens, stop)

    def generate_texts_with_logprobs(self, prompt: str, n: int = 1, temperature: float = 0.7, max_new_tokens: int = 128, top_k: int = 5, stop: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.client.generate_texts_with_logprobs(prompt, n, temperature, max_new_tokens, top_k, stop)

    def generate_one_with_top_logprobs(self, prompt: str, temperature: float = 0.0, top_k: int = 5) -> Dict[str, Any]:
        return self.client.generate_one_with_top_logprobs(prompt, temperature, top_k)

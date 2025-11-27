import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_STEP_PATTERNS = [
    "\n- Step",
    "- Step",
    "\nStep",
    "\n\n",
    "\n**Step",
    "\n## Step",
]

DEFAULT_ANSWER_PATTERNS = [
    "<Answer>:",
    "\n<Answer>:",
    "\n\nAnswer:",
    "\nFinal Answer:",
    "\n\nThe answer is",
]

_TAIL_MARKERS = ["<end of response>", "<end response>", "<end>", "<|im_end|>"]
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "then",
    "so",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "for",
    "with",
    "let",
    "we",
    "thus",
    "therefore",
    "hence",
    "this",
    "that",
    "it",
}


def _normalize_lower(s: str) -> str:
    return s.lower() if s else ""


def extract_answer_span(text: str, answer_patterns: Sequence[str]) -> Tuple[int, int]:
    """Locate the answer span; prefer the last non-empty answer block."""
    if not text:
        return 0, 0

    haystack = text.lower()

    def _compute_end(start_idx: int) -> int:
        end = len(text)
        for marker in _TAIL_MARKERS:
            marker_idx = haystack.find(marker.lower(), start_idx)
            if marker_idx != -1:
                end = min(end, marker_idx)
        return end

    candidates: List[Tuple[int, int]] = []
    for pattern in answer_patterns or DEFAULT_ANSWER_PATTERNS:
        if not pattern:
            continue
        pattern_lower = pattern.lower()
        idx = haystack.find(pattern_lower)
        while idx != -1:
            candidates.append((idx, idx + len(pattern)))
            idx = haystack.find(pattern_lower, idx + 1)

    candidates.sort()
    for start_idx, marker_end in reversed(candidates):
        start = marker_end
        end = _compute_end(start)
        snippet = text[start:end]
        if clean_answer_text(snippet):
            return start, end

    if candidates:
        _, marker_end = candidates[-1]
        start = marker_end
        end = _compute_end(start)
        return start, end

    last_nl = text.rfind("\n")
    start = last_nl + 1 if last_nl >= 0 else 0
    end = _compute_end(start)
    return start, end


def clean_answer_text(answer: str) -> str:
    """Strip tail markers and lightweight math wrappers."""
    if not answer:
        return ""

    s = answer.strip()
    for marker in _TAIL_MARKERS:
        s = re.sub(re.escape(marker) + r"\s*$", "", s, flags=re.IGNORECASE)

    s = s.strip()
    latex_inline = re.match(r"^\\\((.*?)\\\)$", s, flags=re.DOTALL)
    if latex_inline:
        s = latex_inline.group(1).strip()

    dollar_inline = re.match(r"^\${1,2}(.*)\${1,2}$", s, flags=re.DOTALL)
    if dollar_inline:
        s = dollar_inline.group(1).strip()

    s = re.sub(r"\\n$", "", s).strip()
    return s


def compute_reasoning_importance(reasoning_text: str) -> float:
    """Fallback heuristic mirroring the original implementation."""
    if not reasoning_text:
        return 0.5
    digits = sum(ch.isdigit() for ch in reasoning_text)
    ops = sum(ch in "+-*/" for ch in reasoning_text)
    tokens = max(1, len(reasoning_text.split()))
    score = (digits + ops) / tokens
    return float(max(0.0, min(1.0, score)))


class CotUqEvidenceExtractor:
    """Reusable extractor that computes answer/keyword evidence from traces."""

    def __init__(
        self,
        step_patterns: Optional[Sequence[str]] = None,
        answer_patterns: Optional[Sequence[str]] = None,
        max_steps: Optional[int] = None,
        max_empty_steps: Optional[int] = None,
        max_keywords: int = 5,
    ):
        self.step_patterns = step_patterns or DEFAULT_STEP_PATTERNS
        self.answer_patterns = answer_patterns or DEFAULT_ANSWER_PATTERNS
        self.max_steps = max_steps
        self.max_empty_steps = max_empty_steps
        self.max_keywords = max(1, max_keywords or 1)

    def extract(
        self, text: str, token_probs: Optional[List[Tuple[str, Optional[float]]]]
    ) -> Dict[str, Any]:
        text = text or ""
        token_probs = token_probs or []
        token_spans = self._build_token_spans(text, token_probs)
        ans_start, ans_end = extract_answer_span(text, self.answer_patterns)
        raw_answer = text[ans_start:ans_end]
        clean_answer = clean_answer_text(raw_answer)
        answer_tokens = self._collect_tokens_in_span(token_spans, ans_start, ans_end)
        answer_probs = [tok["prob"] for tok in answer_tokens if tok["prob"] is not None]
        answer_mean = (
            float(sum(answer_probs) / len(answer_probs)) if answer_probs else 0.5
        )

        reasoning_text = text[:ans_start]
        steps = self._extract_steps(reasoning_text, ans_start)
        step_evidence: List[Dict[str, Any]] = []
        empty_streak = 0

        for idx, step in enumerate(steps):
            keywords = self._extract_keywords(step["text"])
            contributions = self._compute_keyword_contributions(keywords)
            keyword_probs: Dict[str, List[Optional[float]]] = {}
            keyword_tokens: Dict[str, List[str]] = {}

            for keyword in keywords:
                span = self._locate_keyword_span(
                    step["text"], keyword, step["text_start"]
                )
                if not span:
                    continue
                tokens = self._collect_tokens_in_span(token_spans, span[0], span[1])
                keyword_tokens[keyword] = [t["token"] for t in tokens]
                keyword_probs[keyword] = [t["prob"] for t in tokens]

            has_signal = any(
                probs
                for probs in keyword_probs.values()
                if any(p is not None for p in probs)
            )
            if has_signal:
                empty_streak = 0
            else:
                empty_streak += 1

            step_evidence.append(
                {
                    "name": step["name"],
                    "text": step["text"],
                    "span": [step["text_start"], step["text_end"]],
                    "keywords": keywords,
                    "keyword_probabilities": keyword_probs,
                    "keyword_token_text": keyword_tokens,
                    "keyword_contributions": contributions,
                }
            )

            if self.max_steps and len(step_evidence) >= self.max_steps:
                break
            if self.max_empty_steps and empty_streak >= self.max_empty_steps:
                break

        keyword_scores: List[float] = []
        for step in step_evidence:
            for keyword in step["keywords"]:
                probs = step["keyword_probabilities"].get(keyword, [])
                valid = [p for p in probs if p is not None]
                if not valid:
                    continue
                weight = step["keyword_contributions"].get(keyword, 0.0) or 0.0
                mean_prob = float(sum(valid) / len(valid))
                keyword_scores.append(mean_prob * (weight if weight > 0 else 1.0))

        keyword_confidence = (
            float(sum(keyword_scores) / len(keyword_scores))
            if keyword_scores
            else compute_reasoning_importance(reasoning_text)
        )

        return {
            "answer": {
                "text": raw_answer,
                "clean_text": clean_answer,
                "span": [ans_start, ans_end],
                "token_details": answer_tokens,
                "probabilities": answer_probs,
                "mean_probability": answer_mean,
            },
            "reasoning_text": reasoning_text.strip(),
            "steps": step_evidence,
            "keyword_confidence": keyword_confidence,
        }

    def _build_token_spans(
        self, text: str, token_probs: List[Tuple[str, Optional[float]]]
    ) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        cursor = 0
        for token_text, prob in token_probs:
            token_text = token_text or ""
            start = cursor
            if token_text:
                expected = text[cursor : cursor + len(token_text)]
                if expected != token_text:
                    idx = text.find(token_text, cursor)
                    if idx != -1:
                        start = idx
                end = start + len(token_text)
            else:
                end = start
            spans.append(
                {
                    "token": token_text,
                    "prob": prob,
                    "start": start,
                    "end": end,
                }
            )
            cursor = end
        return spans

    def _collect_tokens_in_span(
        self, token_spans: List[Dict[str, Any]], start: int, end: int
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        if start >= end:
            return collected
        for token in token_spans:
            if token["end"] <= start or token["start"] >= end:
                continue
            collected.append(token)
        return collected

    def _extract_steps(self, reasoning_text: str, offset: int) -> List[Dict[str, Any]]:
        if not reasoning_text:
            return []

        boundaries: List[int] = []
        for pattern in self.step_patterns:
            if not pattern:
                continue
            start = 0
            while True:
                idx = reasoning_text.find(pattern, start)
                if idx == -1:
                    break
                boundaries.append(idx)
                start = idx + max(1, len(pattern))
        boundaries = sorted(set(boundaries))

        segments: List[Tuple[int, int]] = []
        if boundaries:
            for idx, start in enumerate(boundaries):
                end = (
                    boundaries[idx + 1]
                    if idx + 1 < len(boundaries)
                    else len(reasoning_text)
                )
                segments.append((start, end))
        elif reasoning_text.strip():
            segments.append((0, len(reasoning_text)))

        steps: List[Dict[str, Any]] = []
        for seg_start, seg_end in segments:
            # raw_segment = reasoning_text[seg_start:seg_end]
            content_start = seg_start
            while content_start < seg_end and reasoning_text[content_start].isspace():
                content_start += 1
            content_end = seg_end
            while (
                content_end > content_start
                and reasoning_text[content_end - 1].isspace()
            ):
                content_end -= 1
            if content_end <= content_start:
                continue
            content = reasoning_text[content_start:content_end]
            name, body, body_start = self._split_step_label(
                content, content_start + offset, len(steps) + 1
            )
            body_end = body_start + len(body)
            if not body:
                continue
            steps.append(
                {
                    "name": name,
                    "text": body,
                    "text_start": body_start,
                    "text_end": body_end,
                }
            )

        return steps

    def _split_step_label(
        self, content: str, global_start: int, idx: int
    ) -> Tuple[str, str, int]:
        colon_index = content.find(":")
        if colon_index == -1:
            return f"Step {idx}", content.strip(), global_start

        label = content[:colon_index].strip() or f"Step {idx}"
        body_rel_start = colon_index + 1
        while body_rel_start < len(content) and content[body_rel_start].isspace():
            body_rel_start += 1
        body = content[body_rel_start:].strip()
        body_start = global_start + body_rel_start
        return label, body, body_start

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9\./\\%]+", text)
        keywords: List[str] = []
        seen = set()
        for token in tokens:
            normalized = token.strip()
            if not normalized:
                continue
            lower = normalized.lower()
            if lower in _STOPWORDS:
                continue
            if len(normalized) <= 2 and not any(ch.isdigit() for ch in normalized):
                continue
            if lower in seen:
                continue
            keywords.append(normalized)
            seen.add(lower)
            if len(keywords) >= self.max_keywords:
                break
        if not keywords and tokens:
            for token in tokens:
                normalized = token.strip()
                if not normalized:
                    continue
                keywords.append(normalized)
                if len(keywords) >= self.max_keywords:
                    break
        return keywords

    def _compute_keyword_contributions(self, keywords: List[str]) -> Dict[str, float]:
        if not keywords:
            return {}
        weights: Dict[str, float] = {}
        for keyword in keywords:
            weight = 2.0 if any(ch.isdigit() for ch in keyword) else 1.0
            weights[keyword] = weight
        total = sum(weights.values()) or 1.0
        return {kw: weight / total for kw, weight in weights.items()}

    def _locate_keyword_span(
        self, step_text: str, keyword: str, step_start: int
    ) -> Optional[Tuple[int, int]]:
        if not keyword:
            return None
        idx = step_text.lower().find(keyword.lower())
        if idx == -1:
            return None
        return step_start + idx, step_start + idx + len(keyword)

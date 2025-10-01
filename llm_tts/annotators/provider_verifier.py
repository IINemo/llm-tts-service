import re
from typing import List, Optional


class ProviderVerifier:
    """
    Generic verifier that uses an existing text-generation client (Together/OpenRouter wrappers)
    to judge whether a solution is correct.

    Expects the client to expose `generate_texts(prompt, n, temperature, max_new_tokens, stop)`.
    """

    def __init__(
        self,
        client,
        prompt_template: str,
        n: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ):
        self.client = client
        self.prompt_template = prompt_template
        self.n = max(1, int(n))
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.stop = list(stop) if stop else None

        # Precompile robust parsers
        self._grade_re = re.compile(r"<\s*grade\s*>\s*:\s*(correct|incorrect)", re.IGNORECASE)
        self._word_re = re.compile(r"\b(correct|incorrect)\b", re.IGNORECASE)

    def _render_prompt(self, question: str, solution: str, gold_answer: str = None) -> str:
        # Support multiple variable names for convenience
        return self.prompt_template.format(
            question=question,
            answer=solution,
            problem=question,
            solution=solution,
            gold_answer=gold_answer,
        )

    def _parse_is_correct(self, text: str) -> Optional[bool]:
        if not text:
            return None
        m = self._grade_re.search(text)
        if m:
            label = m.group(1).strip().lower()
            return label == "correct"
        # Fallback: last mention of the keywords as a whole word
        hits = list(self._word_re.finditer(text))
        if hits:
            label = hits[-1].group(1).strip().lower()
            return label == "correct"
        return None

    def verify_batch(self, problems: List[str], solutions: List[str], gold_answers: List[str]) -> List[Optional[bool]]:
        outputs: List[Optional[bool]] = []
        for q, s, g in zip(problems, solutions, gold_answers):
            prompt = self._render_prompt(q or "", s or "", g)
            texts = self.client.generate_texts(
                prompt,
                n=self.n,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stop=self.stop,
            )
            first = texts[0] if texts else ""
            outputs.append(self._parse_is_correct(first))
        return outputs

    def verify_batch_with_texts(self, problems: List[str], solutions: List[str], gold_answers: List[str] = None):
        """
        Returns list of dicts: {"is_correct": Optional[bool], "texts": List[str]}
        """
        outputs = []
        for q, s, g in zip(problems, solutions, gold_answers):
            prompt = self._render_prompt(q or "", s or "", g)
            texts = self.client.generate_texts(
                prompt,
                n=self.n,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stop=self.stop,
            )
            first = texts[0] if texts else ""
            outputs.append({
                "is_correct": self._parse_is_correct(first),
                "texts": texts or [],
            })
        return outputs



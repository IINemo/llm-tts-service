import logging
import math
from typing import Dict, List, Optional, Tuple

from llm_tts.step_boundary_detector import uncert_detector
from llm_tts.step_candidate_generator_base import StepCandidate

log = logging.getLogger(__name__)


class UncertaintyGuidedCoT_PD:
    def __init__(
        self,
        api_client: Optional[object] = None,
        candidates_per_step: int = 3,
        max_steps: int = 10,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        problem_type: str = "math",
        uncertainty_threshold: Optional[float] = None,
        uncertainty_metric: str = "pd",
        uncertainty_top_k: Optional[int] = None,
        step_marker_patterns: Optional[List[str]] = None,
        detector_step_patterns: Optional[List[str]] = None,
        detector_answer_patterns: Optional[List[str]] = None,
        eos_token: Optional[str] = None,
    ):
        """
        Initialize Uncertainty-Guided CoT with Predictive Distribution (PD).

        Args:
            api_client: Unified API client (TogetherAIModel, OpenRouterModel, or compatible)
            candidates_per_step: Number of candidate paths to generate at each step
            max_steps: Maximum number of reasoning steps
            max_new_tokens: Maximum tokens per step
            temperature: Sampling temperature for generation
            problem_type: Type of problem ("math" or "qa")
            uncertainty_threshold: Threshold for triggering CoT branching
            uncertainty_metric: Metric for uncertainty ("pd" or "entropy")
            uncertainty_top_k: Number of top logprobs to consider
            step_marker_patterns: Patterns for step markers (used for generation formatting)
            detector_step_patterns: Patterns for step detection (optional override)
            detector_answer_patterns: Patterns for answer detection (optional override)
            eos_token: End-of-sequence token
        """
        if api_client is None:
            raise RuntimeError(
                "api_client must be provided (TogetherAIModel/OpenRouterModel or compatible)"
            )
        self.api_client = api_client

        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.problem_type = problem_type.lower()
        self.uncertainty_metric = (uncertainty_metric or "pd").lower()
        self.uncertainty_top_k = (
            int(uncertainty_top_k) if uncertainty_top_k is not None else None
        )
        self.step_marker_patterns = step_marker_patterns
        self.eos_token = eos_token

        if uncertainty_threshold is None:
            if self.uncertainty_metric == "entropy":
                self.uncertainty_threshold = 1.2 if self.problem_type == "qa" else 1.0
            else:
                self.uncertainty_threshold = 0.45 if self.problem_type == "qa" else 0.40
        else:
            self.uncertainty_threshold = uncertainty_threshold

        # Create detector with optional pattern overrides
        # UncertStepBoundaryDetector uses case-insensitive matching and flexible patterns
        self.detector = uncert_detector(
            step_patterns=detector_step_patterns,
            answer_patterns=detector_answer_patterns,
            max_tokens_per_step=max_new_tokens,
        )

    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        """
        Uncertainty-guided decoding pipeline.

        Loop:
          1) Probe next-token uncertainty at current prompt using top-2 probabilities.
          2) If uncertainty <= threshold → greedy extend by up to max_new_tokens.
             Else → sample K paths, compute path confidences (avg p1-p2), pick best.
          3) Repeat until an <Answer>: tag is produced or max_steps is reached.
        """
        log_prompt = prompt.replace("\n", "\\n")
        log.info(f"Initial prompt: {log_prompt}")
        trajectory = prompt
        prompt_cut = len(prompt)
        selected_steps: List[str] = []
        uncertainties: List[Dict[str, float]] = []
        decision_trace: List[Dict[str, any]] = []

        for step_num in range(self.max_steps):
            log.info(f"\n=== PD step {step_num+1} ===")
            log_trajectory = trajectory.replace("\n", "\\n")
            log.info(f"Trajectory: {log_trajectory}")

            # 1) Probe uncertainty BEFORE deciding how many paths to generate
            #    Use a single-token generation with logprobs to estimate PD uncertainty.

            # Build stop markers for the next step, OpenAI API only supports 4 stop markers,
            # so we pass only markers from configured patterns
            stop_markers = self._build_stop_variants(step_num, build_all=False)

            try:
                # Generate 1 token with logprobs for uncertainty estimation
                probe_results = self.api_client.generate_with_confidence(
                    prompt=trajectory,
                    max_tokens=1,
                    temperature=0.0,
                    n=1,
                    top_k=self.uncertainty_top_k or 5,
                )

                # Extract token probabilities from the first (only) result
                _, token_data = probe_results[0]
                token_probs_last = {}

                if token_data and len(token_data) > 0:
                    # Get the first (only) token's top alternatives
                    first_token = token_data[0]
                    top_alts = first_token.get("top_logprobs", [])

                    for alt in top_alts:
                        tok = alt.get("token")
                        lp = alt.get("logprob")
                        if tok is not None and lp is not None:
                            token_probs_last[tok] = math.exp(lp)

                pd_info = self._compute_uncertainty(
                    trajectory, token_to_prob=token_probs_last
                )
            except Exception as e:
                raise RuntimeError(f"Uncertainty probe failed: {e}")
            uncertainties.append(pd_info)
            uncertainty = pd_info["uncertainty"]
            used_cot = bool(uncertainty > self.uncertainty_threshold)

            # Decide branching parameters
            n_for_gen = self.candidates_per_step if used_cot else 1
            # Ensure sampling when branching to CoT even if configured temperature is 0
            temp_for_gen = (
                self.temperature
                if used_cot and self.temperature > 0
                else (0.0 if not used_cot else 0.7)
            )

            # 2) Generate step(s) according to the chosen branch
            gen_results = self.api_client.generate_with_confidence(
                prompt=trajectory,
                max_tokens=self.max_new_tokens,
                temperature=temp_for_gen,
                n=n_for_gen,
                top_k=self.uncertainty_top_k or 5,
                stop=stop_markers,
            )

            if used_cot:
                # 2.2) Multi-path: compute confidence per path and select max
                candidates: List[StepCandidate] = []
                confidences: List[float] = []

                for text, token_data in gen_results:
                    # Truncate at stop markers
                    cut_text = text
                    for marker in stop_markers:
                        if marker in cut_text:
                            cut_text = cut_text[: cut_text.index(marker)]

                    cut_text = self.detector.extract_step_text(cut_text)

                    # Compute confidence from token logprobs
                    gaps: List[float] = []
                    if token_data:
                        for token_info in token_data:
                            top_alts = token_info.get("top_logprobs", [])
                            if len(top_alts) >= 2:
                                # Get top 2 probabilities
                                probs = []
                                for alt in top_alts[:2]:
                                    lp = alt.get("logprob")
                                    if lp is not None:
                                        probs.append(math.exp(lp))
                                if len(probs) >= 2:
                                    gaps.append(max(0.0, probs[0] - probs[1]))
                            elif len(top_alts) == 1:
                                lp = top_alts[0].get("logprob")
                                if lp is not None:
                                    gaps.append(math.exp(lp))

                    conf = float(sum(gaps) / len(gaps)) if gaps else 0.0
                    confidences.append(conf)
                    candidates.append(
                        StepCandidate(
                            text=cut_text.strip(),
                            token_ids=[],
                            is_complete=True,
                            is_trajectory_complete=self.detector.is_trajectory_complete(
                                text
                            ),
                            generation_scores={"confidence": conf},
                            raw_text=text,
                        )
                    )

                if not candidates:
                    raise RuntimeError("No candidates returned for CoT branch")
                best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
                best = candidates[best_idx]
                chosen_text = best.text
                branch = "cot"
                extra = {
                    "candidates": [
                        {
                            "text": c.text,
                            "confidence": confidences[i],
                            "selected": i == best_idx,
                        }
                        for i, c in enumerate(candidates)
                    ]
                }
            else:
                # 2.1) Greedy continuation: take first completion
                text, _ = gen_results[0]

                # Truncate at stop markers
                cut_text = text
                for marker in stop_markers:
                    if marker in cut_text:
                        cut_text = cut_text[: cut_text.index(marker)]

                chosen_text = self.detector.extract_step_text(cut_text).strip()
                branch = "greedy"
                extra = {}

            # 3) Append and check for answer
            trajectory += chosen_text
            log_chosen_text = chosen_text.replace("\n", "\\n")
            log.info(f"Step {step_num+1} generation: {log_chosen_text}")
            selected_steps.append(chosen_text)

            decision = {
                "step": step_num + 1,
                "branch": branch,
                "uncertainty": float(uncertainty),
                "threshold": float(self.uncertainty_threshold),
            }
            decision.update(extra)
            decision_trace.append(decision)

            if self.detector.contains_answer_pattern(trajectory[prompt_cut:]):
                log.info(f"Answer found in step {step_num+1}")
                break

            if trajectory.endswith("#"):
                trajectory = trajectory.rstrip("#").rstrip()
            next_header = self._format_step_header(step_num + 2)
            if not trajectory.endswith("\n"):
                trajectory += "\n"
            trajectory += next_header

        # If no explicit <Answer>:, try to elicit one succinctly
        if not self.detector.contains_answer_pattern(trajectory):
            final_answer, _ = self._generate_final_answer(trajectory)
            trajectory += final_answer
            selected_steps.append(final_answer)

        return {
            "trajectory": trajectory,
            "steps": selected_steps,
            "uncertainties": uncertainties,
            "completed": self.detector.contains_answer_pattern(trajectory),
            "decision_trace": decision_trace,
        }

    # Removed legacy scorer-based selection method

    def _compute_uncertainty(
        self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        if self.uncertainty_metric == "entropy":
            return self._compute_entropy_uncertainty(trajectory)
        return self._compute_pd_uncertainty(trajectory, token_to_prob=token_to_prob)

    def _compute_pd_uncertainty(
        self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        # Strict: require provided token_to_prob collected at the decision point
        if not token_to_prob:
            raise RuntimeError(
                "Missing token probabilities for PD uncertainty computation"
            )
        probs_list = sorted(token_to_prob.values(), reverse=True)
        if len(probs_list) == 0:
            raise RuntimeError("No probabilities provided for PD uncertainty")
        p1 = probs_list[0]
        p2 = probs_list[1] if len(probs_list) > 1 else 0.0
        uncertainty = 1.0 - (p1 - p2)
        return {
            "uncertainty": float(uncertainty),
            "p1": float(p1),
            "p2": float(p2),
            "confidence_gap": float(p1 - p2),
        }

    def _build_stop_variants(self, step_num: int, build_all: bool = True) -> List[str]:
        # Build from configured patterns to be robust across models.
        # We stop at the NEXT step header number (e.g., generating Step N should stop at Step N+1).
        if not build_all:
            return [
                p.replace("{n}", str(step_num + 2)) for p in self.step_marker_patterns
            ]

        next_headers = [
            p.replace("{n}", str(step_num + 2)) for p in self.step_marker_patterns
        ]

        variants: List[str] = []
        bases = list(next_headers)
        # Add relaxed header forms (preserve the explicit number to avoid zero-length cuts)
        for nh in list(bases):
            if nh.startswith("## "):
                bare = nh[3:]
                bases.extend(
                    [
                        bare,
                        nh.rstrip(":"),
                        bare.rstrip(":"),
                        nh + ":",
                        bare + ":",
                        nh.replace("## ", "### "),
                        nh.replace("## ", "# "),
                    ]
                )
            elif nh.startswith("##"):
                spaced = nh.replace("##", "## ")
                bases.extend(
                    [
                        spaced,
                        spaced + ":",
                    ]
                )
        # Prepend whitespace/newlines variants
        prefixes = ["", "\n", "\n\n", " ", "\t"]
        variants += [p + b for p in prefixes for b in bases]

        if self.eos_token:
            variants.append(self.eos_token)

        uniq = list(set(variants))
        return uniq

    def _format_step_header(self, n: int) -> str:
        try:
            # Prefer configured step marker patterns; use the first as the canonical formatter
            if self.step_marker_patterns and isinstance(
                self.step_marker_patterns, (list, tuple)
            ):
                base = (
                    self.step_marker_patterns[0]
                    if len(self.step_marker_patterns) > 0
                    else "Step {n}:"
                )
                if "{n}" in base:
                    return base.replace("{n}", str(n))
                # If no placeholder is present, append the number sensibly
                # Ensure trailing colon if the base ends with a word character
                suffix = ":" if base and base[-1].isalnum() else ""
                return f"{base} {n}{suffix}"
        except Exception:
            pass
        return f"Step {n}:"

    def _compute_entropy_uncertainty(
        self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute entropy-based uncertainty using only the top-20 probabilities.
        We renormalize the top-20 probabilities to sum to 1 and compute Shannon entropy (nats).
        """
        if not token_to_prob:
            raise RuntimeError(
                "Missing token probabilities for entropy uncertainty computation"
            )
        probs = sorted(token_to_prob.values(), reverse=True)
        if len(probs) == 0:
            raise RuntimeError("No probabilities provided for entropy uncertainty")

        total = sum(probs)
        if total <= 0:
            raise RuntimeError("Invalid probability sum for entropy computation")

        probs = [p / total for p in probs]
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)

        if len(probs) == 0:
            raise RuntimeError("No valid probabilities for entropy computation")

        return {
            "uncertainty": float(entropy),
        }

    def _generate_final_answer(self, trajectory: str) -> Tuple[str, float]:
        """
        Generate multiple answer candidates and select the one with the highest
        confidence score based on average top-2 probability gap per token,
        i.e., mean_t [ p1(t) - p2(t) ]. Returns (best_answer_text, best_confidence).
        """
        num = self.candidates_per_step
        results = self.api_client.generate_with_confidence(
            prompt=trajectory,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            n=num,
            top_k=self.uncertainty_top_k or 5,
        )

        if not results:
            return "\n<Answer>:", 0.0

        # get the confidence of the answer as in the paper
        confidences: List[float] = []
        texts: List[str] = []

        for text, token_data in results:
            texts.append(text)
            gaps: List[float] = []

            if token_data:
                for token_info in token_data:
                    top_alts = token_info.get("top_logprobs", [])
                    if len(top_alts) >= 2:
                        # Get top 2 probabilities
                        probs = []
                        for alt in top_alts[:2]:
                            lp = alt.get("logprob")
                            if lp is not None:
                                probs.append(math.exp(lp))
                        if len(probs) >= 2:
                            gaps.append(max(0.0, probs[0] - probs[1]))
                    elif len(top_alts) == 1:
                        lp = top_alts[0].get("logprob")
                        if lp is not None:
                            gaps.append(math.exp(lp))

            conf = float(sum(gaps) / len(gaps)) if gaps else 0.0
            confidences.append(conf)

        best_idx = max(range(len(texts)), key=lambda i: confidences[i])
        best = texts[best_idx] or ""

        # Ensure explicit <Answer>: tag; if missing, try to extract a number and format
        low = best.lower()
        if "<answer>:" not in low:
            import re as _re

            m = _re.search(r"-?\d+(?:\.\d+)?", best)
            if m:
                best = f"\n<Answer>: {m.group(0)}"
            else:
                best = "\n<Answer>:"

        return best, float(confidences[best_idx])

    def cleanup(self):
        pass

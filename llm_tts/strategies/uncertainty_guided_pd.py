import math
import re
from typing import Dict, List, Optional, Tuple

from llm_tts.step_generation import StepCandidateGenerator, StepCandidate
from llm_tts.step_detection import uncert_detector, BatchStepStoppingCriteria

import logging
log = logging.getLogger(__name__)


class UncertaintyGuidedCoT_PD:
    def __init__(
        self,
        model,
        scorer,
        candidates_per_step: int,
        max_steps: int,
        max_new_tokens: int,
        temperature: float,
        generation_batch_size: Optional[int] = None,
        problem_type: str = "math",
        uncertainty_threshold: Optional[float] = None,
        greedy_temperature: float = 0.1,
        openrouter_client=None,
        together_client=None,
        uncertainty_metric: str = "pd",
        uncertainty_top_k: Optional[int] = None,
        step_header_template: Optional[str] = None,
        step_marker_patterns: Optional[List[str]] = None,
    ):
        self.model = model
        self.scorer = scorer
        self.openrouter = openrouter_client
        self.together = together_client
        # Prefer explicit provided client; Together and OpenRouter share a compat interface
        self.api_client = together_client if together_client is not None else openrouter_client
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.generation_batch_size = generation_batch_size or candidates_per_step
        self.problem_type = problem_type.lower()
        self.uncertainty_metric = (uncertainty_metric or "pd").lower()
        self.uncertainty_top_k = int(uncertainty_top_k) if uncertainty_top_k is not None else 2
        # self.step_header_template = step_header_template or "## Step {n}"
        self.step_marker_patterns = step_marker_patterns
        # Optional regex, used only if caller formats prompts with explicit headers.
        # try:
        #     placeholder = "{n}"
        #     if placeholder in self.step_header_template:
        #         parts = self.step_header_template.split(placeholder)
        #         esc0 = re.escape(parts[0])
        #         esc1 = re.escape(parts[1]) if len(parts) > 1 else ""
        #         self.step_header_re = re.compile(rf"{esc0}\s*\d+{esc1}")
        #     else:
        #         self.step_header_re = re.compile(re.escape(self.step_header_template))
        # except Exception:
        #     self.step_header_re = None
        
        if uncertainty_threshold is None:
            if self.uncertainty_metric == "entropy":
                self.uncertainty_threshold = 1.2 if self.problem_type == "qa" else 1.0
            else:
                self.uncertainty_threshold = 0.45 if self.problem_type == "qa" else 0.40
        else:
            self.uncertainty_threshold = uncertainty_threshold

        self.detector = uncert_detector(
            max_tokens_per_step=max_new_tokens,
        )

        if self.api_client is None:
            raise RuntimeError("Must provide either api_client (together_client or openrouter_client) or local model")

    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        """
        Uncertainty-guided decoding pipeline.

        Loop:
          1) Probe next-token uncertainty at current prompt using top-2 probabilities.
          2) If uncertainty <= threshold → greedy extend by up to max_new_tokens.
             Else → sample K paths, compute path confidences (avg p1-p2), pick best.
          3) Repeat until an <Answer>: tag is produced or max_steps is reached.
        """
        log_prompt = prompt.replace('\n', '\\n')
        log.info(f"Initial prompt: {log_prompt}")
        trajectory = prompt
        prompt_cut = len(prompt)
        selected_steps: List[str] = []
        uncertainties: List[Dict[str, float]] = []
        decision_trace: List[Dict[str, any]] = []

        for step_num in range(self.max_steps):
            log.info(f"\n=== PD step {step_num+1} ===")
            log_trajectory = trajectory.replace('\n', '\\n')
            log.info(f"Trajectory: {log_trajectory}")

            # 1) Probe uncertainty BEFORE deciding how many paths to generate
            #    Use a single-token top-k distribution to estimate PD uncertainty.
            stop_markers = self._build_stop_variants(step_num)
            try:
                probe = self.api_client.generate_one_with_top_logprobs(
                    prompt=trajectory,
                    temperature=0.0,
                    top_k=max(5, self.uncertainty_top_k),
                )
                token_probs_last = probe.get("top_logprobs", {}) or {}
                pd_info = self._compute_pd_uncertainty(trajectory, token_to_prob=token_probs_last)
            except Exception as e:
                raise f"Uncertainty probe failed: {e}"
            uncertainties.append(pd_info)
            uncertainty = pd_info["uncertainty"]
            used_cot = bool(uncertainty > self.uncertainty_threshold)

            # Decide branching parameters
            n_for_gen = self.candidates_per_step if used_cot else 1
            # Ensure sampling when branching to CoT even if configured temperature is 0
            temp_for_gen = (self.temperature if used_cot and self.temperature > 0 else (0.0 if not used_cot else 0.7))

            # 2) Generate step(s) according to the chosen branch
            raw_outputs = self.api_client.generate_texts_with_logprobs(
                prompt=trajectory,
                n=n_for_gen,
                temperature=temp_for_gen,
                max_new_tokens=self.max_new_tokens,
                top_k=max(5, self.uncertainty_top_k),
                stop=stop_markers,
            )

            # Prepare helper data from first path for greedy branch
            token_probs_last = {}
            first_cut_text = None
            first_last_top = None
            if raw_outputs:
                first_cut_text, first_last_top, _ = self._truncate_generated_result(raw_outputs[0], stop_markers)
                if first_last_top is not None:
                    if isinstance(first_last_top, dict):
                        token_probs_last = {t: math.exp(lp) for t, lp in first_last_top.items() if lp is not None}
                    elif isinstance(first_last_top, list):
                        for ent in first_last_top:
                            tok = ent.get('token') if isinstance(ent, dict) else getattr(ent, 'token', None)
                            lp = ent.get('logprob') if isinstance(ent, dict) else getattr(ent, 'logprob', None)
                            if tok is not None and lp is not None:
                                token_probs_last[tok] = math.exp(lp)

            if used_cot:
                # 2.2) Multi-path: compute confidence per path and select max
                candidates: List[StepCandidate] = []
                confidences: List[float] = []
                for r in raw_outputs:
                    raw_generated = r.get("text", "")
                    cut_text, _, kept = self._truncate_generated_result(r, stop_markers)
                    cut_text = self.detector.extract_step_text(cut_text)
                    tokens = r.get("tokens", [])
                    top_ls = r.get("top_logprobs", [])
                    gaps: List[float] = []
                    token_limit = min(len(tokens), len(top_ls), kept if kept is not None else len(top_ls))
                    for idx in range(token_limit):
                        cur = top_ls[idx]
                        p1, p2 = self._extract_top_two_probs(cur)
                        gaps.append(max(0.0, p1 - p2))
                    conf = float(sum(gaps) / len(gaps)) if gaps else 0.0
                    confidences.append(conf)
                    candidates.append(
                        StepCandidate(
                            text=cut_text.strip(),
                            token_ids=[],
                            is_complete=True,
                            is_trajectory_complete=self.detector.is_trajectory_complete(raw_generated),
                            generation_scores={"confidence": conf},
                            raw_text=raw_generated,
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
                # 2.1) Greedy continuation: take first completion produced above
                raw_generated = raw_outputs[0].get("text", "") if raw_outputs else ""
                cut_text = first_cut_text if first_cut_text is not None else raw_generated
                chosen_text = self.detector.extract_step_text(cut_text).strip()
                branch = "greedy"
                extra = {}

            # 3) Append and check for answer
            trajectory += chosen_text
            log_chosen_text = chosen_text.replace('\n', '\\n')
            log.info(f"Step {step_num+1} generation: {log_chosen_text}")
            selected_steps.append(chosen_text)

            decision = {
                "step": step_num+1,
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


    def _score_and_select_best(self, trajectory: str, candidates) -> Tuple[int, any]:
        candidate_texts = [c.text for c in candidates]
        all_validities = self.scorer.score_candidates(trajectory, candidate_texts)
        best_idx = max(range(len(all_validities)), key=lambda i: all_validities[i])
        return best_idx, candidates[best_idx]

    def _compute_uncertainty(self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        if self.uncertainty_metric == "entropy":
            return self._compute_entropy_uncertainty(trajectory)
        return self._compute_pd_uncertainty(trajectory, token_to_prob=token_to_prob)

    def _compute_pd_uncertainty(self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        # Strict: require provided token_to_prob collected at the decision point
        if not token_to_prob:
            raise RuntimeError("Missing token probabilities for PD uncertainty computation")
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

    def _extract_top_two_probs(self, top_logprobs_entry) -> Tuple[float, float]:
        """Extract p1 and p2 from a top_logprobs structure which may be dict or list.
        Returns (p1, p2) as probabilities (not logprobs). Missing entries map to 0.
        """
        try:
            if isinstance(top_logprobs_entry, dict):
                pairs = list(top_logprobs_entry.items())
                try:
                    pairs.sort(key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
                except Exception:
                    pass
                vals = [math.exp(v) if v is not None and v <= 0 else (v if v <= 1.0 else 0.0) for _, v in pairs]
            elif isinstance(top_logprobs_entry, list):
                try:
                    sorted_top = sorted(
                        top_logprobs_entry,
                        key=lambda e: (e.get('logprob') if isinstance(e, dict) else getattr(e, 'logprob', None)) or -1e9,
                        reverse=True,
                    )
                except Exception:
                    sorted_top = top_logprobs_entry
                vals = []
                for entry in sorted_top:
                    lp = entry.get('logprob') if isinstance(entry, dict) else getattr(entry, 'logprob', None)
                    if lp is None:
                        continue
                    vals.append(math.exp(lp))
            else:
                vals = []
        except Exception:
            vals = []

        if not vals:
            return 0.0, 0.0
        if len(vals) == 1:
            return float(vals[0]), 0.0
        return float(vals[0]), float(vals[1])

    def _build_stop_variants(self, step_num: int) -> List[str]:
        # Build from configured patterns to be robust across models.
        # We stop at the NEXT step header number (e.g., generating Step N should stop at Step N+1).
        next_headers = [p.replace("{n}", str(step_num + 2)) for p in self.step_marker_patterns]
        variants: List[str] = []
        bases = list(next_headers)
        # Add relaxed header forms (preserve the explicit number to avoid zero-length cuts)
        for nh in list(bases):
            if nh.startswith("## "):
                bare = nh[3:]
                bases.extend([
                    bare,
                    nh.rstrip(":"),
                    bare.rstrip(":"),
                    nh + ":",
                    bare + ":",
                    nh.replace("## ", "### "),
                    nh.replace("## ", "# "),
                ])
            elif nh.startswith("##"):
                spaced = nh.replace("##", "## ")
                bases.extend([
                    spaced,
                    spaced + ":",
                ])
        # Prepend whitespace/newlines variants
        prefixes = ["", "\n", "\n\n", " ", "\t"]
        for b in bases:
            for p in prefixes:
                variants.append(p + b)
        # Answer tag variants
        for a in ["<Answer>:", "\n<Answer>:", "\n\n<Answer>:", " Answer:", "\nAnswer:"]:
            variants.append(a)
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for v in variants:
            if v not in seen:
                uniq.append(v)
                seen.add(v)
        return uniq

    def _truncate_generated_result(self, entry: Dict[str, any], stop_variants: List[str]) -> Tuple[str, Optional[any], Optional[int]]:
        """Cut raw generated text at earliest occurrence of any stop variant.
        Returns (cut_text, last_toplogprobs_before_cut, num_tokens_kept).
        We align by accumulating token strings until reaching the cut position.
        """
        raw = entry.get("text", "") or ""
        # Find earliest stop occurrence
        cut_pos = len(raw)
        for s in stop_variants:
            if not s:
                continue
            pos = raw.find(s)
            if pos != -1:
                cut_pos = min(cut_pos, pos)
        if cut_pos == len(raw):
            # nothing to cut
            tops = entry.get("top_logprobs", [])
            last_top = tops[-1] if tops else None
            return raw, last_top, None
        # Align tokens to character cut position
        tokens = entry.get("tokens", []) or []
        tops = entry.get("top_logprobs", []) or []
        acc = ""
        kept = 0
        for i, tok in enumerate(tokens):
            acc += tok or ""
            if len(acc) >= cut_pos:
                kept = i  # keep tokens up to i-1; boundary before token i
                break
        if kept <= 0:
            last_top = None
        else:
            last_top = tops[kept - 1] if kept - 1 < len(tops) else (tops[-1] if tops else None)
        return raw[:cut_pos], last_top, kept

    def _format_step_header(self, n: int) -> str:
        try:
            # Prefer configured step marker patterns; use the first as the canonical formatter
            if self.step_marker_patterns and isinstance(self.step_marker_patterns, (list, tuple)):
                base = self.step_marker_patterns[0] if len(self.step_marker_patterns) > 0 else "Step {n}:"
                if "{n}" in base:
                    return base.replace("{n}", str(n))
                # If no placeholder is present, append the number sensibly
                # Ensure trailing colon if the base ends with a word character
                suffix = ":" if base and base[-1].isalnum() else ""
                return f"{base} {n}{suffix}"
        except Exception:
            pass
        return f"Step {n}:"

    def _compute_entropy_uncertainty(self, trajectory: str, token_to_prob: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute entropy-based uncertainty using only the top-20 probabilities.
        We renormalize the top-20 probabilities to sum to 1 and compute Shannon entropy (nats).
        """
        if not token_to_prob:
            raise RuntimeError("Missing token probabilities for entropy uncertainty computation")
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


    def _should_measure_uncertainty(self, trajectory: str, step_num: int) -> bool:
        """Decide when to probe uncertainty to avoid interrupting early planning.

        Rules:
        - Never measure at step 0 (let the model plan first).
        - Trigger if an explicit answer marker appears in tail (to branch to answer phase).
        - Prefer triggering immediately after a configured step header (e.g., "## Step N:").
        - Otherwise, require at least one sentence boundary and sufficient content length.
        """
        if step_num == 0:
            return False

        tail = trajectory[-256:]

        if "<answer>:" in tail.lower() or "\n<answer>:" in tail.lower() or "answer:" in tail.lower():
            return True

        # Prefer triggering right after a configured step header (e.g., "## Step N:")
        if getattr(self, 'step_header_re', None) is not None:
            tail2 = trajectory[-512:]
            last = None
            for m in self.step_header_re.finditer(tail2):
                last = m
            if last is not None:
                after = tail2[last.end():]
                if len(after.strip()) <= 2:
                    return True

        # Require at least one sentence boundary with enough content
        has_sentence_boundary = bool(re.search(r"[\.!\?]\s", tail)) or ("\n\n" in tail)
        long_enough = len(tail.strip()) >= 80

        if self.problem_type == "qa":
            semantic_cues = [
                r"According to",
                r"From the passage",
                r"We can see that",
                r"Combining this with",
                r"This means",
            ]
        else:
            semantic_cues = [
                r"So we have",
                r"Therefore",
                r"Thus",
                r"Hence",
                r"Next,",
            ]

        has_semantic_cue = any(re.search(p, tail) for p in semantic_cues)
        return (has_sentence_boundary and long_enough) or has_semantic_cue

    def _generate_final_answer(self, trajectory: str) -> Tuple[str, float]:
        """
        Generate multiple answer candidates without step detection and select best via scorer.
        Returns (best_answer_text, best_score).
        """
        num = self.candidates_per_step
        answer_candidates = self.api_client.generate_texts(
            prompt=trajectory,
            n=num,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        scores = self.scorer.score_candidates(trajectory, answer_candidates)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best = answer_candidates[best_idx] or ""

        # Ensure explicit <Answer>: tag; if missing, try to extract a number and format
        low = best.lower()
        if "<answer>:" not in low:
            import re as _re
            m = _re.search(r"-?\d+(?:\.\d+)?", best)
            if m:
                best = f"\n<Answer>: {m.group(0)}"
            else:
                # Fallback: leave a tag to be consistent
                best = "\n<Answer>:"

        return best, float(scores[best_idx])

    def cleanup(self):
        try:
            self.scorer.cleanup()
        except Exception:
            pass



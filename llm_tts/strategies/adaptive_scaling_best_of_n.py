import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from llm_tts.generators import (
    StepCandidate,
    StepCandidateGeneratorThroughAPI,
    StepCandidateGeneratorThroughHuggingface,
    convert_trajectory_to_string,
)
from llm_tts.generators.vllm import CompletionReason
from llm_tts.utils import extract_answer

if TYPE_CHECKING:
    from llm_tts.generators import VLLMStepGenerator

from llm_tts.scale_discriminator import ScaleDiscriminator

from .strategy_base import StrategyBase, count_thinking_and_response_steps

log = logging.getLogger(__name__)


class AdaptiveScalingBestOfN(StrategyBase):
    """
    Adaptive scaling online best-of-n strategy.
    """

    def __init__(
        self,
        scorer,
        candidates_per_step: int,
        max_steps: int,
        step_generator: Union[
            StepCandidateGeneratorThroughAPI,
            StepCandidateGeneratorThroughHuggingface,
            "VLLMStepGenerator",
        ],
        scaling_rate: float = 0.9,
        momentum_rate: float = 0.9,
        adaptive_scaling_method: str = "momentum",
        batch_generation: bool = True,
        batch_size: int = 1000,
    ):
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.scorer = scorer
        self.step_generator = step_generator
        self.scaling_rate = scaling_rate
        self.momentum_rate = momentum_rate
        self.adaptive_scaling_method = adaptive_scaling_method
        kwargs = {}
        kwargs["momentum_rate"] = momentum_rate
        kwargs["scaling_rate"] = scaling_rate
        self.scale_discriminator = ScaleDiscriminator(
            criterion=adaptive_scaling_method, **kwargs
        )
        self.batch_generation = batch_generation
        self.batch_size = batch_size
        self.output_dir = None

    def generate_trajectory(
        self, request: List[Dict[str, str]], sample_idx: int = 0
    ) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using specified criterion.

        Args:
            prompt: Initial prompt/question

        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - completed: Whether trajectory reached completion
        """

        trajectory = []
        selected_steps = []
        validity_scores = []
        for step_num in range(self.max_steps):
            log.info(f"\n=== Step {step_num} ===")

            candidates = self.step_generator(
                request,
                trajectory=trajectory,
                candidates_per_step=1,
            )

            if not candidates:
                log.info("No candidates generated, stopping")
                break

            # Score candidates
            candidate_validity_scores = self.scorer.score_candidates(
                request, candidates
            )
            selected_candidate = candidates[0]
            cur_signal = candidate_validity_scores[0]

            log.info(f"Current signal: {cur_signal}")
            log.info(f"Current candidate: {selected_candidate.text}")

            # Decide whether to scale at this step
            if self.scale_discriminator.should_scale(cur_signal):
                log.info("Scaling step - generating new candidates")
                candidates = self.step_generator(
                    request,
                    trajectory=trajectory,
                    candidates_per_step=self.candidates_per_step,
                )
                all_candidate_scores = self.scorer.score_candidates(request, candidates)
                # Select best candidate
                best_idx, selected_candidate = self._select_best_candidate(
                    candidates, all_candidate_scores
                )
                cur_signal = all_candidate_scores[best_idx]

            self.scale_discriminator.update(1 - cur_signal)
            validity_scores.append(cur_signal)
            # Update trajectory
            trajectory.append(selected_candidate)
            selected_steps.append(selected_candidate)

            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                # Get completion reason from candidate
                completion_reason = None
                if selected_candidate.other_data:
                    completion_reason = selected_candidate.other_data.get(
                        "completion_reason"
                    )

                # If stopped at EOS, response is already complete (e.g., Qwen2.5-Math with \boxed{})
                if completion_reason == CompletionReason.EOS_PATTERN:
                    log.info("Stopped at EOS, response already complete")
                    break

                log.info("Answer pattern detected in step")
                if not self._has_answer_content(selected_candidate):
                    log.info("Answer content missing, generating final answer")
                    trajectory.pop()
                    selected_steps.pop()
                    final_answer, final_validity = self._generate_final_answer(
                        request, trajectory
                    )
                    trajectory.append(final_answer)
                    selected_steps.append(final_answer)
                    validity_scores.append(final_validity)
                break

        if not selected_candidate.is_trajectory_complete:
            final_answer, final_validity = self._generate_final_answer(
                request, trajectory
            )
            trajectory.append(final_answer)
            selected_steps.append(final_answer)
            validity_scores.append(final_validity)

        self.scale_discriminator.reset()

        # Extract answer from trajectory
        final_trajectory = convert_trajectory_to_string(trajectory)
        extracted = extract_answer(final_trajectory)

        # Finalize and get token statistics
        self.step_generator.finalize_sample_stats()
        token_stats = self.step_generator.get_sample_stats()
        log.info(
            f"Sample token stats: "
            f"total_tokens={token_stats['total_tokens_this_sample']:,}, "
            f"input_tokens={token_stats.get('input_tokens', 0):,}, "
            f"output_tokens={token_stats.get('output_tokens', 0):,}, "
            f"generations={token_stats['generation_count']}"
            + (f", tflops={token_stats['tflops']:.3f}" if token_stats["tflops"] else "")
        )
        # Count thinking and response steps separately
        thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
            selected_steps
        )

        return {
            "trajectory": final_trajectory,
            "extracted_answer": extracted,
            "steps": selected_steps,
            "validity_scores": validity_scores,
            "completed": len(selected_steps) > 0,
            "thinking_num_steps": thinking_num_steps,
            "response_num_steps": response_num_steps,
            "token_stats": token_stats,
        }

    def _select_best_candidate(self, candidates: List, scores: List[float]) -> tuple:
        """Select the best candidate based on scores"""

        # Higher validity is better
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return best_idx, candidates[best_idx]

    def _generate_final_answer(
        self, chat: List[Dict[str, str]], trajectory: List[StepCandidate]
    ) -> tuple:
        """Generate and select best final answer based on criterion"""

        # Generate answer candidates in batches if needed
        answer_candidates = self.step_generator.generate_answer_candidates(
            chat, trajectory=trajectory, candidates_per_step=self.candidates_per_step
        )

        # Score answer candidates
        answer_validity_scores = self.scorer.score_candidates(chat, answer_candidates)

        # Select best answer based on criterion
        best_idx, _ = self._select_best_candidate(
            answer_candidates, answer_validity_scores
        )

        log.info(f"Generated {len(answer_candidates)} answer candidates")
        log.info(f"Selected answer {best_idx}")
        log.info(f"Validity: {answer_validity_scores[best_idx]:.3f}")
        log.info(f"Text: {answer_candidates[best_idx].text}")

        return answer_candidates[best_idx], answer_validity_scores[best_idx]

    def generate_trajectory_mini_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_idxs: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched version of generate_trajectory that runs multiple samples "online" in parallel.

        Key idea for vLLM speed:
        - At each step, we only call the step_generator ONCE for all active samples.
        - This lets vLLM batch prompts efficiently.

        Args:
            requests: list of chat message lists, one per sample
            sample_idxs: list of sample ids for logging/output file names; defaults to 0..B-1

        Returns:
            List of per-sample outputs (same schema as generate_trajectory()).
        """
        B = len(requests)
        if sample_idxs is None:
            sample_idxs = list(range(B))
        assert len(sample_idxs) == B, "sample_idxs must have same length as requests"

        # --- Per-sample state ---
        trajectories: List[List[StepCandidate]] = [[] for _ in range(B)]
        selected_steps: List[List[StepCandidate]] = [[] for _ in range(B)]
        validity_scores: List[List[float]] = [[] for _ in range(B)]
        completed: List[bool] = [False for _ in range(B)]
        # We need selected_candidate per sample for post-loop logic
        last_selected: List[Optional[StepCandidate]] = [None for _ in range(B)]

        # Optional per-sample logs (keep local to avoid shared mutable state)
        steps_logs: List[List[Dict[str, Any]]] = [[] for _ in range(B)]

        # Reset token tracking (batch-wide). If your generator tracks per-sample stats internally,
        # consider extending it to support batch reset/finalize per sample.
        self.step_generator.reset_sample_stats()

        # ---- Batched helpers (duck-typing) ----
        def _gen_step_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_trajs: List[List[StepCandidate]],
            candidates_per_step: int,
        ) -> List[List[StepCandidate]]:
            """
            Returns list-of-list of candidates aligned to active samples.
            Prefer true batched generator method for vLLM throughput.
            """
            # Preferred: a dedicated batch method on generator
            if hasattr(self.step_generator, "generate_step_candidates_batch"):
                return self.step_generator.generate_step_candidates_batch(
                    active_reqs,
                    trajectories=active_trajs,
                    candidates_per_step=candidates_per_step,
                )

            # Alternative: generator __call__ supports list input (some implementations do)
            try:
                out = self.step_generator(
                    active_reqs,
                    trajectory=active_trajs,
                    candidates_per_step=candidates_per_step,
                )
                # Expect list[list[StepCandidate]]
                if out and isinstance(out[0], list):
                    return out
            except Exception:
                log.info("Error generating step candidates")
                pass

            # Fallback: loop (no vLLM batching)
            return [
                self.step_generator(
                    req,
                    trajectory=traj,
                    candidates_per_step=candidates_per_step,
                )
                for req, traj in zip(active_reqs, active_trajs)
            ]

        def _score_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_cands: List[List[StepCandidate]],
            active_trajs: List[List[StepCandidate]],
        ) -> List[List[float]]:
            """
            Returns list-of-list of scores aligned to active samples.
            """
            if hasattr(self.scorer, "score_candidates_batch"):
                return self.scorer.score_candidates_batch(
                    active_reqs, active_cands, trajectories=active_trajs
                )

            # Fallback: loop per sample (still ok; main win is batched generation)
            scores = []
            for req, cands, traj in zip(active_reqs, active_cands, active_trajs):
                scores.append(self.scorer.score_candidates(req, cands, trajectory=traj))
            return scores

        def _gen_answer_candidates_batch(
            active_reqs: List[List[Dict[str, str]]],
            active_trajs: List[List[StepCandidate]],
        ) -> List[List[StepCandidate]]:
            """
            Batched final answer generation (preferred for vLLM).
            """
            if hasattr(self.step_generator, "generate_answer_candidates_batch"):
                return self.step_generator.generate_answer_candidates_batch(
                    active_reqs,
                    trajectories=active_trajs,
                    candidates_per_step=self.candidates_per_step,
                )

            # Fallback: loop
            return [
                self.step_generator.generate_answer_candidates(
                    req, trajectory=traj, candidates_per_step=self.candidates_per_step
                )
                for req, traj in zip(active_reqs, active_trajs)
            ]

        def _select_best(scores: List[float]) -> int:
            return max(range(len(scores)), key=lambda i: scores[i])

        def _new_scale_discriminator() -> ScaleDiscriminator:
            return ScaleDiscriminator(
                criterion=self.adaptive_scaling_method,
                momentum_rate=self.momentum_rate,
                scaling_rate=self.scaling_rate,
            )

        # ---- Main online loop (batched across samples) ----
        scale_discriminators: List[ScaleDiscriminator] = [
            _new_scale_discriminator() for _ in range(B)
        ]
        needs_final_answer: List[bool] = [False for _ in range(B)]

        for step_num in range(self.max_steps):
            # Which samples are still active?
            active_idx = [i for i in range(B) if not completed[i]]
            if not active_idx:
                break

            active_reqs = [requests[i] for i in active_idx]
            active_trajs = [trajectories[i] for i in active_idx]

            # 1) Generate a single candidate first for each active sample
            step_candidates_batch = _gen_step_candidates_batch(
                active_reqs, active_trajs, candidates_per_step=1
            )
            # log.info(f"Step candidates batch: {step_candidates_batch}")
            # Handle samples that produced no candidates
            # (mark completed; they'll go to final answer later if needed)
            filtered_active_idx = []
            filtered_reqs = []
            filtered_trajs = []
            filtered_cands = []
            for j, bi in enumerate(active_idx):
                cands = step_candidates_batch[j]
                if not cands:
                    completed[bi] = True
                    last_selected[bi] = None
                else:
                    filtered_active_idx.append(bi)
                    filtered_reqs.append(requests[bi])
                    filtered_trajs.append(trajectories[bi])
                    filtered_cands.append(cands)

            if not filtered_active_idx:
                break

            # 2) Score the single candidates (batched if possible)
            scores_batch = _score_candidates_batch(
                filtered_reqs, filtered_cands, filtered_trajs
            )
            log.info(f"Scores batch: {scores_batch}")
            # 3) Decide which samples should scale
            scale_idx = []
            scale_reqs = []
            scale_trajs = []
            for j, bi in enumerate(filtered_active_idx):
                cur_signal = scores_batch[j][0]
                validity_scores[bi].append(cur_signal)
                if scale_discriminators[bi].should_scale(cur_signal):
                    scale_idx.append(bi)
                    scale_reqs.append(requests[bi])
                    scale_trajs.append(trajectories[bi])

            # store current results
            # for j, bi in enumerate(active_idx):
            #     cands = step_candidates_batch[j]
            #     trajectories[bi].append(cands[0])
            #     selected_steps[bi].append(cands[0])
            #     last_selected[bi] = cands[0]

            # 4) For samples that scale, generate N more candidates and rescore
            scaled_candidates = {}
            scaled_scores = {}
            if scale_idx:
                # import pdb; pdb.set_trace();
                scale_cands_batch = _gen_step_candidates_batch(
                    scale_reqs,
                    scale_trajs,
                    candidates_per_step=self.candidates_per_step,
                )
                scale_scores_batch = _score_candidates_batch(
                    scale_reqs, scale_cands_batch, scale_trajs
                )
                for j, bi in enumerate(scale_idx):
                    scaled_candidates[bi] = scale_cands_batch[j]
                    scaled_scores[bi] = scale_scores_batch[j]

            # 5) Select candidate per sample and update states
            for j, bi in enumerate(filtered_active_idx):
                if bi in scaled_candidates and scaled_candidates[bi]:
                    cands = scaled_candidates[bi]
                    scores = scaled_scores[bi]
                    best_idx = _select_best(scores)
                    chosen = cands[best_idx]
                    cur_signal = scores[best_idx]
                else:
                    chosen = filtered_cands[j][0]
                    cur_signal = scores_batch[j][0]

                scale_discriminators[bi].update(1 - cur_signal)
                trajectories[bi].append(chosen)
                selected_steps[bi].append(chosen)
                validity_scores[bi].append(cur_signal)
                last_selected[bi] = chosen

                # Completion checks (mirror single-sample behavior)
                if chosen.is_trajectory_complete:
                    completion_reason = None
                    if chosen.other_data:
                        completion_reason = chosen.other_data.get("completion_reason")

                    # If stopped at EOS pattern, treat as complete
                    if completion_reason == CompletionReason.EOS_PATTERN:
                        completed[bi] = True
                        continue

                    # If answer pattern detected but answer content missing,
                    # remove step and generate final answer later.
                    if not self._has_answer_content(chosen):
                        trajectories[bi].pop()
                        selected_steps[bi].pop()
                        validity_scores[bi].pop()
                        needs_final_answer[bi] = True
                    completed[bi] = True

        # ---- Final answer for samples that need it (batched) ----
        # Need final answer if:
        # - never completed with a valid trajectory, OR
        # - last selected exists but is not trajectory complete, OR
        # - last step signaled completion but missing answer content
        to_finalize: List[int] = []
        for i in range(B):
            if len(selected_steps[i]) == 0:
                to_finalize.append(i)
                continue
            if last_selected[i] is None:
                to_finalize.append(i)
                continue
            if not last_selected[i].is_trajectory_complete:
                to_finalize.append(i)
                continue
            if needs_final_answer[i]:
                to_finalize.append(i)
        # import pdb; pdb.set_trace();
        if to_finalize:
            fin_reqs = [requests[i] for i in to_finalize]
            fin_trajs = [trajectories[i] for i in to_finalize]

            answer_cands_batch = _gen_answer_candidates_batch(fin_reqs, fin_trajs)

            # Score final answers (batched if scorer supports)
            answer_scores_batch = _score_candidates_batch(
                fin_reqs, answer_cands_batch, fin_trajs
            )

            for j, bi in enumerate(to_finalize):
                a_cands = answer_cands_batch[j]
                a_scores = answer_scores_batch[j]
                if not a_cands:
                    # If generation failed, leave as-is (rare)
                    continue
                best_idx = _select_best(a_scores)
                chosen = a_cands[best_idx]
                trajectories[bi].append(chosen)
                selected_steps[bi].append(chosen)
                validity_scores[bi].append(a_scores[best_idx])
                last_selected[bi] = chosen

        # ---- Finalize stats & build outputs ----
        self.step_generator.finalize_sample_stats(num_samples=B)
        token_stats = self.step_generator.get_sample_stats()

        outputs: List[Dict[str, Any]] = []
        for bi in range(B):
            final_trajectory = convert_trajectory_to_string(trajectories[bi])
            extracted = extract_answer(final_trajectory)

            # Save per-sample final logs
            if self.output_dir:
                prev_idx = getattr(self, "_current_sample_idx", None)
                prev_log = getattr(self, "_steps_log", None)
                try:
                    self._current_sample_idx = sample_idxs[bi]
                    self._steps_log = steps_logs[bi]
                    self._save_steps_log()
                    self._save_trajectory_log(final_trajectory)
                finally:
                    self._current_sample_idx = prev_idx
                    self._steps_log = prev_log

            thinking_num_steps, response_num_steps = count_thinking_and_response_steps(
                selected_steps[bi]
            )

            outputs.append(
                {
                    "trajectory": final_trajectory,
                    "extracted_answer": extracted,
                    "steps": selected_steps[bi],
                    "thinking_num_steps": thinking_num_steps,
                    "response_num_steps": response_num_steps,
                    "validity_scores": validity_scores[bi],
                    "completed": len(selected_steps[bi]) > 0,
                    "token_stats": token_stats,  # batch-wide; see note below
                }
            )

        return outputs

    def generate_trajectories_batch(
        self,
        requests: List[List[Dict[str, str]]],
        sample_idxs: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batched version of generate_trajectory that runs multiple samples "online" in parallel.
        """
        # split requests into mini-batches
        sample_idxs = list(range(len(requests)))
        mini_batches = [
            requests[i : min(i + self.batch_size, len(requests))]
            for i in range(0, len(requests), self.batch_size)
        ]
        sample_idxs = [
            sample_idxs[i : min(i + self.batch_size, len(sample_idxs))]
            for i in range(0, len(sample_idxs), self.batch_size)
        ]
        outputs = []
        for i, mini_batch in enumerate(mini_batches):
            log.info(f"Generating mini-batch {i+1} of {len(mini_batches)}")
            outputs.extend(
                self.generate_trajectory_mini_batch(mini_batch, sample_idxs[i])
            )
            log.info(f"Sample output: {outputs[i]['extracted_answer']}")

        return outputs

    def cleanup(self):
        """Clean up resources"""

        self.scorer.cleanup()
        self.scale_discriminator.reset()

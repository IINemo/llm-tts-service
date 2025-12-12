"""
Test ThinkingMarkerDetector against reference marker_semantic_v2 output.

The reference data comes from:
- Source: outputs/2025-12-11/aime2025_thinking_vllm_22-50-50/results.json
- Reference: outputs/2025-12-11/aime2025_thinking_vllm_22-50-50/by_detector/marker_semantic_v2.json

marker_semantic_v2 config (from docs/step_boundary_detectors.md):
- use_sequence=True
- use_conclusion=True
- use_thinking=True
- use_verification=True
- use_structure=False
- use_reasoning=True
- use_sentence_start=False
- use_correction=False
- min_step_chars=100
- max_step_chars=600
"""

import json
import os
import re
from pathlib import Path

import pytest

from llm_tts.step_boundary_detectors import ThinkingMarkerDetector


# Paths to reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILE = FIXTURES_DIR / "aime2025_thinking_traces.json"


def get_marker_semantic_v2_detector() -> ThinkingMarkerDetector:
    """
    Create detector with marker_semantic_v2 config.

    marker_semantic_v2 = marker_semantic + selective additions:
    - for example
    - given that
    - similarly

    Note: use_reasoning=False because REASONING_MARKERS includes too many
    patterns (we have, we can, we need, etc.) that cause over-splitting.
    We add only the 3 documented markers via custom_markers.
    """
    return ThinkingMarkerDetector(
        use_sequence=True,
        use_conclusion=True,
        use_thinking=True,
        use_verification=True,
        use_structure=False,
        use_reasoning=False,  # Don't use full REASONING_MARKERS
        use_sentence_start=False,
        use_correction=False,
        custom_markers=[  # Only the 3 selective additions for v2
            r"\bfor example\b",
            r"\bgiven that\b",
            r"\bsimilarly\b",
        ],
        min_step_chars=100,
        max_step_chars=600,
    )


def load_fixture_data():
    """Load fixture with thinking traces and expected results."""
    if not FIXTURE_FILE.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_FILE}")
    with open(FIXTURE_FILE) as f:
        return json.load(f)


def extract_thinking_content(response: str) -> str:
    """Extract content from <think> tags."""
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


class TestMarkerSemanticV2Offline:
    """Test offline step detection (detect_steps method)."""

    @pytest.fixture
    def detector(self):
        return get_marker_semantic_v2_detector()

    @pytest.fixture
    def fixture_data(self):
        return load_fixture_data()

    def test_detector_config_matches(self, detector, fixture_data):
        """Verify detector config matches reference."""
        ref_config = fixture_data["detector_config"]
        assert detector.min_step_chars == ref_config["min_step_chars"]
        assert detector.max_step_chars == ref_config["max_step_chars"]

    def test_steps_match_reference(self, detector, fixture_data):
        """Test that detected steps match reference exactly."""

        def normalize_input(text: str) -> str:
            """Normalize input text before detection (as production should do)."""
            # Convert multiple newlines to single for consistent detection
            return re.sub(r"\n{2,}", "\n", text)

        for sample in fixture_data["samples"]:
            thinking = sample["thinking_content"]
            ref_steps = sample["reference_steps"]

            if not thinking:
                continue

            # Normalize input for consistent detection
            normalized_thinking = normalize_input(thinking)

            # Run detector on normalized input
            actual_steps = detector.detect_steps(normalized_thinking)

            # Step counts must match exactly
            assert len(actual_steps) == len(ref_steps), (
                f"Sample {sample['index']}: Step count mismatch. "
                f"Expected {len(ref_steps)}, got {len(actual_steps)}"
            )

            # Each step must match exactly
            for i, (actual, expected) in enumerate(zip(actual_steps, ref_steps)):
                # Normalize both for comparison (whitespace may differ slightly)
                actual_norm = re.sub(r"\s+", " ", actual).strip()
                expected_norm = re.sub(r"\s+", " ", expected).strip()
                assert actual_norm == expected_norm, (
                    f"Sample {sample['index']}, Step {i}: Content mismatch.\n"
                    f"Expected: {repr(expected[:80])}...\n"
                    f"Actual:   {repr(actual[:80])}..."
                )

    def test_total_step_count(self, detector, fixture_data):
        """Test total step count matches reference exactly."""

        def normalize_input(text: str) -> str:
            return re.sub(r"\n{2,}", "\n", text)

        ref_total = fixture_data["summary_stats"]["total_steps"]

        total_steps = 0
        for sample in fixture_data["samples"]:
            thinking = sample["thinking_content"]
            if thinking:
                steps = detector.detect_steps(normalize_input(thinking))
                total_steps += len(steps)

        assert total_steps == ref_total, (
            f"Expected {ref_total} total steps, got {total_steps}"
        )

    def test_avg_step_length(self, detector, fixture_data):
        """Test average step length matches reference."""

        def normalize_input(text: str) -> str:
            return re.sub(r"\n{2,}", "\n", text)

        ref_avg = fixture_data["summary_stats"]["avg_step_length"]

        all_steps = []
        for sample in fixture_data["samples"]:
            thinking = sample["thinking_content"]
            if thinking:
                steps = detector.detect_steps(normalize_input(thinking))
                all_steps.extend(steps)

        if not all_steps:
            pytest.skip("No steps extracted")

        avg_length = sum(len(s) for s in all_steps) / len(all_steps)

        # Allow small tolerance for floating point and whitespace normalization
        tolerance = 1.0
        assert abs(avg_length - ref_avg) <= tolerance, (
            f"Expected avg step length ~{ref_avg:.1f}, got {avg_length:.1f}"
        )


class TestMarkerSemanticV2Online:
    """Test online step detection (is_step_complete method)."""

    @pytest.fixture
    def detector(self):
        return get_marker_semantic_v2_detector()

    @pytest.fixture
    def fixture_data(self):
        return load_fixture_data()

    def test_online_produces_meaningful_steps(self, detector, fixture_data):
        """Test that online detection produces meaningful step boundaries."""
        # Use first sample
        thinking = fixture_data["samples"][0]["thinking_content"]

        if not thinking:
            pytest.skip("No thinking content")

        detector.reset_online_state()

        # Simulate character-by-character generation
        steps = []
        current_step_start = 0

        for i in range(len(thinking)):
            text_so_far = thinking[: i + 1]

            is_complete = detector.is_step_complete(text_so_far)
            is_traj_complete = detector.is_trajectory_complete(text_so_far)

            if is_complete or is_traj_complete:
                step_text = thinking[current_step_start : i + 1]
                steps.append(step_text)
                detector.mark_step_complete(text_so_far)
                current_step_start = i + 1

                if is_traj_complete:
                    break

        # Verify we got reasonable steps
        assert len(steps) >= 1, "Should extract at least one step"

        # Verify step lengths are reasonable
        for i, step in enumerate(steps[:-1]):  # Exclude final step (may be short)
            assert len(step) >= detector.min_step_chars * 0.5, (
                f"Step {i} too short: {len(step)} chars"
            )

    def test_online_detects_trajectory_completion(self, detector, fixture_data):
        """Test that trajectory completion is detected."""
        thinking = fixture_data["samples"][0]["thinking_content"]

        if not thinking:
            pytest.skip("No thinking content")

        # Add </think> marker
        full_text = f"<think>{thinking}</think>"

        detector.reset_online_state()
        assert detector.is_trajectory_complete(full_text)

    def test_online_vs_offline_step_count(self, detector, fixture_data):
        """Compare online vs offline step counts (should be similar).

        Simulates real online generation where content past the boundary
        is discarded. After detecting a step, we continue from the boundary
        position (not from where we currently are in iteration).
        """

        def normalize_input(text: str) -> str:
            return re.sub(r"\n{2,}", "\n", text)

        thinking = fixture_data["samples"][0]["thinking_content"]

        if not thinking:
            pytest.skip("No thinking content")

        # Normalize input (same as offline tests)
        thinking = normalize_input(thinking)

        # Offline detection
        offline_steps = detector.detect_steps(thinking)

        # Online detection (fresh detector)
        detector.reset_online_state()
        online_step_count = 0

        for i in range(len(thinking)):
            text_so_far = thinking[: i + 1]
            is_complete = detector.is_step_complete(text_so_far)
            is_traj_complete = detector.is_trajectory_complete(text_so_far)

            if is_complete or is_traj_complete:
                online_step_count += 1
                detector.mark_step_complete(text_so_far)
                if is_traj_complete:
                    break

        # Online and offline use different merging strategies:
        # - Offline: splits at sentence boundaries, then merges consecutive short parts
        # - Online: accumulates until min_step_chars, then splits at first sentence-boundary marker
        # Allow ~5% difference due to these algorithmic differences
        tolerance = max(5, int(len(offline_steps) * 0.05))
        assert abs(online_step_count - len(offline_steps)) <= tolerance, (
            f"Online ({online_step_count}) vs Offline ({len(offline_steps)}) "
            f"step count differs by more than {tolerance} ({100*abs(online_step_count-len(offline_steps))/len(offline_steps):.1f}%)"
        )


class TestMarkerDetectorConfigs:
    """Test different marker detector configurations."""

    def test_marker_all_produces_more_steps(self):
        """marker_all should produce more steps than marker_semantic_v2."""
        detector_v2 = get_marker_semantic_v2_detector()
        detector_all = ThinkingMarkerDetector(
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
            use_structure=True,  # Enable structure markers
            use_reasoning=True,
            use_sentence_start=True,  # Enable sentence start
            use_correction=True,
            min_step_chars=100,
            max_step_chars=600,
        )

        test_text = """
        First, let me analyze this problem. The key insight is understanding the constraints.

        But wait, I need to reconsider. However, the previous approach might work.

        So, the solution involves computing the sum. Therefore, we get the answer.

        Let me verify: 1 + 2 = 3. Yes, that's correct.
        """

        steps_v2 = detector_v2.detect_steps(test_text)
        steps_all = detector_all.detect_steps(test_text)

        # marker_all should produce more or equal steps
        assert len(steps_all) >= len(steps_v2), (
            f"marker_all ({len(steps_all)}) should produce >= steps than "
            f"marker_semantic_v2 ({len(steps_v2)})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

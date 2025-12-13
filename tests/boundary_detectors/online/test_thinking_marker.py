"""
Test ThinkingMarkerDetector online detection (is_step_complete method).

Tests that online detection produces results matching offline detection.
"""

import json
import re
from pathlib import Path

import pytest

from llm_tts.step_boundary_detectors import ThinkingMarkerDetector


# Paths to reference data
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
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

    def test_online_matches_offline_exactly(self, detector, fixture_data):
        """Test that online detection produces EXACT same results as offline.

        After fixing the online algorithm, it should produce identical
        step boundaries as the offline detect_steps() method.
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
        online_steps = []

        for i in range(len(thinking)):
            text_so_far = thinking[: i + 1]
            if detector.is_step_complete(text_so_far):
                step_text = detector.extract_step_text(text_so_far)
                if step_text.strip():
                    online_steps.append(step_text)
                detector.mark_step_complete(text_so_far)
                if detector.is_trajectory_complete(text_so_far):
                    break

        # Add remaining text
        remaining = thinking[detector._last_step_end_pos:].strip()
        if remaining:
            online_steps.append(remaining)

        # Step counts must match exactly
        assert len(online_steps) == len(offline_steps), (
            f"Online ({len(online_steps)}) vs Offline ({len(offline_steps)}) "
            f"step count mismatch"
        )

        # Each step must match (after whitespace normalization)
        for i, (online, offline) in enumerate(zip(online_steps, offline_steps)):
            online_norm = re.sub(r"\s+", " ", online).strip()
            offline_norm = re.sub(r"\s+", " ", offline).strip()
            assert online_norm == offline_norm, (
                f"Step {i}: Content mismatch.\n"
                f"Online: {online_norm[:80]}...\n"
                f"Offline: {offline_norm[:80]}..."
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

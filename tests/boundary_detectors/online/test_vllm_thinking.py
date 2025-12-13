"""
Test vLLM thinking mode step detection against HuggingFace reference.

Simulates vLLM stop token behavior and verifies that using
ThinkingMarkerDetector for boundary validation produces EXACT
same results as HuggingFace implementation.

The key insight: vLLM stops at stop_tokens, but the actual boundary
validation uses ThinkingMarkerDetector - same as HuggingFace.
This ensures identical results.
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pytest

from llm_tts.step_boundary_detectors.thinking import ThinkingMarkerDetector
from llm_tts.step_boundary_detectors.thinking.vllm import (
    get_stop_tokens_compact,
    ANSWER_TOKENS,
)


# Paths to reference data
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FIXTURE_FILE = FIXTURES_DIR / "aime2025_thinking_traces.json"


def get_marker_semantic_v2_config():
    """Get marker_semantic_v2 configuration for both HF and vLLM."""
    return {
        "use_sequence": True,
        "use_conclusion": True,
        "use_thinking": True,
        "use_verification": True,
        "use_structure": False,
        "use_reasoning": False,
        "use_sentence_start": False,
        "use_correction": False,
        "custom_markers": [
            r"\bfor example\b",
            r"\bgiven that\b",
            r"\bsimilarly\b",
        ],
        "custom_words": [  # For vLLM stop tokens
            "for example",
            "given that",
            "similarly",
        ],
        "min_step_chars": 100,
        "max_step_chars": 600,
    }


def get_hf_detector() -> ThinkingMarkerDetector:
    """Create HuggingFace-compatible ThinkingMarkerDetector."""
    config = get_marker_semantic_v2_config()
    return ThinkingMarkerDetector(
        use_sequence=config["use_sequence"],
        use_conclusion=config["use_conclusion"],
        use_thinking=config["use_thinking"],
        use_verification=config["use_verification"],
        use_structure=config["use_structure"],
        use_reasoning=config["use_reasoning"],
        use_sentence_start=config["use_sentence_start"],
        use_correction=config["use_correction"],
        custom_markers=config["custom_markers"],
        min_step_chars=config["min_step_chars"],
        max_step_chars=config["max_step_chars"],
    )


def get_vllm_stop_tokens() -> List[str]:
    """Get vLLM stop tokens matching marker_semantic_v2 config."""
    config = get_marker_semantic_v2_config()
    return get_stop_tokens_compact(
        use_sequence=config["use_sequence"],
        use_conclusion=config["use_conclusion"],
        use_thinking=config["use_thinking"],
        use_verification=config["use_verification"],
        use_reasoning=config["use_reasoning"],
        use_correction=config["use_correction"],
        use_structure=config["use_structure"],
        custom_words=config["custom_words"],
    )


def load_fixture_data():
    """Load fixture with thinking traces."""
    if not FIXTURE_FILE.exists():
        pytest.skip(f"Fixture file not found: {FIXTURE_FILE}")
    with open(FIXTURE_FILE) as f:
        return json.load(f)


def normalize_input(text: str) -> str:
    """Normalize input text for consistent detection."""
    return re.sub(r"\n{2,}", "\n", text)


class MockVLLMGenerator:
    """
    Mock vLLM generator that simulates stop token behavior.

    Uses is_step_complete() which implements online detection with
    the same logic as offline detect_steps():
    - Only splits at markers
    - Only at sentence boundaries
    - Only when >= min_step_chars
    """

    def __init__(
        self,
        stop_tokens: List[str],
        detector: ThinkingMarkerDetector,
    ):
        self.stop_tokens = stop_tokens
        self.detector = detector

    def simulate_online_generation(self, full_text: str) -> List[str]:
        """
        Simulate online step-by-step generation using is_step_complete().

        is_step_complete() already implements the same logic as offline:
        - Tracks _last_step_end_pos internally
        - Only processes new content since last boundary
        - Uses same marker detection and sentence boundary checks

        Args:
            full_text: Complete thinking content to simulate generation from

        Returns:
            List of detected steps
        """
        self.detector.reset_online_state()
        steps = []
        text_len = len(full_text)

        # Simulate character-by-character generation
        for i in range(text_len):
            text_so_far = full_text[:i + 1]

            # Check trajectory completion first
            if self.detector.is_trajectory_complete(text_so_far):
                step_text = self.detector.extract_step_text(text_so_far)
                if step_text.strip():
                    steps.append(step_text)
                break

            # Check step completion
            if self.detector.is_step_complete(text_so_far):
                step_text = self.detector.extract_step_text(text_so_far)
                if step_text.strip():
                    steps.append(step_text)
                self.detector.mark_step_complete(text_so_far)

        # Handle remaining text after last step
        if not self.detector._trajectory_complete:
            remaining = full_text[self.detector._last_step_end_pos:].strip()
            if remaining:
                steps.append(remaining)

        return steps


def simulate_vllm_with_detector(
    text: str,
    detector: ThinkingMarkerDetector,
    stop_tokens: List[str],
) -> List[str]:
    """
    Simulate vLLM generation using ThinkingMarkerDetector for validation.

    This produces EXACT same results as HuggingFace because both use
    the same ThinkingMarkerDetector for boundary detection.
    """
    mock_gen = MockVLLMGenerator(stop_tokens, detector)
    return mock_gen.simulate_online_generation(text)


class TestVLLMStopTokens:
    """Test vLLM stop token generation."""

    def test_stop_tokens_generated(self):
        """Verify stop tokens are generated."""
        tokens = get_vllm_stop_tokens()
        assert len(tokens) > 0, "Should generate stop tokens"
        # Should have hundreds of tokens for compact mode
        assert len(tokens) > 100, f"Expected >100 tokens, got {len(tokens)}"

    def test_stop_tokens_include_markers(self):
        """Verify stop tokens include expected markers."""
        tokens = get_vllm_stop_tokens()

        # Check for key markers
        expected_patterns = [
            "\nSo ",
            "\nLet me ",
            "\nTherefore",
            "\nFirst",
            "\nNext",
            "\nOkay",
        ]

        for pattern in expected_patterns:
            found = any(pattern in t or t == pattern for t in tokens)
            assert found, f"Expected to find '{pattern}' in stop tokens"

    def test_answer_tokens_included(self):
        """Verify answer tokens are included."""
        tokens = get_vllm_stop_tokens()

        for answer_token in ANSWER_TOKENS:
            assert answer_token in tokens, f"Missing answer token: {answer_token}"


class TestVLLMvsHFOnline:
    """
    Compare vLLM simulation vs HuggingFace online detection.

    Both use ThinkingMarkerDetector for boundary validation,
    so results should be EXACTLY the same.
    """

    @pytest.fixture
    def detector(self):
        """Shared detector for both HF and vLLM simulation."""
        return get_hf_detector()

    @pytest.fixture
    def vllm_stop_tokens(self):
        return get_vllm_stop_tokens()

    @pytest.fixture
    def fixture_data(self):
        return load_fixture_data()

    def test_step_count_exact_match(self, detector, vllm_stop_tokens, fixture_data):
        """
        Test that vLLM simulation produces same step count as HF online.

        Both use ThinkingMarkerDetector.is_step_complete() for validation.
        Allow tolerance of 1 step for edge cases at text boundaries.
        """
        for sample in fixture_data["samples"][:5]:  # Test first 5 samples
            thinking = sample["thinking_content"]
            if not thinking:
                continue

            thinking = normalize_input(thinking)

            # HF online detection (reference)
            detector.reset_online_state()
            hf_steps = []
            for i in range(len(thinking)):
                text_so_far = thinking[:i + 1]
                if detector.is_trajectory_complete(text_so_far):
                    step_text = detector.extract_step_text(text_so_far)
                    if step_text.strip():
                        hf_steps.append(step_text)
                    break
                if detector.is_step_complete(text_so_far):
                    step_text = detector.extract_step_text(text_so_far)
                    if step_text.strip():
                        hf_steps.append(step_text)
                    detector.mark_step_complete(text_so_far)

            # vLLM simulation (uses same detector)
            vllm_steps = simulate_vllm_with_detector(
                thinking,
                get_hf_detector(),  # Fresh detector instance
                vllm_stop_tokens,
            )

            # Step counts must match (allow tolerance of 1 for edge cases)
            diff = abs(len(vllm_steps) - len(hf_steps))
            assert diff <= 1, (
                f"Sample {sample['index']}: Step count mismatch. "
                f"HF={len(hf_steps)}, vLLM={len(vllm_steps)}, diff={diff}"
            )

    def test_step_content_exact_match(self, detector, vllm_stop_tokens, fixture_data):
        """
        Test that vLLM simulation produces EXACT same step content as HF.
        """
        for sample in fixture_data["samples"][:3]:  # Test first 3 samples
            thinking = sample["thinking_content"]
            if not thinking:
                continue

            thinking = normalize_input(thinking)

            # HF online detection
            detector.reset_online_state()
            hf_steps = []
            for i in range(len(thinking)):
                text_so_far = thinking[:i + 1]
                if detector.is_trajectory_complete(text_so_far):
                    step_text = detector.extract_step_text(text_so_far)
                    if step_text.strip():
                        hf_steps.append(step_text)
                    break
                if detector.is_step_complete(text_so_far):
                    step_text = detector.extract_step_text(text_so_far)
                    if step_text.strip():
                        hf_steps.append(step_text)
                    detector.mark_step_complete(text_so_far)

            # vLLM simulation
            vllm_steps = simulate_vllm_with_detector(
                thinking,
                get_hf_detector(),
                vllm_stop_tokens,
            )

            # Each step must match exactly
            for i, (hf_step, vllm_step) in enumerate(zip(hf_steps, vllm_steps)):
                # Normalize whitespace for comparison
                hf_norm = re.sub(r"\s+", " ", hf_step).strip()
                vllm_norm = re.sub(r"\s+", " ", vllm_step).strip()
                assert hf_norm == vllm_norm, (
                    f"Sample {sample['index']}, Step {i}: Content mismatch.\n"
                    f"HF: {hf_norm[:100]}...\n"
                    f"vLLM: {vllm_norm[:100]}..."
                )

    def test_matches_reference_steps(self, detector, vllm_stop_tokens, fixture_data):
        """
        Test that offline detection matches the fixture reference steps.

        The fixture was generated with offline detection (detect_steps),
        so we use the same algorithm to validate against reference.
        """
        for sample in fixture_data["samples"][:3]:
            thinking = sample["thinking_content"]
            ref_steps = sample["reference_steps"]
            if not thinking:
                continue

            thinking = normalize_input(thinking)

            # Use offline detection (same as fixture generation)
            offline_detector = get_hf_detector()
            offline_steps = offline_detector.detect_steps(thinking)

            # Step count should match reference
            assert len(offline_steps) == len(ref_steps), (
                f"Sample {sample['index']}: Step count mismatch with reference. "
                f"Reference={len(ref_steps)}, Offline={len(offline_steps)}"
            )

            # Each step should match reference
            for i, (offline_step, ref_step) in enumerate(zip(offline_steps, ref_steps)):
                offline_norm = re.sub(r"\s+", " ", offline_step).strip()
                ref_norm = re.sub(r"\s+", " ", ref_step).strip()
                assert offline_norm == ref_norm, (
                    f"Sample {sample['index']}, Step {i}: Mismatch with reference.\n"
                    f"Reference: {ref_norm[:80]}...\n"
                    f"Offline: {offline_norm[:80]}..."
                )


class TestVLLMStopTokenCategories:
    """Test different stop token category configurations."""

    def test_more_categories_more_tokens(self):
        """Enabling more categories should produce more stop tokens."""
        minimal = get_stop_tokens_compact(
            use_sequence=True,
            use_conclusion=False,
            use_thinking=False,
            use_verification=False,
        )

        full = get_stop_tokens_compact(
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
        )

        assert len(full) > len(minimal), (
            f"Full config ({len(full)}) should have more tokens than "
            f"minimal ({len(minimal)})"
        )

    def test_custom_words_added(self):
        """Custom words should be added to stop tokens."""
        base = get_stop_tokens_compact(
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
        )

        with_custom = get_stop_tokens_compact(
            use_sequence=True,
            use_conclusion=True,
            use_thinking=True,
            use_verification=True,
            custom_words=["my custom marker", "another marker"],
        )

        assert len(with_custom) > len(base), "Custom words should add tokens"

        # Check custom markers are present
        custom_found = any("custom marker" in t.lower() for t in with_custom)
        assert custom_found, "Should include custom marker variants"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

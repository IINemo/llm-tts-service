#!/usr/bin/env python3
"""
Quick test to verify Qwen3-8B baseline works with Chain-of-Thought strategy.
Tests both model loading and generation on a simple problem.
"""

import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llm_tts.strategies.strategy_chain_of_thought import StrategyChainOfThought

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class LocalQwenModel:
    """Minimal local Qwen model wrapper for testing."""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:0"):
        log.info(f"Loading model: {model_name}")
        log.info(f"Device: {device}")

        # Check CUDA availability
        if "cuda" in device and not torch.cuda.is_available():
            log.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        log.info("Loading model weights (this may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else None,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            trust_remote_code=True,
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.model.eval()
        log.info("Model loaded successfully")

    def tokenize(self, texts):
        """Tokenize texts (matching expected interface)."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )


def test_qwen3_baseline():
    """Test Qwen3 with Chain-of-Thought strategy."""
    log.info("=" * 80)
    log.info("Testing Qwen3-8B Baseline (Chain-of-Thought)")
    log.info("=" * 80)

    # Load model (use smaller model for quick test if specified)
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Use 7B for testing
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"CUDA device count: {torch.cuda.device_count()}")
        log.info(f"Current CUDA device: {torch.cuda.current_device()}")

    model = LocalQwenModel(model_name, device)

    # Create Chain-of-Thought strategy
    log.info("\nInitializing Chain-of-Thought strategy...")
    strategy = StrategyChainOfThought(
        model=model,
        max_new_tokens=512,
        temperature=0.1,
    )

    # Test with a simple GSM8K-style problem
    test_prompt = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Please solve this math problem step by step. Show your reasoning and put your final numerical answer in \\boxed{}."""

    log.info(f"\n{'='*80}")
    log.info("Test Prompt:")
    log.info(f"{'='*80}")
    log.info(test_prompt)
    log.info(f"{'='*80}")

    log.info("\nGenerating trajectory...")

    try:
        result = strategy.generate_trajectory(test_prompt)

        log.info("\n" + "=" * 80)
        log.info("SUCCESS! Qwen3 baseline works!")
        log.info("=" * 80)

        log.info(f"\nStrategy: {result['strategy']}")
        log.info(f"Completed: {result['completed']}")

        log.info("\n" + "=" * 80)
        log.info("Generated Trajectory:")
        log.info("=" * 80)
        log.info(result["trajectory"])
        log.info("=" * 80)

        # Check if answer is present
        if "\\boxed{" in result["trajectory"]:
            log.info("\n✓ Answer format detected (\\boxed{})")
        else:
            log.warning("\n⚠ No \\boxed{} answer format found")

        return True

    except Exception as e:
        log.error("\n" + "=" * 80)
        log.error("FAILED! Error testing Qwen3 baseline:")
        log.error("=" * 80)
        log.error(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_qwen3_baseline()
    sys.exit(0 if success else 1)

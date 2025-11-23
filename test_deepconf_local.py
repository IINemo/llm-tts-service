#!/usr/bin/env python3
"""
Quick test to verify DeepConf works with local HuggingFace models.
Uses a tiny model (facebook/opt-125m) for fast CPU testing.
"""

import logging
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project to path
sys.path.insert(0, "/Users/karantonis/MBZUAI/courses/NLP/llm-tts-service")

from llm_tts.strategies.deepconf.strategy import StrategyDeepConf  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class SimpleLocalModel:
    """Minimal local model wrapper for testing."""

    def __init__(self, model_name="facebook/opt-125m"):
        log.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cpu"  # Use CPU for quick test
        self.model.to(self.device)
        log.info("Model loaded successfully")

    def tokenize(self, texts):
        """Tokenize texts (matching expected interface)."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )


def test_deepconf_local():
    """Test DeepConf with a local model."""
    log.info("=" * 60)
    log.info("Testing DeepConf with Local HuggingFace Model")
    log.info("=" * 60)

    # Load tiny model
    model = SimpleLocalModel("facebook/opt-125m")

    # Create DeepConf strategy (offline mode, minimal budget for speed)
    log.info("\nInitializing DeepConf strategy...")
    strategy = StrategyDeepConf(
        model=model,
        mode="offline",
        budget=3,  # Generate only 3 traces for quick test
        window_size=5,
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,  # Short generation for speed
        top_logprobs=5,  # Request top-5 logprobs
        filter_method="top2",  # Keep top 2 traces
        n_threads=1,  # Single thread for simplicity
    )

    # Test with a simple math problem
    test_prompt = "What is 2 + 2? Please put your answer in \\boxed{}"

    log.info(f"\nTest prompt: {test_prompt}")
    log.info("\nGenerating trajectory...")

    try:
        result = strategy.generate_trajectory(test_prompt)

        log.info("\n" + "=" * 60)
        log.info("SUCCESS! DeepConf works with local models!")
        log.info("=" * 60)

        log.info(f"\nStrategy: {result['strategy']}")
        log.info(f"Completed: {result['completed']}")

        if "metadata" in result:
            metadata = result["metadata"]
            if "config" in metadata:
                log.info(
                    f"Model type detected: {metadata.get('model_type', 'Unknown')}"
                )
                log.info(
                    f"Total traces: {metadata['config'].get('total_traces', 'N/A')}"
                )
                log.info(
                    f"Filtered traces: {metadata['config'].get('filtered_traces', 'N/A')}"
                )

        log.info("\nGenerated trajectory (first 200 chars):")
        log.info(result["trajectory"][:200] + "...")

        return True

    except Exception as e:
        log.error("\n" + "=" * 60)
        log.error("FAILED! Error testing DeepConf with local model:")
        log.error("=" * 60)
        log.error(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_deepconf_local()
    sys.exit(0 if success else 1)

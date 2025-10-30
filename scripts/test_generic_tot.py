#!/usr/bin/env python3
"""
Test script for generic Tree-of-Thoughts mode.

Usage:
    python scripts/test_generic_tot.py "Your question here"

Example:
    python scripts/test_generic_tot.py "How can I improve my startup's user retention?"
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from .env file
from dotenv import load_dotenv  # noqa: E402

# Load .env from project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging to show progress
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from llm_tts.models.blackboxmodel_with_streaming import (  # noqa: E402
    BlackboxModelWithStreaming,
)
from llm_tts.scorers.tree_of_thoughts.value_scorer import TotValueScorer  # noqa: E402
from llm_tts.strategies.tree_of_thoughts.strategy import (  # noqa: E402
    StrategyTreeOfThoughts,
)


def test_generic_tot(question: str):
    """
    Test Tree-of-Thoughts in generic mode on any question.

    Args:
        question: User's question or problem to solve
    """
    # Get API key from .env
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file or environment")
        print(f"Checked .env path: {env_path}")
        print("\nPlease create a .env file in the project root with:")
        print("OPENROUTER_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 80)
    print("Tree-of-Thoughts: Generic Mode Test")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")

    # Create model
    print("Initializing model (gpt-4o-mini)...")
    model = BlackboxModelWithStreaming(
        openai_api_key=api_key,
        model_path="openai/gpt-4o-mini",
        supports_logprobs=False,
        base_url="https://openrouter.ai/api/v1",
    )

    # Create scorer
    print("Creating value scorer...")
    scorer = TotValueScorer(
        model=model,
        n_evaluate_sample=3,
        temperature=0.0,
        max_tokens=50,
        timeout=120,
        value_prompt_path="config/prompts/generic_tot_value.txt",
    )

    # Create strategy in GENERIC mode
    print("Initializing ToT strategy in GENERIC mode...")
    strategy = StrategyTreeOfThoughts(
        model=model,
        scorer=scorer,
        mode="generic",  # Generic mode
        method_generate="propose",
        beam_width=3,  # Smaller for faster testing
        n_generate_sample=3,
        steps=3,
        temperature=0.7,
        max_tokens_per_step=150,
        n_threads=4,
        propose_prompt_path="config/prompts/generic_tot_propose.txt",
    )

    print("\nRunning Tree-of-Thoughts search...")
    print("-" * 80)

    # Generate trajectory
    result = strategy.generate_trajectory(question)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nCompleted: {result['completed']}")
    print(f"Total API calls: {result['metadata']['total_api_calls']}")
    print(f"Best state score: {result['metadata'].get('best_state_score', 'N/A')}")

    print("\n" + "-" * 80)
    print("TRAJECTORY")
    print("-" * 80)
    print(result["trajectory"])

    print("\n" + "-" * 80)
    print("GENERATED ANSWER")
    print("-" * 80)
    print(result.get("generated_answer", "No answer generated"))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample questions:")
        print(
            '  python scripts/test_generic_tot.py "How can I learn machine learning in 3 months?"'
        )
        print(
            '  python scripts/test_generic_tot.py "Design a database schema for a social network"'
        )
        print(
            '  python scripts/test_generic_tot.py "What is the best way to structure a REST API?"'
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    test_generic_tot(question)

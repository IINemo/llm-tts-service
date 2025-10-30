#!/usr/bin/env python3
"""
Quick demo of Tree-of-Thoughts visualization.

This script runs a minimal ToT search and generates an interactive visualization.
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables
from dotenv import load_dotenv  # noqa: E402

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
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
from llm_tts.visualization import TotVisualizer  # noqa: E402


def main():
    """Run minimal ToT example with visualization."""
    # Check API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in .env file")
        sys.exit(1)

    question = "What are 3 benefits of test-time scaling in LLMs?"

    print("=" * 80)
    print("ToT Visualization Demo")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")
    print("Configuration: Minimal settings for fast execution")
    print("  - Beam width: 2")
    print("  - Steps: 2")
    print("  - Generation samples: 2")
    print("  - Evaluation samples: 1")
    print()

    # Create model
    model = BlackboxModelWithStreaming(
        openai_api_key=api_key,
        model_path="openai/gpt-4o-mini",
        supports_logprobs=False,
        base_url="https://openrouter.ai/api/v1",
    )

    # Create scorer
    scorer = TotValueScorer(
        model=model,
        n_evaluate_sample=1,  # Minimal for speed
        temperature=0.0,
        max_tokens=50,
        timeout=120,
        value_prompt_path="config/prompts/tot/generic_tot_value.txt",
    )

    # Create strategy
    strategy = StrategyTreeOfThoughts(
        model=model,
        scorer=scorer,
        mode="generic",
        method_generate="propose",
        beam_width=2,  # Minimal for speed
        n_generate_sample=2,  # Minimal for speed
        steps=2,  # Minimal for speed
        temperature=0.7,
        max_tokens_per_step=100,
        n_threads=2,
        propose_prompt_path="config/prompts/tot/generic_tot_propose.txt",
    )

    print("Running ToT search...")
    print("-" * 80)

    # Generate trajectory
    result = strategy.generate_trajectory(question)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nCompleted: {result['completed']}")
    metadata = result.get("metadata", {})
    gen_details = metadata.get("generation_details", {})
    print(f"Total API calls: {gen_details.get('total_api_calls', 'N/A')}")

    # Generate visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    visualizer = TotVisualizer(
        width=1400,
        height=900,
        show_state_preview=True,
        max_state_chars=60,
    )

    output_file = "tot_demo_visualization.html"
    print("\nCreating interactive visualization...")

    try:
        visualizer.visualize(
            result,
            output_path=output_file,
            title=f"ToT Demo: {question}",
            show=False,
        )
        print(f"\n✓ SUCCESS! Visualization saved to: {output_file}")
        print("\nOpen this file in your browser to explore the reasoning tree:")
        print(f"  file://{Path(output_file).absolute()}")
        print("\nVisualization features:")
        print("  • Interactive tree showing all reasoning paths explored")
        print("  • Color-coded nodes by score (green=high, red=low)")
        print("  • Hover over nodes to see full state content")
        print("  • Zoom and pan to explore different parts of the tree")
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

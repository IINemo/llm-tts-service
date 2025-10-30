#!/usr/bin/env python3
"""
Visualize ToT with mock/predefined metadata for quick testing.

This script creates realistic ToT metadata without making API calls,
allowing rapid iteration on visualization development.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_tts.visualization import TotVisualizer  # noqa: E402


def create_mock_tot_result():
    """
    Create realistic ToT result with predefined metadata.

    Returns a complete result dictionary matching the structure
    from StrategyTreeOfThoughts.generate_trajectory()
    """
    # Mock data for a simple reasoning tree
    # Problem: "What are 3 key benefits of microservices?"

    result = {
        "trajectory": (
            "**Break down the question**: Identify that we need 3 distinct benefits\n"
            "**Consider scalability aspect**: Microservices can scale independently\n"
            "**Final answer**: The 3 key benefits are: 1) Independent scaling, "
            "2) Technology flexibility, 3) Fault isolation"
        ),
        "steps": [
            "**Break down the question**: Identify that we need 3 distinct benefits",
            "**Consider scalability aspect**: Microservices can scale independently",
            "**Final answer**: The 3 key benefits are: 1) Independent scaling, 2) Technology flexibility, 3) Fault isolation",
        ],
        "validity_scores": [1.0, 20.0, 20.0],
        "completed": True,
        "generated_answer": "Independent scaling, Technology flexibility, Fault isolation",
        "metadata": {
            "strategy": "tree_of_thoughts",
            "config": {
                "method_generate": "propose",
                "scorer": "TotValueScorer(tot_value_scorer)",
                "beam_width": 3,
                "n_generate_sample": 5,
                "steps": 3,
                "temperature": 0.7,
            },
            "results": {
                "selected_answer": "Independent scaling, Technology flexibility, Fault isolation",
                "best_score": 20.0,
                "final_states": [
                    "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently\n**Final answer**: The 3 key benefits are: 1) Independent scaling, 2) Technology flexibility, 3) Fault isolation",
                    "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services\n**Final answer**: The 3 key benefits are: 1) Faster development, 2) Independent deployment, 3) Team autonomy",
                    "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services\n**Final answer**: The 3 key benefits are: 1) Modularity, 2) Easier updates, 3) Better code organization",
                ],
                "final_scores": [20.0, 20.0, 1.0],
            },
            "generation_details": {
                "total_api_calls": 15,
                "scorer_evaluations": 25,
                "total_candidates_evaluated": 15,
                # This is the key part - step-by-step exploration
                "all_steps": [
                    {
                        "step_idx": 0,
                        "candidates": [
                            "**Break down the question**: Identify that we need 3 distinct benefits",
                            "**Research existing knowledge**: Recall common microservices advantages",
                            "**Analyze from architecture perspective**: Microservices promote modularity",
                            "**Consider organizational impact**: How microservices affect team structure",
                            "**Compare with monoliths**: Identify what microservices do better",
                        ],
                        "scores": [20.0, 1.0, 1.0, 1.0, 0.001],
                        "selected_states": [
                            "**Break down the question**: Identify that we need 3 distinct benefits",
                            "**Research existing knowledge**: Recall common microservices advantages",
                            "**Analyze from architecture perspective**: Microservices promote modularity",
                        ],
                        "selected_scores": [20.0, 1.0, 1.0],
                    },
                    {
                        "step_idx": 1,
                        "candidates": [
                            # From state 1 (Break down)
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Think about deployment**: Each service can be deployed separately",
                            # From state 2 (Research)
                            "**Research existing knowledge**: Recall common microservices advantages\n**List technical benefits**: Scalability, flexibility, resilience",
                            "**Research existing knowledge**: Recall common microservices advantages\n**Focus on business value**: Faster time to market, better resource usage",
                            # From state 3 (Architecture)
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services",
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Think about testing**: Smaller services are easier to test",
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider technology choices**: Different services can use different tech stacks",
                        ],
                        "scores": [20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        "selected_states": [
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services",
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services",
                        ],
                        "selected_scores": [20.0, 20.0, 1.0],
                    },
                    {
                        "step_idx": 2,
                        "candidates": [
                            # From state 1 (scalability path)
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently\n**Final answer**: The 3 key benefits are: 1) Independent scaling, 2) Technology flexibility, 3) Fault isolation",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently\n**Add resilience benefit**: Services can fail independently without bringing down entire system",
                            # From state 2 (development speed path)
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services\n**Final answer**: The 3 key benefits are: 1) Faster development, 2) Independent deployment, 3) Team autonomy",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services\n**Think about CI/CD**: Easier to implement continuous deployment",
                            # From state 3 (architecture path)
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services\n**Final answer**: The 3 key benefits are: 1) Modularity, 2) Easier updates, 3) Better code organization",
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services\n**Add reusability**: Services can be reused across different applications",
                        ],
                        "scores": [20.0, 1.0, 20.0, 1.0, 1.0, 1.0],
                        "selected_states": [
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently\n**Final answer**: The 3 key benefits are: 1) Independent scaling, 2) Technology flexibility, 3) Fault isolation",
                            "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services\n**Final answer**: The 3 key benefits are: 1) Faster development, 2) Independent deployment, 3) Team autonomy",
                            "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services\n**Final answer**: The 3 key benefits are: 1) Modularity, 2) Easier updates, 3) Better code organization",
                        ],
                        "selected_scores": [20.0, 20.0, 1.0],
                    },
                ],
            },
        },
    }

    return result


def main():
    """Generate visualization from mock data."""
    print("=" * 80)
    print("Mock ToT Visualization (No API Calls)")
    print("=" * 80)
    print("\nUsing predefined metadata for:")
    print('Question: "What are 3 key benefits of microservices?"')
    print("\nConfiguration:")
    print("  - Beam width: 3")
    print("  - Steps: 3")
    print("  - Candidates per step: 5")
    print()

    # Create mock result
    result = create_mock_tot_result()

    # Create visualizer
    visualizer = TotVisualizer(
        width=1400,
        height=900,
        show_state_preview=True,
        max_state_chars=10,
    )

    output_file = "tot_mock_visualization.html"
    print("Generating visualization...")

    try:
        visualizer.visualize(
            result,
            output_path=output_file,
            title="Mock ToT: Microservices Benefits",
            show=False,
        )
        print(f"\n✓ SUCCESS! Visualization saved to: {output_file}")
        print("\nOpen this file in your browser:")
        print(f"  file://{Path(output_file).absolute()}")
        print("\nThis mock tree shows:")
        print("  • 3 steps of reasoning")
        print("  • Multiple candidate paths at each step")
        print("  • Beam search selecting top 3 states")
        print("  • Final answers with different approaches")
        print("\nTry dragging nodes with Shift + Click!")
    except Exception as e:
        print(f"\n✗ Visualization failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

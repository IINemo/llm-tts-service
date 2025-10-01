#!/usr/bin/env python3
"""
Complex math tests for DeepConf strategy.

Run with:
    export OPENROUTER_API_KEY="your-key"
    python tests/test_deepconf_math.py

Options:
    --verbose, -v    Show full reasoning paths for each trace
    --budget, -b N   Number of traces per problem (default: 5)

Examples:
    # Standard run
    python tests/test_deepconf_math.py

    # Verbose mode with full reasoning
    python tests/test_deepconf_math.py --verbose

    # More traces for better accuracy
    python tests/test_deepconf_math.py --budget 10 --verbose
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
import importlib.util

# Import models normally
from llm_tts.models import create_model

# Import deepconf_strategy directly without triggering strategies/__init__.py
spec = importlib.util.spec_from_file_location(
    "llm_tts.strategies.deepconf_strategy",
    "llm_tts/strategies/deepconf_strategy.py"
)
deepconf_module = importlib.util.module_from_spec(spec)
sys.modules['llm_tts.strategies.deepconf_strategy'] = deepconf_module
spec.loader.exec_module(deepconf_module)
DeepConfStrategy = deepconf_module.DeepConfStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# Test problems with expected answers
MATH_PROBLEMS = [
    {
        "problem": "Calculate 15^2 - 8^2. Put your answer in \\boxed{}.",
        "expected": "161",
        "description": "Difference of squares"
    },
    {
        "problem": "A rectangle has length 12 cm and width 8 cm. What is its area in square centimeters? Put answer in \\boxed{}.",
        "expected": "96",
        "description": "Rectangle area"
    },
    {
        "problem": "If a train travels 120 km in 2 hours at constant speed, how far will it travel in 5 hours? Put answer in \\boxed{}.",
        "expected": "300",
        "description": "Speed-distance problem"
    },
    {
        "problem": "What is the sum of the first 10 positive integers (1+2+3+...+10)? Put answer in \\boxed{}.",
        "expected": "55",
        "description": "Arithmetic series"
    },
    {
        "problem": "Calculate (3 + 4) * (5 + 6). Put your answer in \\boxed{}.",
        "expected": "77",
        "description": "Order of operations"
    },
]


def run_math_test(model, problem_data: dict, budget: int = 5, verbose: bool = False) -> dict:
    """Run a single math problem through DeepConf

    Args:
        model: Model instance
        problem_data: Problem dict with 'problem', 'expected', 'description'
        budget: Number of traces to generate
        verbose: If True, log full reasoning paths
    """
    problem = problem_data["problem"]
    expected = problem_data["expected"]
    description = problem_data["description"]

    log.info("="*70)
    log.info(f"Problem: {description}")
    log.info(f"Question: {problem}")
    log.info(f"Expected: {expected}")
    log.info("="*70)

    strategy = DeepConfStrategy(
        model=model,
        budget=budget,
        window_size=16,
        temperature=0.7,
        max_tokens=500,
        filter_method="none"  # Use all traces
    )

    try:
        result = strategy.generate_trajectory(problem)

        selected = result['metadata']['selected_answer']
        confidence = result['metadata']['confidence_score']
        num_used = result['metadata']['num_paths_used']
        num_total = result['metadata']['num_paths_generated']

        is_correct = selected == expected

        # Log full reasoning paths if verbose
        if verbose:
            log.info(f"\n📝 Full Reasoning Paths:")
            log.info("-" * 70)
            for i, trace in enumerate(result['metadata']['all_traces'], 1):
                log.info(f"\n🧠 Trace {i}/{num_total}:")
                log.info(f"   Answer: {trace.get('extracted_answer', 'N/A')}")
                log.info(f"   Min confidence: {trace.get('min_conf', 0):.3f}")
                log.info(f"   Tokens: {trace.get('num_tokens', 0)}")
                log.info(f"\n   Reasoning:")
                reasoning_text = trace.get('text', '')
                for line_num, line in enumerate(reasoning_text.split('\n'), 1):
                    if line.strip():
                        log.info(f"     {line_num:2d}: {line.strip()}")
                log.info("-" * 70)

        log.info(f"\n📊 Results:")
        log.info(f"   Selected answer: {selected}")
        log.info(f"   Expected answer: {expected}")
        log.info(f"   Correct: {'✅ YES' if is_correct else '❌ NO'}")
        log.info(f"   Confidence: {confidence:.3f}")
        log.info(f"   Traces used: {num_used}/{num_total}")

        if result['metadata']['vote_distribution']:
            log.info(f"   Vote distribution:")
            for ans, pct in sorted(result['metadata']['vote_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
                log.info(f"     {ans}: {pct:.1f}%")

        return {
            "problem": description,
            "correct": is_correct,
            "selected": selected,
            "expected": expected,
            "confidence": confidence,
            "num_traces": num_total
        }

    except Exception as e:
        log.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "problem": description,
            "correct": False,
            "selected": None,
            "expected": expected,
            "confidence": 0.0,
            "num_traces": 0
        }


def main():
    """Run all math tests"""
    import argparse

    parser = argparse.ArgumentParser(description='DeepConf Complex Math Tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Log full reasoning paths for each problem')
    parser.add_argument('--budget', '-b', type=int, default=5,
                       help='Number of reasoning traces per problem (default: 5)')
    args = parser.parse_args()

    log.info("\n" + "="*70)
    log.info("🧮 DeepConf - Complex Math Problems")
    if args.verbose:
        log.info("   (Verbose mode: showing full reasoning paths)")
    log.info("="*70 + "\n")

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.error("❌ OPENROUTER_API_KEY environment variable not set")
        log.info("Set it with: export OPENROUTER_API_KEY='your-key'")
        return 1

    # Create model
    log.info("Initializing model...")
    try:
        model = create_model(
            provider="openrouter",
            model_name="openai/gpt-4o-mini",
            api_key=api_key,
            top_logprobs=20
        )
        log.info("✅ Model initialized\n")
    except Exception as e:
        log.error(f"❌ Failed to initialize model: {e}")
        return 1

    # Run all problems
    results = []
    for problem_data in MATH_PROBLEMS:
        result = run_math_test(model, problem_data, budget=args.budget, verbose=args.verbose)
        results.append(result)
        log.info("")  # Blank line between problems

    # Summary
    log.info("="*70)
    log.info("📊 SUMMARY")
    log.info("="*70)

    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    log.info(f"\nOverall Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
    log.info(f"\nDetailed Results:")
    log.info("-" * 70)

    for r in results:
        status = "✅" if r["correct"] else "❌"
        log.info(f"{status} {r['problem']:<30} Expected: {r['expected']:<10} Got: {r['selected']:<10} Conf: {r['confidence']:.3f}")

    # Confidence analysis
    correct_confs = [r['confidence'] for r in results if r['correct']]
    incorrect_confs = [r['confidence'] for r in results if not r['correct']]

    if correct_confs:
        avg_correct_conf = sum(correct_confs) / len(correct_confs)
        log.info(f"\nAvg confidence (correct): {avg_correct_conf:.3f}")

    if incorrect_confs:
        avg_incorrect_conf = sum(incorrect_confs) / len(incorrect_confs)
        log.info(f"Avg confidence (incorrect): {avg_incorrect_conf:.3f}")

    log.info("\n" + "="*70)

    if accuracy == 1.0:
        log.info("🎉 Perfect score! All problems solved correctly!")
        return 0
    elif accuracy >= 0.8:
        log.info("✅ Good performance!")
        return 0
    else:
        log.warning("⚠️  Some problems were not solved correctly")
        return 0  # Still return 0 as tests measure performance, not pass/fail


if __name__ == "__main__":
    sys.exit(main())

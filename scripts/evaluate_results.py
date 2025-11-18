import argparse
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scripts.run_tts_eval import build_evaluators

log = logging.getLogger("score_results")


def load_results(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, "json"
    except json.JSONDecodeError:
        records = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records, "jsonl"


def _save_results_jsonl(results, jsonl_path: Path):
    jsonl_path = Path(jsonl_path)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _save_results_json(results, json_path: Path):
    json_path = Path(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def analyze_uncert_cot_correlation(results, evaluators):
    for eval_name in evaluators.keys():
        print(f"\n[{eval_name}] Correlation Analysis:")
        print("-" * 40)

        correct_validities = []
        incorrect_validities = []
        correct_avg_validities = []
        incorrect_avg_validities = []
        correct_num_steps = []
        incorrect_num_steps = []
        correct_num_greedy = []
        incorrect_num_greedy = []
        correct_num_multipath = []
        incorrect_num_multipath = []

        for r in results:
            eval_field = r.get("eval", {})
            res = eval_field.get(eval_name)
            if not res:
                continue

            is_correct = res.get("is_correct")
            if is_correct is None:
                continue

            # Get validity scores and metadata
            validity_scores = r.get("validity_scores", [])
            metadata = r.get("metadata", {})

            if validity_scores:
                if is_correct:
                    correct_validities.extend(validity_scores)
                    correct_avg_validities.append(np.mean(validity_scores))
                else:
                    incorrect_validities.extend(validity_scores)
                    incorrect_avg_validities.append(np.mean(validity_scores))

            # Get step counts
            num_steps = metadata.get("num_steps", 0)
            num_greedy = metadata.get("num_greedy_steps", 0)
            num_multipath = metadata.get("num_multi_path_steps", 0)

            if is_correct:
                correct_num_steps.append(num_steps)
                correct_num_greedy.append(num_greedy)
                correct_num_multipath.append(num_multipath)
            else:
                incorrect_num_steps.append(num_steps)
                incorrect_num_greedy.append(num_greedy)
                incorrect_num_multipath.append(num_multipath)

        # Print statistics
        if correct_avg_validities and incorrect_avg_validities:
            print("Average Validity Scores:")
            print(
                f"  Correct samples: {np.mean(correct_avg_validities):.4f} ± {np.std(correct_avg_validities):.4f} (n={len(correct_avg_validities)})"
            )
            print(
                f"  Incorrect samples: {np.mean(incorrect_avg_validities):.4f} ± {np.std(incorrect_avg_validities):.4f} (n={len(incorrect_avg_validities)})"
            )

        if correct_num_steps and incorrect_num_steps:
            print("Number of Steps:")
            print(
                f"  Correct samples: {np.mean(correct_num_steps):.2f} ± {np.std(correct_num_steps):.2f} (n={len(correct_num_steps)})"
            )
            print(
                f"  Incorrect samples: {np.mean(incorrect_num_steps):.2f} ± {np.std(incorrect_num_steps):.2f} (n={len(incorrect_num_steps)})"
            )

        if correct_num_greedy and incorrect_num_greedy:
            print("Number of Greedy Steps:")
            print(
                f"  Correct samples: {np.mean(correct_num_greedy):.2f} ± {np.std(correct_num_greedy):.2f} (n={len(correct_num_greedy)})"
            )
            print(
                f"  Incorrect samples: {np.mean(incorrect_num_greedy):.2f} ± {np.std(incorrect_num_greedy):.2f} (n={len(incorrect_num_greedy)})"
            )

        if correct_num_multipath and incorrect_num_multipath:
            print("Number of Multi-path Steps:")
            print(
                f"  Correct samples: {np.mean(correct_num_multipath):.2f} ± {np.std(correct_num_multipath):.2f} (n={len(correct_num_multipath)})"
            )
            print(
                f"  Incorrect samples: {np.mean(incorrect_num_multipath):.2f} ± {np.std(incorrect_num_multipath):.2f} (n={len(incorrect_num_multipath)})"
            )

        if correct_validities and incorrect_validities:
            print("Overall Validity Score Distribution:")
            print(
                f"  Correct step validities: {np.mean(correct_validities):.4f} ± {np.std(correct_validities):.4f} (n={len(correct_validities)})"
            )
            print(
                f"  Incorrect step validities: {np.mean(incorrect_validities):.4f} ± {np.std(incorrect_validities):.4f} (n={len(incorrect_validities)})"
            )


def summarize_and_print(results, evaluators):
    completed = sum(r.get("completed", False) for r in results)
    errors = sum("error" in r for r in results)

    summary_correct = {name: 0 for name in evaluators.keys()}
    summary_incorrect = {name: 0 for name in evaluators.keys()}
    correct_ids = {name: [] for name in evaluators.keys()}
    incorrect_ids = {name: [] for name in evaluators.keys()}

    for r in results:
        eval_field = r.get("eval", {})
        for name in evaluators.keys():
            res = eval_field.get(name)
            if not res:
                continue
            if res.get("is_correct"):
                summary_correct[name] += 1
                correct_ids[name].append(r.get("index"))
            elif res.get("is_correct") is False:
                summary_incorrect[name] += 1
                incorrect_ids[name].append(r.get("index"))

    print(f"Total samples: {len(results)}")
    print(f"Completed: {completed} ({completed/len(results):.1%})")
    print(f"Errors: {errors} ({errors/len(results):.1%})")
    for name in sorted(list(evaluators.keys())):
        correct = summary_correct[name]
        incorrect = summary_incorrect[name]
        print(f"[{name}]")
        print(f"  Correct: {correct} ({correct/len(results):.1%})")
        print(f"  Incorrect: {incorrect} ({incorrect/len(results):.1%})")
        # print(f"  Correct IDs: {correct_ids[name]}")
        # print(f"  Incorrect IDs: {incorrect_ids[name]}")

    all_validities = []
    all_steps = []
    for r in results:
        if "validity_scores" in r and r["validity_scores"]:
            all_validities.extend(r["validity_scores"])
            all_steps.append(len(r.get("steps", [])))
    if all_steps:
        print(f"Avg steps per trajectory: {np.mean(all_steps):.1f}")
    if all_validities:
        print(f"Avg validity score: {np.mean(all_validities):.3f}")

    analyze_uncert_cot_correlation(results, evaluators)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results_json", type=str)
    group.add_argument("--results_dir", type=str)

    parser.add_argument("--evaluators", nargs="+", default=None)
    parser.add_argument(
        "--llm_judge_config",
        type=str,
        default="./config/evaluation/llm_judge/default.yaml",
    )
    parser.add_argument(
        "--alignscore_config",
        type=str,
        default="./config/evaluation/alignscore/default.yaml",
    )
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    if args.results_dir:
        dir_path = Path(args.results_dir)
        results_path = next(
            (
                p
                for p in (dir_path / "results.json", dir_path / "results.jsonl")
                if p.exists()
            ),
            None,
        )
        config_path = dir_path / ".hydra" / "config.yaml"
    else:
        results_path = Path(args.results_json)
        config_path = Path(__file__).parent.parent / "config" / "run_tts_eval.yaml"

    if not results_path or not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    cfg = OmegaConf.load(config_path) if config_path.exists() else {}
    cfg_raw = OmegaConf.to_container(cfg, resolve=False) or {}
    eval_raw = cfg_raw.get("evaluation", {}) or {}
    dataset_raw = cfg_raw.get("dataset", {}) or {}

    if args.evaluators is not None:
        requested = [e.strip() for e in args.evaluators]
    else:
        requested = eval_raw.get("evaluators", ["exact_match"])

    eval_cfg = {"evaluators": []}

    for name in requested:
        if name == "exact_match":
            eval_cfg["evaluators"].append("exact_match")
            continue

        if name not in ("llm_judge", "alignscore"):
            log.warning(f"Unknown evaluator: {name}")
            continue

        override_path = (
            args.llm_judge_config if name == "llm_judge" else args.alignscore_config
        )
        if override_path and Path(override_path).exists():
            source_cfg = OmegaConf.load(override_path)
            source_cfg = OmegaConf.to_container(source_cfg, resolve=False)
            log.info(f"Using custom config for {name}: {override_path}")
        else:
            source_cfg = eval_raw.get(name)

        if source_cfg is None:
            log.warning(f"No config for {name}, skipping")
            continue

        eval_cfg["evaluators"].append(name)
        eval_cfg[name] = source_cfg

    if not eval_cfg["evaluators"]:
        print("No evaluators to run.")
        return

    minimal_cfg = OmegaConf.create({"evaluation": eval_cfg, "dataset": dataset_raw})
    evaluators = build_evaluators(minimal_cfg)

    results, input_format = load_results(results_path)
    results = results

    problems = []
    solutions = []
    gold_answers = []
    indices = []

    for i, r in enumerate(results):
        if r.get("error"):
            continue
        problems.append(r["question"])
        solutions.append(r.get("generated_answer", ""))
        gold_answers.append(r.get("gold_answer", ""))
        indices.append(i)

    if not problems:
        print("No valid samples found.")
        return

    for name, fn in evaluators.items():
        log.info(f"Running {name} on {len(problems)} samples...")

        if "llm_judge" in name:
            try:
                labels, raw_responses = fn(problems, solutions, gold_answers)
            except Exception as e:
                log.error(f"LLM judge failed: {e}")
                continue
        else:
            labels = fn(problems, solutions, gold_answers)
            raw_responses = [None] * len(labels)

        for idx, label, response in zip(indices, labels, raw_responses):
            label = float(label)
            is_correct = label == 1

            results[idx].setdefault("eval", {})[name] = {
                "label": label,
                "is_correct": is_correct,
                "raw_judge_response": response.strip() if response else None,
            }

    save_path = Path(args.save_to) if args.save_to else results_path
    if save_path.is_dir():
        save_path = save_path / (
            "results.jsonl" if input_format == "jsonl" else "results.json"
        )
    if save_path.suffix == ".jsonl":
        _save_results_jsonl(results, save_path)
    else:
        _save_results_json(results, save_path)

    _save_results_json(results, save_path)
    print("\nScoring complete!")
    print(f"Evaluators used: {list(evaluators.keys())}")
    print(f"Results saved to: {save_path}")

    summarize_and_print(results, evaluators)


if __name__ == "__main__":
    main()

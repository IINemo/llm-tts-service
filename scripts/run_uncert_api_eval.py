#!/usr/bin/env python3
import os
import re
import random
import json
import uuid
import sys
import subprocess
from datetime import datetime
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from hydra.utils import to_absolute_path
from datasets import load_dataset

from llm_tts.openrouter_chat import OpenRouterChat
from llm_tts.together_chat import TogetherChatCompat
from llm_tts.strategies.uncertainty_guided_pd import UncertaintyGuidedCoT_PD
from llm_tts.step_detection import StepBoundaryDetector, uncert_detector
from llm_tts.annotators.provider_verifier import ProviderVerifier

import logging
log = logging.getLogger(__name__)


def format_prompt_gsm8k(question: str) -> str:
    return (
        f"Question: {question}\n"
        f"Let's solve this step by step. Think first, then provide the final answer in the form '<Answer>: <number>'.\n"
        "## Step 1: "
    )


def format_prompt_hotpot(question: str, context: List) -> str:
    snippets = []
    for title, sents in context[:2]:
        if not sents:
            continue
        take = " ".join(sents[:2])
        snippets.append(f"{title}: {take}")
    ctx_text = "\n- ".join(snippets)
    preface = (
        "Use the following passages to answer the question. Cite facts as you go.\n"
        + (f"- {ctx_text}\n\n" if ctx_text else "")
    )
    return (
        f"{preface}Question: {question}\n\n"
        "Let's solve this step by step. Think first, then provide the final answer in the form '<Answer>: <text>'.\n\n"
        "## Step 1: "
    )


def base_prompt_gsm8k(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Provide the final answer in the form '<Answer>: <number>'.\n"
    )


def base_prompt_hotpot(question: str, context: List) -> str:
    snippets = []
    for title, sents in context[:2]:
        if not sents:
            continue
        take = " ".join(sents[:2])
        snippets.append(f"{title}: {take}")
    ctx_text = "\n- ".join(snippets)
    preface = (
        "Use the following passages to answer the question.\n"
        + (f"- {ctx_text}\n\n" if ctx_text else "")
    )
    return (
        f"{preface}Question: {question}\n\n"
        "Provide the final answer in the form '<Answer>: <text>'.\n\n"
    )

def extract_answer_text(output: str, prompt_prefix: str = "") -> str:
    """Extract answer consistently with StepBoundaryDetector's answer patterns.

    - Uses the last occurrence of any answer pattern (case-insensitive)
    - Works for both '<answer>:' and 'the final answer is: '
    - Returns the first number after the marker if present, else first line
    """
    if not output:
        return ""
    text = output[len(prompt_prefix):] if prompt_prefix and output.startswith(prompt_prefix) else output
    # Build patterns from detector (default) and add capitalized variants
    det = uncert_detector()
    pats = list(det.answer_patterns)
    # Ensure canonical capitalized tags are also covered for slicing length
    pats.extend(["<Answer>:", "\n<Answer>:", "Answer:", "\nAnswer:"])
    # Find last match
    low = text.lower()
    last_idx = -1
    last_pat = None
    for p in pats:
        q = p.lower()
        idx = low.rfind(q)
        if idx > last_idx:
            last_idx = idx
            last_pat = p
    if last_idx == -1 or last_pat is None:
        return ""
    tail = text[last_idx + len(last_pat):].lstrip()
    m = re.search(r"-?\d+(?:\.\d+)?", tail)
    if m:
        return m.group(0).strip()
    # If no numeric match, return the first non-empty line if present
    if not tail:
        return ""
    lines = tail.splitlines()
    if not lines:
        return ""
    first = lines[0].strip()
    return first


def gold_answer_gsm8k(answer_field: str) -> str:
    # GSM8K format ends with '#### <number>'
    import re as _re
    m = _re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_field or "")
    return m.group(1) if m else ""


def normalize_num(x: str) -> str:
    try:
        # Normalize numeric answers to canonical string
        if x is None or x == "":
            return ""
        v = float(str(x).strip())
        if abs(v - int(v)) < 1e-9:
            return str(int(v))
        return ("%.6f" % v).rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip()


def check_correctness(dataset_name: str, ds_item: Dict, pred_answer: str) -> bool:
    if dataset_name == "gsm8k":
        gold = gold_answer_gsm8k(ds_item.get("answer", ""))
        return normalize_num(pred_answer) == normalize_num(gold)
    elif dataset_name == "hotpotqa":
        gold = (ds_item.get("answer", "") or "").strip().lower()
        return (pred_answer or "").strip().lower() == gold
    return False


def sample_indices(n: int, frac: float, seed: int = 0) -> List[int]:
    k = max(1, int(n * frac))
    rng = random.Random(seed)
    return rng.sample(range(n), k)


def run_eval(cfg: DictConfig):
    # Prompt template loader and renderers
    def _load_template_file(path: str) -> str:
        try:
            abs_path = to_absolute_path(path)
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt template from {path}: {e}")

    def _render_template(tpl: str, variables: Dict) -> str:
        try:
            return tpl.format(**variables)
        except Exception as e:
            raise RuntimeError(f"Failed to render prompt template with variables {list(variables.keys())}: {e}")

    def _format_hotpot_context(ctx_list: List) -> str:
        snippets = []
        for title, sents in ctx_list[:2]:
            if not sents:
                continue
            take = " ".join(sents[:2])
            snippets.append(f"{title}: {take}")
        return "\n- ".join(snippets)

    def render_base_prompt(dataset_name: str, question: str, ctx_list: List = None) -> str:
        # Prefer top-level overrides, then fall back to prompts group
        path = (
            OmegaConf.select(cfg, f"prompt_paths.{dataset_name}.base")
            or OmegaConf.select(cfg, f"prompts.{dataset_name}.base_template_path")
        )
        if path:
            tpl = _load_template_file(path)
            variables: Dict[str, str] = {"q": question}
            if ctx_list is not None:
                variables["context"] = _format_hotpot_context(ctx_list)
            else:
                variables["context"] = ""
            return _render_template(tpl, variables)
        # Fallbacks
        if dataset_name == "gsm8k":
            return base_prompt_gsm8k(question)
        elif dataset_name == "hotpotqa":
            return base_prompt_hotpot(question, ctx_list or [])
        else:
            return f"Question: {question}\n"

    def render_cot_prompt(dataset_name: str, question: str, ctx_list: List = None) -> str:
        # Prefer top-level overrides, then fall back to prompts group
        path = (
            OmegaConf.select(cfg, f"prompt_paths.{dataset_name}.cot")
            or OmegaConf.select(cfg, f"prompts.{dataset_name}.cot_template_path")
        )
        if path:
            tpl = _load_template_file(path)
            variables: Dict[str, str] = {"q": question}
            if ctx_list is not None:
                variables["context"] = _format_hotpot_context(ctx_list)
            else:
                variables["context"] = ""
            return _render_template(tpl, variables)
        # Fallbacks to existing hardcoded prompts
        if dataset_name == "gsm8k":
            return format_prompt_gsm8k(question)
        elif dataset_name == "hotpotqa":
            return format_prompt_hotpot(question, ctx_list or [])
        else:
            return f"Question: {question}\n\n## Step 1: "
    ds_cfg = cfg.dataset
    st_cfg = cfg.strategy
    
    # Base generation configuration (overrides for greedy/base runs)
    base_temp = cfg.base_generation.get('temperature', 0.0)
    base_max_new_tokens = cfg.base_generation.get('max_new_tokens', st_cfg.max_new_tokens)
    base_n = cfg.base_generation.get('n', 1)
    base_stop = list(cfg.base_generation.get('stop', ["<|endoftext|>"]))

    # Unified provider selection
    if not hasattr(cfg, 'provider'):
        raise ValueError("Must specify 'provider' configuration group")
    provider_type = (cfg.provider.type or "").strip().lower()
    provider_model = cfg.model.model_name
    provider_api_key = getattr(cfg.provider, 'api_key', None)
    
    if provider_type == "together":
        print(f"Using Together.ai with model: {provider_model}")
        client = TogetherChatCompat(model=provider_model, api_key=provider_api_key)
        client_type = "together"
    elif provider_type == "openrouter":
        print(f"Using OpenRouter with model: {provider_model}")
        api_base = getattr(cfg.provider, 'api_base', "https://openrouter.ai/api/v1")
        client = OpenRouterChat(model=provider_model, api_key=provider_api_key, api_base=api_base)
        client_type = "openrouter"
    else:
        raise ValueError("provider.type must be one of: openrouter, together")
    
    # Resolve step header markers with precedence:
    # 1) Explicit in main YAML under strategy.step_marker_patterns
    # 2) From Hydra group step_markers.model_patterns (matching provider model)
    # 3) Hardcoded defaults
    step_marker_patterns = None
    if hasattr(cfg, 'strategy') and hasattr(cfg.strategy, 'step_marker_patterns') and cfg.strategy.step_marker_patterns:
        try:
            step_marker_patterns = list(cfg.strategy.step_marker_patterns)
        except Exception:
            step_marker_patterns = cfg.strategy.step_marker_patterns
    elif hasattr(cfg, 'step_markers') and hasattr(cfg.step_markers, 'model_patterns'):
        model_name = provider_model
        try:
            for mp in cfg.step_markers.model_patterns:
                patt = mp.get('model_regex') if isinstance(mp, dict) else None
                if not patt:
                    continue
                import re as _re
                if _re.match(patt, model_name):
                    pats = mp.get('patterns', [])
                    if pats:
                        step_marker_patterns = pats
                        break
        except Exception:
            pass
    if not step_marker_patterns:
        step_marker_patterns = [
            "## Step {n}:",
            "Step {n}:",
            "- Step {n}:",
        ]

    strategy_kwargs = dict(
        model=None,
        scorer=None,
        candidates_per_step=st_cfg.candidates_per_step,
        max_steps=st_cfg.max_steps,
        max_new_tokens=st_cfg.max_new_tokens,
        temperature=st_cfg.temperature,
        generation_batch_size=st_cfg.candidates_per_step,
        uncertainty_metric=st_cfg.uncertainty_metric,
        uncertainty_threshold=st_cfg.uncertainty_threshold,
        uncertainty_top_k=getattr(st_cfg, 'uncertainty_top_k', 2),
        # step_header_template=step_header_template,
        step_marker_patterns=step_marker_patterns,
        # Pass the appropriate client based on configuration
        together_client=client if client_type == "together" else None,
        openrouter_client=client if client_type == "openrouter" else None,
        eos_token=cfg.model.eos_token,
    )

    # Decide which methods to run based solely on Hydra config: run.methods
    run_cfg = getattr(cfg, 'run', None)
    if run_cfg is None or not hasattr(run_cfg, 'methods'):
        raise ValueError("Missing 'run.methods' in config. Allowed values: ['base', 'uncert'] or a list of them.")
    methods_field = getattr(run_cfg, 'methods')
    
    if isinstance(methods_field, str):
        selected_methods = [methods_field]
    elif isinstance(methods_field, (list, tuple, ListConfig)):
        selected_methods = list(methods_field)
    else:
        raise ValueError("'run.methods' must be a string or list of strings: ['base', 'uncert']")
    
    normalized = [str(m).strip().lower() for m in selected_methods]
    allowed = {"base", "uncert"}
    invalid = [m for m in normalized if m not in allowed]
    
    if invalid:
        raise ValueError(f"Invalid run.methods values: {invalid}. Allowed: {sorted(allowed)}")
    
    do_base = "base" in normalized
    do_uncert = "uncert" in normalized
    
    if not do_base and not do_uncert:
        raise ValueError("No run methods selected. Set run.methods: [base, uncert]")

    # Prepare output path once per run and create JSONL file
    model_name = (cfg.model.model_name or "model").strip()
    model_name_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    dataset_name = getattr(cfg.dataset, 'name', 'dataset') or 'dataset'
    dataset_name = str(dataset_name).strip()
    dataset_name_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = getattr(getattr(cfg, 'output', {}), 'run_id', None)
    if not run_id:
        run_id = uuid.uuid4().hex[:8]
    run_id_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id))

    out_dir = os.path.join("results", model_name_safe, dataset_name_safe)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{timestamp}_{run_id_safe}.jsonl")
    # Truncate or create the file at the start of the run
    with open(out_path, "w", encoding="utf-8"):
        pass
    saved_count = 0

    if ds_cfg.name.lower() == "gsm8k":
        ds = load_dataset(ds_cfg.hf_path, ds_cfg.hf_config, split=ds_cfg.split)
        idxs = sample_indices(len(ds), ds_cfg.subset_frac)
        problem_type = "math"
        results: List[Dict] = []
        gen = UncertaintyGuidedCoT_PD(problem_type=problem_type, **strategy_kwargs) if do_uncert else None
        
        for count, i in enumerate(idxs):
            log.info(f"\n === Task {count+1} / {len(idxs)} ===")
            log.info(f"=== GSM8K Task {i+1} ===")
            q = ds[i]["question"]
            rec = {
                "index": i,
                "question": q,
            }
            # Base greedy completion (no reasoning headers)
            if do_base:
                base_p = render_base_prompt("gsm8k", q)
                base_texts = client.generate_texts(base_p, n=base_n, temperature=base_temp, max_new_tokens=base_max_new_tokens, stop=base_stop)
                base_gen = base_texts[0] if base_texts else ""
                base_ans = extract_answer_text(base_gen)
                base_ok = check_correctness("gsm8k", ds[i], base_ans)
                rec["base"] = {
                    "prompt": base_p,
                    "completion": base_gen,
                    "answer": base_ans,
                    "correct": base_ok,
                }
            # Uncertainty-guided reasoning (CoT)
            if do_uncert and gen is not None:
                prompt = render_cot_prompt("gsm8k", q)
                out = gen.generate_trajectory(prompt)
                rg_traj = out.get("trajectory", "")
                # Strip the prompt prefix so parsing does not see instruction examples
                rg_tail = rg_traj[len(prompt):] if rg_traj.startswith(prompt) else rg_traj
                rg_ans = extract_answer_text(rg_tail)
                rg_ok = check_correctness("gsm8k", ds[i], rg_ans)
                rec["uncert"] = {
                    "trajectory": rg_traj,
                    "steps": out.get("steps", []),
                    "uncertainties": out.get("uncertainties", []),
                    "decision_trace": out.get("decision_trace", []),
                    "answer": rg_ans,
                    "correct": rg_ok,
                }
            results.append(rec)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved_count += 1
    
    elif ds_cfg.name.lower() == "hotpotqa":
        ds = load_dataset(ds_cfg.hf_path, ds_cfg.hf_config, split=ds_cfg.split)
        idxs = sample_indices(len(ds), ds_cfg.subset_frac)
        problem_type = "qa"
        results = []
        gen = UncertaintyGuidedCoT_PD(problem_type=problem_type, **strategy_kwargs) if do_uncert else None
        for i in idxs:
            q = ds[i]["question"]
            ctx = ds[i]["context"]
            rec = {
                "index": i,
                "question": q,
            }
            # Base greedy completion
            if do_base:
                base_p = render_base_prompt("hotpotqa", q, ctx)
                base_texts = client.generate_texts(base_p, n=base_n, temperature=base_temp, max_new_tokens=base_max_new_tokens)
                base_gen = base_texts[0] if base_texts else ""
                base_ans = extract_answer_text(base_gen)
                base_ok = check_correctness("hotpotqa", ds[i], base_ans)
                rec["base"] = {
                    "prompt": base_p,
                    "completion": base_gen,
                    "answer": base_ans,
                    "correct": base_ok,
                }
            # Uncertainty-guided reasoning
            if do_uncert and gen is not None:
                prompt = render_cot_prompt("hotpotqa", q, ctx)
                out = gen.generate_trajectory(prompt)
                rg_traj = out.get("trajectory", "")
                rg_tail = rg_traj[len(prompt):] if rg_traj.startswith(prompt) else rg_traj
                rg_ans = extract_answer_text(rg_tail)
                rg_ok = check_correctness("hotpotqa", ds[i], rg_ans)
                rec["uncert"] = {
                    "trajectory": rg_traj,
                    "answer": rg_ans,
                    "correct": rg_ok,
                    "steps": out.get("steps", []),
                    "uncertainties": out.get("uncertainties", []),
                    "decision_trace": out.get("decision_trace", []),
                }
            results.append(rec)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved_count += 1
    else:
        raise ValueError("dataset_name must be one of: gsm8k, hotpotqa")

    print(f"Saved {saved_count} results to {out_path}")
    
    # Also write a human-readable pretty JSON snapshot in the end
    pretty_path = out_path.replace('.jsonl', '_pretty.json')
    try:
        with open(pretty_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Pretty snapshot written to {pretty_path}")
    except Exception:
        pass

    # Optional provider-based correctness post-processing (LLM as a judge)
    try:
        ver_cfg = getattr(cfg, 'verification', None)
        if ver_cfg and bool(getattr(ver_cfg, 'enabled', False)):
            print("\nRunning LLM as a judge verification")

            
            prompt_template_path = cfg.verification.get('prompt_template_path', None)
            if prompt_template_path:
                try:
                    p = to_absolute_path(prompt_template_path)
                    with open(p, 'r', encoding='utf-8') as f:
                        prompt_template = f.read()
                except Exception as e:
                    raise RuntimeError(f"Failed to load verification prompt from {ver_cfg.prompt_template_path}: {e}")
            if prompt_template is None:
                prompt_template = (
                    "You will be given a <Problem> and its proposed <Solution>.\n"
                    "Assess whether the solution is correct.\n\n"
                    "<Problem>: {q}\n\n<Solution>: {a}\n\n"
                    "Reply with '<Grade>: Correct' or '<Grade>: Incorrect' only."
                )

            # Resolve verification generation params
            ver_model = cfg.verification.get('model', None)
            ver_temperature = float(cfg.verification.get('temperature', 1.0))
            ver_max_new_tokens = int(cfg.verification.get('max_new_tokens', 256))
            ver_n = int(cfg.verification.get('n', 1))
            ver_stop = list(cfg.verification.get('stop', ["<|endoftext|>"]))

            # Build verification client matching the provider family
            if client_type == "together":
                ver_client = TogetherChatCompat(model=ver_model, api_key=provider_api_key)
            elif client_type == "openrouter":
                ver_client = OpenRouterChat(model=ver_model, api_key=provider_api_key, api_base="https://openrouter.ai/api/v1")
            else:
                raise RuntimeError("Unsupported provider for verification")

            annotator = ProviderVerifier(
                client=ver_client,
                prompt_template=prompt_template,
                n=ver_n,
                temperature=ver_temperature,
                max_new_tokens=ver_max_new_tokens,
                stop=ver_stop,
            )

            # Collect base and uncert tracks separately for annotation
            base_indices = []
            base_problems = []
            base_solutions = []
            uncert_indices = []
            uncert_problems = []
            uncert_solutions = []

            for idx, rec in enumerate(results):
                q = rec.get("question", "")
                b = rec.get("base")
                if isinstance(b, dict) and 'answer' in b:
                    base_indices.append(idx)
                    base_problems.append(q)
                    base_solutions.append(b.get("answer", ""))
                u = rec.get("uncert")
                if isinstance(u, dict) and 'answer' in u:
                    uncert_indices.append(idx)
                    uncert_problems.append(q)
                    uncert_solutions.append(u.get("answer", ""))

            # Annotate base
            if base_problems:
                base_annotations = annotator.verify_batch_with_texts(base_problems, base_solutions)
                for rec_idx, ann in zip(base_indices, base_annotations):
                    is_ok = bool(ann.get("is_correct")) if ann and ann.get("is_correct") is not None else False
                    results[rec_idx].setdefault("base", {})["is_correct_verifier"] = bool(is_ok)
                    # Save full annotator completions
                    results[rec_idx].setdefault("base", {})["annotator_completions"] = ann.get("texts", [])

            # Annotate uncert
            if uncert_problems:
                uncert_annotations = annotator.verify_batch_with_texts(uncert_problems, uncert_solutions)
                for rec_idx, ann in zip(uncert_indices, uncert_annotations):
                    is_ok = bool(ann.get("is_correct")) if ann and ann.get("is_correct") is not None else False
                    results[rec_idx].setdefault("uncert", {})["is_correct_verifier"] = bool(is_ok)
                    results[rec_idx].setdefault("uncert", {})["annotator_completions"] = ann.get("texts", [])

            # Rewrite JSONL and pretty JSON with the new fields
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    for rec in results:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                with open(pretty_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Verification results written to output files: {out_path} and {pretty_path}")
            except Exception as e:
                print(f"Failed to write verification results: {e}")
    except Exception as e:
        print(f"Verification step failed: {e}")

    # Auto-run analysis for GSM8K results (after verification so it can use judge fields)
    try:
        if ds_cfg.name.lower() == "gsm8k":
            analyze_script = to_absolute_path("scripts/analyze_gsm8k.py")
            tracks = []
            if 'do_base' in locals() and do_base:
                tracks.append("base")
            if 'do_uncert' in locals() and do_uncert:
                tracks.append("uncert")
            cmd = [sys.executable, analyze_script, "--input", out_path]
            if tracks:
                cmd.extend(["--tracks", *tracks])
            print("\nRunning GSM8K analysis")
            subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"Failed to run analysis automatically: {e}")


@hydra.main(version_base=None, config_path="../config/uncert_api", config_name="default")
def main(cfg: DictConfig):
    if not hasattr(cfg, 'provider'):
        raise RuntimeError("Must specify 'provider' configuration")
    ptype = (cfg.provider.type or "").strip().lower()
    if ptype not in ("together", "openrouter"):
        raise RuntimeError("provider.type must be one of: openrouter, together")
    # Allow api_key to be provided via YAML or environment
    if ptype == "together":
        if not (getattr(cfg.provider, 'api_key', None) or os.environ.get("TOGETHER_API_KEY")):
            raise RuntimeError("Provide Together API key via provider.api_key or TOGETHER_API_KEY env var")
    elif ptype == "openrouter":
        if not (getattr(cfg.provider, 'api_key', None) or os.environ.get("OPENROUTER_API_KEY")):
            raise RuntimeError("Provide OpenRouter API key via provider.api_key or OPENROUTER_API_KEY env var")
    
    return run_eval(cfg)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
import os
import re
import random
import json
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from datasets import load_dataset

from llm_tts.openrouter_chat import OpenRouterChat
from llm_tts.together_chat import TogetherChatCompat
from llm_tts.strategies.uncertainty_guided_pd import UncertaintyGuidedCoT_PD
from llm_tts.scorers.base import StepScorer, CandidateScore
from llm_tts.step_detection import StepBoundaryDetector

import logging


log = logging.getLogger(__name__)

class DummyScorer(StepScorer):
    def __init__(self):
        super().__init__(name="DummyScorer")

    def score_candidates_detailed(self, trajectory: str, candidates: List[str], **kwargs):
        detailed = []
        for cand in candidates:
            length = max(1, len(cand.strip()))
            score = max(0.0, 1.0 - (length / 512.0))
            detailed.append(CandidateScore(candidate_text=cand, claim_scores=[score], aggregate_scores={}))
        return detailed

    def prepare_model(self):
        return


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
    det = StepBoundaryDetector()
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
    return m.group(0).strip() if m else tail.splitlines()[0].strip()


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

    # Unified provider selection
    if not hasattr(cfg, 'provider'):
        raise ValueError("Must specify 'provider' configuration group")
    provider_type = (cfg.provider.type or "").strip().lower()
    provider_model = cfg.provider.model
    if provider_type == "together":
        print(f"Using Together.ai with model: {provider_model}")
        client = TogetherChatCompat(model=provider_model)
        client_type = "together"
    elif provider_type == "openrouter":
        print(f"Using OpenRouter with model: {provider_model}")
        api_base = getattr(cfg.provider, 'api_base', "https://openrouter.ai/api/v1")
        client = OpenRouterChat(model=provider_model, api_key=None, api_base=api_base)
        client_type = "openrouter"
    else:
        raise ValueError("provider.type must be one of: openrouter, together")

    # Resolve step header markers
    # 1) Prefer explicit top-level list: cfg.step_marker_patterns
    # 2) Else, support legacy mapping: cfg.step_markers.model_patterns (if defined in main YAML)
    # 3) Else, fall back to robust defaults
    step_marker_patterns = None
    # Case 1: explicit list on top-level
    if hasattr(cfg, 'step_marker_patterns') and isinstance(cfg.step_marker_patterns, (list, tuple)):
        step_marker_patterns = list(cfg.step_marker_patterns)
    # Case 2: legacy per-model mapping
    elif hasattr(cfg, 'step_markers') and hasattr(cfg.step_markers, 'model_patterns'):
        model_name = provider_model
        for mp in cfg.step_markers.model_patterns:
            patt = mp.get('model_regex') if isinstance(mp, dict) else None
            if patt:
                try:
                    import re as _re
                    if _re.match(patt, model_name):
                        pats = mp.get('patterns', [])
                        if pats:
                            step_marker_patterns = pats
                            break
                except Exception:
                    pass
    # Case 3: defaults
    if not step_marker_patterns:
        step_marker_patterns = [
            "## Step {n}:",
            "Step {n}:",
            "- Step {n}:",
        ]

    strategy_kwargs = dict(
        model=None,
        scorer=DummyScorer(),
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
    )

    if ds_cfg.name.lower() == "gsm8k":
        ds = load_dataset(ds_cfg.hf_path, ds_cfg.hf_config, split=ds_cfg.split)
        idxs = sample_indices(len(ds), ds_cfg.subset_frac)
        problem_type = "math"
        results: List[Dict] = []
        gen = UncertaintyGuidedCoT_PD(problem_type=problem_type, **strategy_kwargs)
        
        for i in idxs:
            log.info(f"\n=== GSM8K Task {i+1} ===")
            q = ds[i]["question"]
            # Base greedy completion (no reasoning headers)
            base_p = render_base_prompt("gsm8k", q)
            base_texts = client.generate_texts(base_p, n=1, temperature=0.0, max_new_tokens=st_cfg.max_new_tokens)
            base_gen = base_texts[0] if base_texts else ""
            base_ans = extract_answer_text(base_gen)
            base_ok = check_correctness("gsm8k", ds[i], base_ans)

            # Uncertainty-guided reasoning (CoT)
            prompt = render_cot_prompt("gsm8k", q)
            out = gen.generate_trajectory(prompt)
            rg_traj = out.get("trajectory", "")
            # Strip the prompt prefix so parsing does not see instruction examples
            rg_tail = rg_traj[len(prompt):] if rg_traj.startswith(prompt) else rg_traj
            rg_ans = extract_answer_text(rg_tail)
            rg_ok = check_correctness("gsm8k", ds[i], rg_ans)

            results.append({
                "index": i,
                "question": q,
                "base": {
                    "prompt": base_p,
                    "completion": base_gen,
                    "answer": base_ans,
                    "correct": base_ok,
                },
                "uncert": {
                    "trajectory": rg_traj,
                    "steps": out.get("steps", []),
                    "uncertainties": out.get("uncertainties", []),
                    "decision_trace": out.get("decision_trace", []),
                    "answer": rg_ans,
                    "correct": rg_ok,
                },
            })
    
    elif ds_cfg.name.lower() == "hotpotqa":
        ds = load_dataset(ds_cfg.hf_path, ds_cfg.hf_config, split=ds_cfg.split)
        idxs = sample_indices(len(ds), ds_cfg.subset_frac)
        problem_type = "qa"
        results = []
        gen = UncertaintyGuidedCoT_PD(problem_type=problem_type, **strategy_kwargs)
        for i in idxs:
            q = ds[i]["question"]
            ctx = ds[i]["context"]
            # Base greedy completion
            base_p = render_base_prompt("hotpotqa", q, ctx)
            base_texts = client.generate_texts(base_p, n=1, temperature=0.0, max_new_tokens=st_cfg.max_new_tokens)
            base_gen = base_texts[0] if base_texts else ""
            base_ans = extract_answer_text(base_gen)
            base_ok = check_correctness("hotpotqa", ds[i], base_ans)

            # Uncertainty-guided reasoning
            prompt = render_cot_prompt("hotpotqa", q, ctx)
            out = gen.generate_trajectory(prompt)
            rg_traj = out.get("trajectory", "")
            rg_tail = rg_traj[len(prompt):] if rg_traj.startswith(prompt) else rg_traj
            rg_ans = extract_answer_text(rg_tail)
            rg_ok = check_correctness("hotpotqa", ds[i], rg_ans)

            results.append({
                "index": i,
                "question": q,
                "base": {
                    "prompt": base_p,
                    "completion": base_gen,
                    "answer": base_ans,
                    "correct": base_ok,
                },
                "uncert": {
                    "trajectory": rg_traj,
                    "steps": out.get("steps", []),
                    "uncertainties": out.get("uncertainties", []),
                    "decision_trace": out.get("decision_trace", []),
                    "answer": rg_ans,
                    "correct": rg_ok,
                },
            })
    else:
        raise ValueError("dataset_name must be one of: gsm8k, hotpotqa")

    os.makedirs(os.path.dirname(cfg.output.out_path) or ".", exist_ok=True)
    with open(cfg.output.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} results to {cfg.output.out_path}")


@hydra.main(version_base=None, config_path="../config/uncert_api", config_name="default")
def main(cfg: DictConfig):
    if not hasattr(cfg, 'provider'):
        raise RuntimeError("Must specify 'provider' configuration")
    ptype = (cfg.provider.type or "").strip().lower()
    if ptype == "together":
        if not os.environ.get("TOGETHER_API_KEY"):
            raise RuntimeError("Please set TOGETHER_API_KEY in environment")
    elif ptype == "openrouter":
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("Please set OPENROUTER_API_KEY in environment")
    else:
        raise RuntimeError("provider.type must be one of: openrouter, together")
    
    return run_eval(cfg)


if __name__ == "__main__":
    main()



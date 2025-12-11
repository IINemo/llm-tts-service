# ============== Addition Imports ===============
import torch

from lm_polygraph.estimators import Perplexity
from lm_polygraph.stat_calculators import InferCausalLMCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================================


def _get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid torch_dtype: {dtype_str}. Options: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def create_uncertainty_model(config):
    # LLM inference with uncertainty estimation

    # Loading standard LLM
    torch_dtype = _get_torch_dtype(config.system.torch_dtype)
    llm = AutoModelForCausalLM.from_pretrained(config.model.model_path, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(config.model.device)

    # ======= Wrapping LLM with uncertainty estimator =========
    stat_calculators = [InferCausalLMCalculator(tokenize=False)]
    estimator = Perplexity()
    llm_with_uncertainty = CausalLMWithUncertainty(
        llm, tokenizer, stat_calculators, estimator
    )

    return llm_with_uncertainty

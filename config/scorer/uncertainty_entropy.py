# ============== Addition Imports ===============
from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.stat_calculators import EntropyCalculator, InferCausalLMCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_tts.utils import get_torch_dtype

# ===============================================


def create_uncertainty_model(config):
    # LLM inference with uncertainty estimation

    # Loading standard LLM
    torch_dtype = get_torch_dtype(config.system.torch_dtype)
    llm = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(config.model.device)
    if getattr(config.generation, "capture_hidden_states", False):
        llm.config.output_hidden_states = True

    # ======= Wrapping LLM with uncertainty estimator =========
    stat_calculators = [InferCausalLMCalculator(tokenize=False), EntropyCalculator()]
    estimator = MeanTokenEntropy()
    keep_deps = None
    if getattr(config.generation, "capture_hidden_states", False):
        keep_deps = ["out"]
    llm_with_uncertainty = CausalLMWithUncertainty(
        llm, tokenizer, stat_calculators, estimator, keep_deps=keep_deps
    )

    return llm_with_uncertainty

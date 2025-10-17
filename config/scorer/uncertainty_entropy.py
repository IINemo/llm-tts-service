# ============== Addition Imports ===============
from lm_polygraph.estimators import MeanTokenEntropy
from lm_polygraph.stat_calculators import EntropyCalculator, InferCausalLMCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================================


def create_uncertainty_model(config):
    # LLM inference with uncertainty estimation

    # Loading standard LLM
    llm = AutoModelForCausalLM.from_pretrained(config.model.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(config.model.device)

    # ======= Wrapping LLM with uncertainty estimator =========
    stat_calculators = [InferCausalLMCalculator(tokenize=False), EntropyCalculator()]
    estimator = MeanTokenEntropy()
    llm_with_uncertainty = CausalLMWithUncertainty(
        llm, tokenizer, stat_calculators, estimator
    )

    return llm_with_uncertainty

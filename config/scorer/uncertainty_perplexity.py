# ============== Addition Imports ===============
from lm_polygraph.estimators import Perplexity
from lm_polygraph.stat_calculators import InferCausalLMCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================================

def create_uncertainty_model(config):
    # LLM inference with uncertainty estimation

    # Loading standard LLM
    llm = AutoModelForCausalLM.from_pretrained(config.model.model_path, torch_dtype= "auto")
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

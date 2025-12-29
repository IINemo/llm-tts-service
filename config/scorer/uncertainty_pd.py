from lm_polygraph.stat_calculators import InferCausalLMCalculator
from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_tts.scorers.estimator_uncertainty_pd import PDGap
from llm_tts.utils import get_torch_dtype


def create_uncertainty_model(config):
    """Create a local whitebox model wrapped with PD-Gap uncertainty.

    Returns a CausalLMWithUncertainty whose generate() returns
    GenerateDecoderOnlyOutputWithUncertainty with a per-sample scalar
    uncertainty_score computed by PDGap over greedy token logits.
    """

    torch_dtype = get_torch_dtype(config.system.torch_dtype)
    llm = AutoModelForCausalLM.from_pretrained(
        config.model.model_path, torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(config.model.device)

    stat_calculators = [InferCausalLMCalculator(tokenize=False)]
    estimator = PDGap()

    model_with_uncertainty = CausalLMWithUncertainty(
        llm=llm,
        tokenizer=tokenizer,
        stat_calculators=stat_calculators,
        estimator=estimator,
    )

    return model_with_uncertainty

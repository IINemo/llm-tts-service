import numpy as np
import torch
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.utils.causal_lm_with_uncertainty import CausalLMWithUncertainty
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.model_adapters import WhiteboxModel
from lm_polygraph.stat_calculators.extract_claims import Claim

from luh.luh_claim_estimator_dummy import LuhClaimEstimatorDummy
from luh.calculator_infer_luh import CalculatorInferLuh
from luh.calculator_apply_uq_head import CalculatorApplyUQHead
from luh.auto_uncertainty_head import AutoUncertaintyHead


class ReasonStepClaimExtractor(StatCalculator):
    def __init__(self):
        pass

    def __call__(
        self,
        deps: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        deps["claims"] = [
            [
                Claim(
                    claim_text=text,
                    sentence=text,
                    aligned_token_ids=list(range(len(deps["greedy_tokens"][i]))),
                )
                for i, text in enumerate(texts)
            ]
        ]
        return deps


def create_uncertainty_model(config):
    llm = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    llm = llm.to(config.model.device)

    uhead_cfg = config.scorer.uncertainty.cfg
    uncertainty_head = AutoUncertaintyHead.from_pretrained(
        uhead_cfg.uq_head_path, base_model=llm
    )

    calc_infer_llm = CalculatorInferLuh(
        uncertainty_head,
        tokenize=True,
        # args_generate=args_generate, TODO:
        device="cuda" if torch.cuda.is_available() else "cpu",
        generations_cache_dir="",
        predict_token_uncertainties=False,
    )
    calc_apply_uhead = CalculatorApplyUQHead(uncertainty_head)
    stat_calculators = [calc_infer_llm, ReasonStepClaimExtractor(), calc_apply_uhead]
    estimator = LuhClaimEstimatorDummy()
    llm_with_uncertainty = CausalLMWithUncertainty(
        llm, tokenizer, stat_calculators, estimator
    )

    return llm_with_uncertainty

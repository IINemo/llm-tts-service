from .strategy_base import StrategyBase
from typing import Dict, Any
import torch
import logging

log = logging.getLogger(__name__)


class StrategyChainOfThought(StrategyBase)  :
    """
    Basic Chain-of-Thought strategy for comparison.
    Generates a single reasoning path with step-by-step thinking.
    """

    def __init__(
        self,
        model,
        max_new_tokens: int = 512,
        temperature: float = 0.1  # Lower temperature for more deterministic reasoning
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a single Chain-of-Thought reasoning path.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information
        """
        log.info(f"Starting Chain-of-Thought reasoning for prompt: {prompt[:100]}...")

        # Check if this is an API model
        if hasattr(self.model, 'api_model') or self.model.device == "api":
            log.info("Using API model for CoT generation")
            try:
                if hasattr(self.model, 'generate'):
                    # Direct API model
                    completions = self.model.generate(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        num_return_sequences=1
                    )
                    generated_text = completions[0] if completions else ""
                else:
                    # API model wrapped in adapter
                    completions = self.model.api_model.generate(
                        prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        num_return_sequences=1
                    )
                    generated_text = completions[0] if completions else ""

                # Combine with prompt
                full_reasoning = prompt + generated_text

            except Exception as e:
                log.error(f"Failed to generate CoT reasoning: {e}")
                full_reasoning = prompt + f" [Generation failed: {str(e)}]"

        else:
            # Use local model
            log.info("Using local model for CoT generation")
            # Tokenize prompt
            inputs = self.model.tokenize([prompt])
            input_ids = inputs['input_ids'].to(self.model.device)
            attention_mask = inputs['attention_mask'].to(self.model.device)

            # Generate single reasoning path
            with torch.no_grad():
                outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=1,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    eos_token_id=self.model.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )

            # Decode generated reasoning
            output_seq = outputs[0]
            new_tokens = output_seq[input_ids.shape[1]:]
            generated_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Combine with prompt
            full_reasoning = prompt + generated_text

        log.info("Chain-of-Thought reasoning completed")

        return {
            "trajectory": full_reasoning,
            "steps": [full_reasoning],
            "completed": True,
            "strategy": "chain_of_thought",
            "metadata": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens
            }
        }

    def cleanup(self):
        """Clean up resources (no-op for basic CoT)"""
        pass

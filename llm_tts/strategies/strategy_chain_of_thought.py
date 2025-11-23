import logging
from typing import Any, Dict, Optional, Tuple

import torch

from llm_tts.models.blackboxmodel_with_streaming import BlackboxModelWithStreaming

from .strategy_base import StrategyBase

log = logging.getLogger(__name__)


class StrategyChainOfThought(StrategyBase):
    """
    Basic Chain-of-Thought strategy for comparison.
    Generates a single reasoning path with step-by-step thinking.
    """

    def __init__(
        self,
        model,
        max_new_tokens: int = 512,
        temperature: float = 0.1,  # Lower temperature for more deterministic reasoning
    ):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _generate_single_reasoning(
        self, args: Tuple[Any, str]
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a single Chain-of-Thought reasoning path (for parallel_execute support).

        This method wraps the generation logic to enable automatic client recreation
        on API timeouts via parallel_execute().

        Args:
            args: Tuple of (messages, prompt_text) where:
                  - messages: Chat format messages for API models
                  - prompt_text: Original prompt text for local models

        Returns:
            Tuple of (generated_text, prompt_text) if successful, None if failed
        """
        messages, prompt_text = args

        try:
            # Check if this is an API model (BlackboxModelWithStreaming)
            if isinstance(self.model, BlackboxModelWithStreaming):
                results = self.model.generate_texts(
                    chats=[messages],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )

                if results and results[0] and results[0].get("text"):
                    generated_text = results[0]["text"]
                    return (generated_text, prompt_text)
                else:
                    log.warning("Empty generation from model")
                    return None

            else:
                # Use local model
                # Tokenize prompt
                inputs = self.model.tokenize([prompt_text])
                # Get device from model - handle quantized models
                device = next(self.model.model.parameters()).device
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

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
                        repetition_penalty=1.1,
                    )

                # Decode generated reasoning
                output_seq = outputs[0]
                new_tokens = output_seq[input_ids.shape[1] :]
                generated_text = self.model.tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                )

                return (generated_text, prompt_text)

        except Exception as e:
            log.error(f"Failed to generate CoT reasoning: {e}")
            return None

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a single Chain-of-Thought reasoning path.

        Uses parallel_execute() with n_threads=1 to enable automatic client
        recreation on API timeouts, while maintaining sequential execution.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information
        """
        log.info(f"Starting Chain-of-Thought reasoning for prompt: {prompt[:100]}...")

        # Prepare arguments for generation
        if isinstance(prompt, list):
            messages = prompt
            # Extract user content from messages for prompt text
            prompt_text = next(
                (msg["content"] for msg in prompt if msg["role"] == "user"),
                str(prompt),
            )
        else:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = prompt

        # Use parallel_execute with n_threads=1 for automatic client recreation
        # This provides robust timeout handling without changing execution order
        results = self._parallel_generate(
            worker_func=self._generate_single_reasoning,
            task_args=[(messages, prompt_text)],
            n_threads=1,  # Sequential execution, but with retry logic
            desc="Generating CoT reasoning",
            model=self.model,  # Critical: enables recreate_client() on timeout
        )

        # Process results
        if results and len(results) > 0:
            generated_text, prompt_text = results[0]
            full_reasoning = prompt_text + "\n\n" + generated_text
            log.info("Chain-of-Thought reasoning completed")
        else:
            # Generation failed (timeout or other error)
            log.error("Failed to generate CoT reasoning (no results returned)")
            full_reasoning = prompt_text + "\n\n[Generation failed: timeout or error]"

        return {
            "trajectory": full_reasoning,
            "steps": [full_reasoning],
            "completed": True,
            "strategy": "chain_of_thought",
            "metadata": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
            },
        }

    def cleanup(self):
        """Clean up resources (no-op for basic CoT)"""
        pass

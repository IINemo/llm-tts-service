"""
Standalone API-based strategies that don't depend on lm-polygraph.
These can be used when you only want to use API models without local model dependencies.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import Counter

from .api_models import TogetherAIModel

log = logging.getLogger(__name__)


class APISelfConsistencyStrategy:
    """
    Self-consistency strategy specifically designed for API models.
    Doesn't depend on lm-polygraph or transformers.
    """

    def __init__(
        self,
        api_model: TogetherAIModel,
        num_paths: int = 10,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        answer_extraction_patterns: Optional[List[str]] = None
    ):
        """
        Initialize API-based self-consistency strategy.

        Args:
            api_model: TogetherAIModel instance
            num_paths: Number of reasoning paths to generate
            max_new_tokens: Maximum tokens per reasoning path
            temperature: Sampling temperature for diversity
            answer_extraction_patterns: Regex patterns to extract answers
        """
        self.api_model = api_model
        self.num_paths = num_paths
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Default answer extraction patterns
        self.answer_patterns = answer_extraction_patterns or [
            r"<Answer>:\s*(.+?)(?:\n|$)",
            r"The answer is\s*(.+?)(?:\n|\.|\$)",
            r"Therefore,?\s*(.+?)(?:\n|\.|\$)",
            r"Final answer:\s*(.+?)(?:\n|\.|\$)",
            r"Answer:\s*(.+?)(?:\n|\.|\$)",
        ]

    def extract_answer(self, text: str) -> str:
        """Extract the final answer from a reasoning chain"""
        import re

        text = text.strip()

        # Try each pattern
        for pattern in self.answer_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                answer = matches[-1].strip()  # Take the last match
                # Clean up the answer
                answer = re.sub(r'[^\w\s\-\+\*\/\(\)\.\,]', '', answer)
                return answer.lower()

        # If no pattern matches, try to extract number from the end
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1].lower()

        # Fallback: return last meaningful word/phrase
        words = text.split()
        if words:
            return words[-1].lower()

        return "unknown"

    def generate_reasoning_paths(self, prompt: str) -> List[str]:
        """
        Generate multiple diverse reasoning paths for the given prompt.

        Args:
            prompt: Input prompt/question

        Returns:
            List of complete reasoning paths (prompt + generated reasoning)
        """
        log.info(f"Generating {self.num_paths} reasoning paths with temperature {self.temperature}")

        all_paths = []

        for i in range(self.num_paths):
            log.info(f"\n{'='*60}")
            log.info(f"🧠 REASONING PATH {i+1}/{self.num_paths}")
            log.info(f"{'='*60}")

            try:
                completions = self.api_model.generate(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    num_return_sequences=1
                )
                generated_text = completions[0] if completions else ""

                # Combine with original prompt
                full_path = prompt + generated_text
                all_paths.append(full_path)

                # Log the reasoning for this path
                log.info(f"📝 Generated reasoning:")
                log.info(f"{'-'*50}")

                # Split reasoning into lines for better readability
                reasoning_lines = generated_text.strip().split('\n')
                for line_num, line in enumerate(reasoning_lines, 1):
                    if line.strip():  # Only log non-empty lines
                        log.info(f"  {line_num:2d}: {line.strip()}")

                log.info(f"{'-'*50}")

                # Extract and log the answer for this path
                answer = self.extract_answer(full_path)
                log.info(f"🎯 Extracted answer from path {i+1}: '{answer}'")

            except Exception as e:
                log.warning(f"❌ Failed to generate path {i+1}: {e}")
                # Add fallback path
                fallback_path = prompt + f" [Generation failed: {str(e)}]"
                all_paths.append(fallback_path)
                log.info(f"🎯 Extracted answer from path {i+1}: 'generation_failed'")

        log.info(f"\n✅ Generated {len(all_paths)} reasoning paths total")
        return all_paths

    def select_best_answer(self, reasoning_paths: List[str]) -> Dict[str, Any]:
        """
        Select the best answer using majority voting across reasoning paths.

        Args:
            reasoning_paths: List of complete reasoning paths

        Returns:
            Dictionary containing best answer and consensus information
        """
        if not reasoning_paths:
            return {
                "best_path": "",
                "best_answer": "no_answer",
                "consensus_score": 0.0,
                "all_answers": [],
                "answer_distribution": {}
            }

        log.info(f"\n{'='*60}")
        log.info(f"🗳️  MAJORITY VOTING ANALYSIS")
        log.info(f"{'='*60}")

        # Extract answers from all paths
        all_answers = [self.extract_answer(path) for path in reasoning_paths]

        # Count answer frequencies
        answer_counts = Counter(all_answers)
        total_paths = len(reasoning_paths)

        # Log detailed voting analysis
        log.info(f"📊 Voting results:")
        for i, answer in enumerate(all_answers, 1):
            log.info(f"  Path {i}: '{answer}'")

        log.info(f"\n📈 Answer frequency distribution:")
        for answer, count in answer_counts.most_common():
            percentage = (count / total_paths) * 100
            log.info(f"  '{answer}': {count}/{total_paths} paths ({percentage:.1f}%)")

        # Find the most frequent answer
        most_common_answer = answer_counts.most_common(1)[0][0]
        most_common_count = answer_counts[most_common_answer]
        consensus_score = most_common_count / total_paths

        # Find a path that has the most common answer
        best_path_idx = None
        for i, answer in enumerate(all_answers):
            if answer == most_common_answer:
                best_path_idx = i
                break

        best_path = reasoning_paths[best_path_idx] if best_path_idx is not None else reasoning_paths[0]

        log.info(f"\n🏆 FINAL DECISION:")
        log.info(f"  Selected answer: '{most_common_answer}'")
        log.info(f"  Consensus strength: {consensus_score:.3f} ({most_common_count}/{total_paths} paths)")
        log.info(f"  Selected reasoning path: #{best_path_idx + 1}")

        # Show confidence level
        if consensus_score >= 0.8:
            confidence_level = "Very High"
        elif consensus_score >= 0.6:
            confidence_level = "High"
        elif consensus_score >= 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        log.info(f"  Confidence level: {confidence_level}")

        return {
            "best_path": best_path,
            "best_answer": most_common_answer,
            "consensus_score": consensus_score,
            "all_answers": all_answers,
            "answer_distribution": dict(answer_counts),
            "all_paths": reasoning_paths
        }

    def generate_trajectory(self, prompt: str) -> Dict[str, Any]:
        """
        Main entry point for self-consistency reasoning.

        Args:
            prompt: Input prompt/question

        Returns:
            Dictionary with trajectory information compatible with evaluation framework
        """
        log.info(f"Starting API self-consistency reasoning for prompt: {prompt[:100]}...")

        # Generate multiple reasoning paths
        reasoning_paths = self.generate_reasoning_paths(prompt)

        # Select best answer via majority voting
        result = self.select_best_answer(reasoning_paths)

        # Format output to match expected interface
        return {
            "trajectory": result["best_path"],
            "steps": [result["best_path"]],  # Single step containing full reasoning
            "completed": True,
            "strategy": "api_self_consistency",
            "metadata": {
                "num_paths": len(reasoning_paths),
                "consensus_score": result["consensus_score"],
                "answer_distribution": result["answer_distribution"],
                "all_answers": result["all_answers"],
                "selected_answer": result["best_answer"]
            }
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.api_model, 'cleanup'):
            self.api_model.cleanup()


class APIChainOfThoughtStrategy:
    """
    Basic Chain-of-Thought strategy for API models.
    Doesn't depend on lm-polygraph or transformers.
    """

    def __init__(
        self,
        api_model: TogetherAIModel,
        max_new_tokens: int = 512,
        temperature: float = 0.1  # Lower temperature for more deterministic reasoning
    ):
        self.api_model = api_model
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
        log.info(f"\n{'='*60}")
        log.info(f"🤔 CHAIN-OF-THOUGHT REASONING")
        log.info(f"{'='*60}")
        log.info(f"📤 Prompt: {prompt[:100]}...")

        try:
            completions = self.api_model.generate(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_return_sequences=1
            )
            generated_text = completions[0] if completions else ""

            # Combine with prompt
            full_reasoning = prompt + generated_text

            # Log the detailed reasoning
            log.info(f"\n📝 Generated reasoning:")
            log.info(f"{'-'*50}")

            # Split reasoning into lines for better readability
            reasoning_lines = generated_text.strip().split('\n')
            for line_num, line in enumerate(reasoning_lines, 1):
                if line.strip():  # Only log non-empty lines
                    log.info(f"  {line_num:2d}: {line.strip()}")

            log.info(f"{'-'*50}")

            # Try to extract final answer using the same patterns
            import re
            answer_patterns = [
                r"<Answer>:\s*(.+?)(?:\n|$)",
                r"The answer is\s*(.+?)(?:\n|\.|\$)",
                r"Therefore,?\s*(.+?)(?:\n|\.|\$)",
                r"Final answer:\s*(.+?)(?:\n|\.|\$)",
                r"Answer:\s*(.+?)(?:\n|\.|\$)",
            ]

            final_answer = "unknown"
            text = full_reasoning.strip()

            # Try each pattern
            for pattern in answer_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
                if matches:
                    answer = matches[-1].strip()
                    answer = re.sub(r'[^\w\s\-\+\*\/\(\)\.\,]', '', answer)
                    final_answer = answer.lower()
                    break

            # Fallback to number extraction
            if final_answer == "unknown":
                numbers = re.findall(r'-?\d+\.?\d*', text)
                if numbers:
                    final_answer = numbers[-1].lower()
            log.info(f"🎯 Extracted final answer: '{final_answer}'")

        except Exception as e:
            log.error(f"❌ Failed to generate CoT reasoning: {e}")
            full_reasoning = prompt + f" [Generation failed: {str(e)}]"
            final_answer = "generation_failed"

        log.info(f"✅ Chain-of-Thought reasoning completed")

        return {
            "trajectory": full_reasoning,
            "steps": [full_reasoning],
            "completed": True,
            "strategy": "api_chain_of_thought",
            "metadata": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "extracted_answer": final_answer
            }
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.api_model, 'cleanup'):
            self.api_model.cleanup()


def create_api_self_consistency_strategy(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key: Optional[str] = None,
    num_paths: int = 10,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> APISelfConsistencyStrategy:
    """
    Factory function to create an API-based self-consistency strategy.

    Args:
        model_name: Together AI model identifier
        api_key: API key (if None, uses TOGETHER_API_KEY env var)
        num_paths: Number of reasoning paths
        max_new_tokens: Max tokens per path
        temperature: Sampling temperature

    Returns:
        APISelfConsistencyStrategy instance
    """
    api_model = TogetherAIModel(model_name=model_name, api_key=api_key)
    return APISelfConsistencyStrategy(
        api_model=api_model,
        num_paths=num_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )


def create_api_chain_of_thought_strategy(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1
) -> APIChainOfThoughtStrategy:
    """
    Factory function to create an API-based chain-of-thought strategy.

    Args:
        model_name: Together AI model identifier
        api_key: API key (if None, uses TOGETHER_API_KEY env var)
        max_new_tokens: Max tokens per reasoning
        temperature: Sampling temperature

    Returns:
        APIChainOfThoughtStrategy instance
    """
    api_model = TogetherAIModel(model_name=model_name, api_key=api_key)
    return APIChainOfThoughtStrategy(
        api_model=api_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
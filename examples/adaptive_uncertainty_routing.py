"""
Adaptive Uncertainty Routing with ChatTTS.

This script demonstrates how to use ChatTTS with conditional routing
based on uncertainty scores. If uncertainty is high, it automatically
retries with increased budget.

Usage:
    python examples/adaptive_uncertainty_routing.py

Requirements:
    - service_app running at localhost:8001
    - OPENROUTER_API_KEY set in environment
"""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from llm_tts.integrations import ChatTTS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """Result from adaptive routing."""
    answer: str
    content: str
    confidence: float
    uncertainty: float
    status: Literal["accepted", "escalated"]
    attempts: int
    final_budget: int
    all_attempts: list


def adaptive_tts_call(
    question: str,
    initial_budget: int = 3,
    max_budget: int = 15,
    budget_multiplier: float = 2.0,
    uncertainty_threshold: float = 0.3,
    max_attempts: int = 3,
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.7,
    service_url: str = "http://localhost:8001/v1",
) -> RoutingResult:
    """
    Call ChatTTS with adaptive budget based on uncertainty.

    If uncertainty is above threshold, retry with increased budget.

    Args:
        question: The question to answer
        initial_budget: Starting number of paths (default: 3)
        max_budget: Maximum budget to try (default: 15)
        budget_multiplier: How much to increase budget on retry (default: 2.0)
        uncertainty_threshold: Retry if uncertainty > this (default: 0.3)
        max_attempts: Maximum retry attempts (default: 3)
        model: Model to use
        temperature: Sampling temperature
        service_url: TTS service URL

    Returns:
        RoutingResult with answer, confidence, and routing metadata
    """

    messages = [
        SystemMessage(content="Solve step by step. Put your final answer in \\boxed{}."),
        HumanMessage(content=question),
    ]

    current_budget = initial_budget
    attempts = []

    for attempt in range(1, max_attempts + 1):
        log.info(f"\n{'='*50}")
        log.info(f"Attempt {attempt}/{max_attempts} with budget={current_budget}")
        log.info(f"{'='*50}")

        # Create ChatTTS with current budget
        llm = ChatTTS(
            base_url=service_url,
            model=model,
            tts_strategy="self_consistency",
            tts_budget=current_budget,
            temperature=temperature,
            max_tokens=512,
        )

        # Call the service
        response = llm.invoke(messages)

        # Extract metadata
        meta = response.response_metadata.get("tts_metadata", {})
        uncertainty = meta.get("uncertainty_score", 1.0)
        confidence = meta.get("consensus_score", 0.0)
        selected_answer = meta.get("selected_answer", "no_answer")
        answer_dist = meta.get("answer_distribution", {})

        attempt_info = {
            "attempt": attempt,
            "budget": current_budget,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "answer": selected_answer,
            "distribution": answer_dist,
        }
        attempts.append(attempt_info)

        log.info(f"Answer: {selected_answer}")
        log.info(f"Confidence: {confidence:.2f}")
        log.info(f"Uncertainty: {uncertainty:.2f}")
        log.info(f"Distribution: {answer_dist}")

        # Check if we should accept
        if uncertainty <= uncertainty_threshold:
            log.info(f"\n>>> ACCEPTED: Uncertainty {uncertainty:.2f} <= threshold {uncertainty_threshold}")
            return RoutingResult(
                answer=selected_answer,
                content=response.content,
                confidence=confidence,
                uncertainty=uncertainty,
                status="accepted",
                attempts=attempt,
                final_budget=current_budget,
                all_attempts=attempts,
            )

        # Uncertainty too high - should we retry?
        if attempt < max_attempts:
            new_budget = min(int(current_budget * budget_multiplier), max_budget)
            if new_budget > current_budget:
                log.info(f"\n>>> RETRY: Uncertainty {uncertainty:.2f} > threshold {uncertainty_threshold}")
                log.info(f">>> Increasing budget: {current_budget} -> {new_budget}")
                current_budget = new_budget
            else:
                log.info(f"\n>>> MAX BUDGET REACHED: {current_budget}")
                break

    # Exhausted attempts - escalate
    log.info(f"\n>>> ESCALATED: Max attempts reached, uncertainty still high")

    final_attempt = attempts[-1]
    return RoutingResult(
        answer=final_attempt["answer"],
        content=response.content,
        confidence=final_attempt["confidence"],
        uncertainty=final_attempt["uncertainty"],
        status="escalated",
        attempts=len(attempts),
        final_budget=current_budget,
        all_attempts=attempts,
    )


def main():
    """Demo the adaptive uncertainty routing."""

    print("\n" + "="*60)
    print("ADAPTIVE UNCERTAINTY ROUTING DEMO")
    print("="*60)

    # Test cases with varying difficulty
    test_questions = [
        # Easy - should accept quickly with low budget
        "What is 7 * 8?",

        # Medium - might need higher budget
        "What is 15% of 240?",

        # Harder - likely needs more paths for consensus
        "A train travels at 60 mph for 2.5 hours. How far does it travel?",
    ]

    for question in test_questions:
        print(f"\n{'#'*60}")
        print(f"QUESTION: {question}")
        print(f"{'#'*60}")

        result = adaptive_tts_call(
            question=question,
            initial_budget=3,
            max_budget=15,
            uncertainty_threshold=0.3,
            max_attempts=3,
        )

        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print(f"{'='*60}")
        print(f"Status: {result.status.upper()}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Uncertainty: {result.uncertainty:.2f}")
        print(f"Total attempts: {result.attempts}")
        print(f"Final budget: {result.final_budget}")
        print(f"\nFull response:\n{result.content[:500]}...")


if __name__ == "__main__":
    main()

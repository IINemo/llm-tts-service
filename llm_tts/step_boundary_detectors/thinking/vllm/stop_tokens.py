"""
vLLM stop tokens derived from ThinkingMarkerDetector regex patterns.

Converts regex word boundary patterns (\b) to literal stop strings
for use with vLLM's SamplingParams.stop parameter.
"""

from typing import List, Set


def expand_word_boundary(word: str, include_lowercase: bool = True) -> List[str]:
    """
    Expand a word to literal stop tokens that simulate \b (word boundary).

    For \bword\b pattern, generates variants like:
    - " word " (spaces)
    - "\nWord " (newline + capital)
    - ". Word " (sentence start)
    - ", word " (after comma)
    etc.

    Args:
        word: The word/phrase to expand
        include_lowercase: Include lowercase variants

    Returns:
        List of literal stop tokens
    """
    tokens = set()

    # Capitalize first letter for sentence-start variants
    word_cap = word.capitalize()

    # Common prefixes (what comes before word boundary)
    # Note: Only space/newline to avoid capturing punctuation in step text
    prefixes = [
        " ",  # space
        "\n",  # newline
    ]

    # Common suffixes (what comes after word boundary)
    suffixes = [
        " ",  # space
        ",",  # comma
        ".",  # period
        ":",  # colon
        "\n",  # newline
    ]

    # Generate combinations - only capitalized to avoid mid-sentence matches
    for prefix in prefixes:
        for suffix in suffixes:
            # Only capitalized version (sentence start)
            tokens.add(f"{prefix}{word_cap}{suffix}")

    return sorted(tokens)


def expand_phrase(phrase: str) -> List[str]:
    """
    Expand a multi-word phrase to stop tokens.

    For phrases like "let me", we need:
    - " Let me "
    - "\nLet me "
    - ". Let me "
    etc.
    """
    return expand_word_boundary(phrase, include_lowercase=True)


# =============================================================================
# Marker Categories (matching ThinkingMarkerDetector)
# =============================================================================

SEQUENCE_WORDS = [
    "first",
    "second",
    "third",
    "next",
    "then",
    "finally",
    "lastly",
    "after that",
]

CONCLUSION_WORDS = [
    "so",
    "therefore",
    "thus",
    "hence",
    "consequently",
    "as a result",
    "this means",
    "which means",
    "which gives",
    "this suggests",
    "the answer",
]

THINKING_WORDS = [
    "let me",
    "let's",
    "i need to",
    "i should",
    "i can",
    "i'll",
    "wait",
    "hmm",
    "okay",
    "oh",
    "actually",
    # Extended
    "let me think",
    "let me consider",
    "let me compute",
    "let me try",
    "let me denote",
    "wait but",
    "wait no",
    "wait maybe",
    "wait perhaps",
    "wait, but",
    "so maybe",
    "so perhaps",
    "so,",
    "but since",
    "but maybe",
    "but the",
    "but before",
    "but wait",
    "but how",
    "to find",
    "now,",
]

VERIFICATION_WORDS = [
    "to verify",
    "to check",
    "let's check",
    "let's verify",
    "substituting",
    "plugging in",
    "if we",
    "when we",
]

REASONING_WORDS = [
    "alternatively",
    "for example",
    "similarly",
    "note that",
    "recall that",
    "given that",
    "consider",
    "assume",
    "suppose",
    "we have",
    "we can",
    "we need",
]

CORRECTION_WORDS = [
    "mistake",
    "error",
    "wrong",
]

# Structure markers (already literal, not word-boundary based)
STRUCTURE_TOKENS = [
    "\n\n",  # Paragraph break
    "\n- ",  # Bullet point
    "\n* ",  # Asterisk bullet
    "\n1. ",  # Numbered list
    "\n2. ",
    "\n3. ",
    "\n4. ",
    "\n5. ",
]

# Answer/completion patterns (for trajectory completion)
ANSWER_TOKENS = [
    "</think>",
    "<Answer>:",
    "\n<Answer>:",
    "\\boxed{",
]


def get_stop_tokens(
    use_sequence: bool = True,
    use_conclusion: bool = True,
    use_thinking: bool = True,
    use_verification: bool = True,
    use_reasoning: bool = False,
    use_correction: bool = False,
    use_structure: bool = False,
    custom_words: List[str] = None,
    include_answer_tokens: bool = True,
    max_tokens: int = None,
) -> List[str]:
    """
    Get vLLM stop tokens based on enabled marker categories.

    Matches ThinkingMarkerDetector configuration options.

    Args:
        use_sequence: Include sequence markers
        use_conclusion: Include conclusion markers
        use_thinking: Include thinking markers
        use_verification: Include verification markers
        use_reasoning: Include reasoning markers
        use_correction: Include correction markers
        use_structure: Include structure markers
        custom_words: Additional words to expand
        include_answer_tokens: Include answer completion tokens
        max_tokens: Limit total number of tokens (vLLM may have limits)

    Returns:
        List of stop token strings for vLLM
    """
    all_tokens: Set[str] = set()

    # Collect words from enabled categories
    words = []
    if use_sequence:
        words.extend(SEQUENCE_WORDS)
    if use_conclusion:
        words.extend(CONCLUSION_WORDS)
    if use_thinking:
        words.extend(THINKING_WORDS)
    if use_verification:
        words.extend(VERIFICATION_WORDS)
    if use_reasoning:
        words.extend(REASONING_WORDS)
    if use_correction:
        words.extend(CORRECTION_WORDS)
    if custom_words:
        words.extend(custom_words)

    # Expand each word to stop tokens
    for word in words:
        tokens = expand_word_boundary(word)
        all_tokens.update(tokens)

    # Add structure tokens (already literal)
    if use_structure:
        all_tokens.update(STRUCTURE_TOKENS)

    # Add answer tokens
    if include_answer_tokens:
        all_tokens.update(ANSWER_TOKENS)

    # Convert to sorted list
    result = sorted(all_tokens)

    # Limit if specified
    if max_tokens and len(result) > max_tokens:
        # Prioritize shorter tokens (more likely to match)
        result = sorted(result, key=len)[:max_tokens]

    return result


def get_stop_tokens_compact(
    use_sequence: bool = True,
    use_conclusion: bool = True,
    use_thinking: bool = True,
    use_verification: bool = True,
    use_reasoning: bool = False,
    use_correction: bool = False,
    use_structure: bool = False,
    custom_words: List[str] = None,
) -> List[str]:
    """
    Get compact stop tokens - only newline-prefixed variants.

    This reduces token count significantly while still catching
    most step boundaries (which typically start on new lines).

    Returns stop tokens like:
    - "\nSo "
    - "\nLet me "
    - "\nTherefore,"
    etc.
    """
    all_tokens: Set[str] = set()

    words = []
    if use_sequence:
        words.extend(SEQUENCE_WORDS)
    if use_conclusion:
        words.extend(CONCLUSION_WORDS)
    if use_thinking:
        words.extend(THINKING_WORDS)
    if use_verification:
        words.extend(VERIFICATION_WORDS)
    if use_reasoning:
        words.extend(REASONING_WORDS)
    if use_correction:
        words.extend(CORRECTION_WORDS)
    if custom_words:
        words.extend(custom_words)

    # Only generate newline-prefixed variants
    for word in words:
        word_cap = word.capitalize()
        word_lower = word.lower()

        # Newline + word + space/punct
        for suffix in [" ", ", ", ". ", ":"]:
            all_tokens.add(f"\n{word_cap}{suffix}")
            all_tokens.add(f"\n{word_lower}{suffix}")

        # After sentence ending + newline
        for prefix in [".\n", "!\n", "?\n"]:
            all_tokens.add(f"{prefix}{word_cap} ")

    # Add structure tokens
    if use_structure:
        all_tokens.update(STRUCTURE_TOKENS)

    # Always add answer tokens
    all_tokens.update(ANSWER_TOKENS)

    return sorted(all_tokens)


# Quick test
if __name__ == "__main__":
    # Test expansion
    print("Expansion of 'so':")
    for t in expand_word_boundary("so")[:10]:
        print(f"  {repr(t)}")

    print("\n" + "=" * 50)

    # Test full token list (compact)
    tokens = get_stop_tokens_compact(
        use_sequence=True,
        use_conclusion=True,
        use_thinking=True,
        use_verification=True,
    )
    print(f"\nCompact stop tokens ({len(tokens)} total):")
    for t in tokens[:20]:
        print(f"  {repr(t)}")
    print("  ...")

    print("\n" + "=" * 50)

    # Test full token list
    tokens_full = get_stop_tokens(
        use_sequence=True,
        use_conclusion=True,
        use_thinking=True,
        use_verification=True,
    )
    print(f"\nFull stop tokens ({len(tokens_full)} total)")

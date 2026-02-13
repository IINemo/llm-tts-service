# Garbage Generation Analysis Report

Generated: 2026-02-06 15:21:05

Garbage token set (14 tokens): `🌈`, `蹩`, `ebx`, `Leone`, `SEEK`, `cdr`, `legate`, `witty`, `mę`, `afi`, `uellen`, `ARRANT`, `ponsored`, `isor`

Detection methods: hardcoded_tokens, unicode_anomaly, ngram_repetition, line_repetition, char_class_shift

## Summary

| Dataset | Model | Strategy | Temp | top_p | Paths | Total | Affected | % |
|---------|-------|----------|------|-------|-------|-------|----------|---|
| gaokao2023en | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 385 | 7 | 1.8% |
| math | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 500 | 6 | 1.2% |
| minerva_math | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 272 | 9 | 3.3% |

## Detection Methods Overview

| Method | Samples Flagged | % of Garbage |
|--------|-----------------|--------------|
| unicode_anomaly | 21 | 95.5% |
| hardcoded_tokens | 19 | 86.4% |
| char_class_shift | 19 | 86.4% |
| ngram_repetition | 1 | 4.5% |

## Garbage Token Frequency

| Token | Total Occurrences |
|-------|-------------------|
| cdr | 2150 |
| Leone | 1798 |
| '🌈' | 1359 |
| mę | 1343 |
| afi | 1337 |
| ebx | 1239 |
| SEEK | 1238 |
| legate | 777 |
| isor | 532 |
| '蹩' | 471 |
| uellen | 391 |
| witty | 207 |
| ARRANT | 143 |
| ponsored | 87 |

## Garbage Onset Position

Where in the generated text does garbage first appear?

| Position | Count | % | Interpretation |
|----------|-------|---|----------------|
| first_25% | 3 | 14% ██░░░░░░░░░░░░░ | Early (prompt/tokenizer issue) |
| middle_50% | 19 | 86% ████████████░░░ | Mid-generation (context overflow, attention drift) |
| last_25% | 0 | 0% ░░░░░░░░░░░░░░░ | Late (EOS/stop token issue) |

## Most Repeated N-grams (in garbage samples)

| N-gram | Max Repetitions |
|--------|-----------------|
| Ang Ang Ang | 951 |
| Ang Ang Ang Ang | 950 |
| + 144} = | 15 |
| + 4\vec{a} \cdot | 12 |
| [2026-01-31 13:54:34,540][__main__][INFO] - | 12 |
| \cdot \vec{b} + | 10 |
| [2026-01-31 13:57:35,232][__main__][INFO] - | 10 |
| [2026-01-31 13:54:34,462][__main__][INFO] - | 10 |
| and \(d = | 9 |
| [2026-01-31 13:54:23,454][__main__][INFO] - | 9 |

## Cross-Run Correlation


### By Temperature

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| 0.7 | 22 | 1157 | 1.9% | ███████████████ |

### By Model

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| Qwen2.5-Math-7B-Instruct | 22 | 1157 | 1.9% | ███████████████ |

### By Strategy

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| self_consistency | 22 | 1157 | 1.9% | ███████████████ |

### By top_p

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| 0.8 | 22 | 1157 | 1.9% | ███████████████ |

## Example Garbage Snippets

### Sample 14 (methods: hardcoded_tokens, unicode_anomaly, char_class_shift, onset: 0.26)

```
[2026-01-31 13:57:34,254][__main__][INFO] - Sample 14/385 [2026-01-31 13:57:34,254][__main__][INFO] - Question: Let $S_{n}$ be the sum of the first $n$ members of an arithmetic sequence $\{a_{n}\}$. If $a_{2}+a_{8}=10$ and $a_{4}a_{8}=45$, find $S_{5}$.... [2026-01-31 13:57:34,254][__main__][INFO] -
```

### Sample 25 (methods: hardcoded_tokens, unicode_anomaly, char_class_shift, onset: 0.53)

```
[2026-01-31 13:57:34,334][__main__][INFO] - Sample 25/385 [2026-01-31 13:57:34,334][__main__][INFO] - Question: Vectors $\vec{a}$, $\vec{b}$, and $\vec{c}$ have the following properties: $|\vec{a}|=|\vec{b}|=1$, $|\vec{c}|=\sqrt{2}$, and $\vec{a}+\vec{b}+\vec{c}=\vec{0}$. Find $\cos \langle\vec{a}-\
```

### Sample 91 (methods: unicode_anomaly, onset: 0.02)

```
[2026-01-31 13:57:34,991][__main__][INFO] - Sample 91/385 [2026-01-31 13:57:34,991][__main__][INFO] - Question: In $\triangle ABC$，$\angle A = 60^{\degree}$ ，$BC=1$ ，Point D is the midpoint of AB and point E is the midpoint of CD. Let $\overline{AB} = \overrightarrow{a},\overline{AC} = \overrightarr
```

### Sample 111 (methods: hardcoded_tokens, unicode_anomaly, char_class_shift, onset: 0.28)

```
[2026-01-31 13:57:35,050][__main__][INFO] - Sample 111/385 [2026-01-31 13:57:35,050][__main__][INFO] - Question: A trapezoid has side lengths 24, 25, 26, and 27 in some order. Find its area.... [2026-01-31 13:57:35,050][__main__][INFO] - Gold answer: $612$ [2026-01-31 13:57:35,050][__main__][INFO] -
```

### Sample 154 (methods: hardcoded_tokens, unicode_anomaly, char_class_shift, onset: 0.35)

```
[2026-01-31 13:57:35,230][__main__][INFO] - Sample 154/385 [2026-01-31 13:57:35,230][__main__][INFO] - Question: The exam scores of twenty-four students are in the stem-and-leaf plot shown, where 7 | 2 represents 72 points. Two students, Erica and Makawee, took the test late, and Makawee earned 6 po
```


## LLM Diagnosis

## Degeneration Type

The degeneration types identified in these samples are:

1. **Hardcoded Tokens**: The model is generating text that includes fixed or predefined sequences of tokens.
2. **Unicode Anomaly**: The model is producing text containing unexpected or incorrect Unicode characters.
3. **Char Class Shift**: The model is shifting between different character classes, such as English letters to symbols or numbers, without proper context.

## Likely Root Causes

For each sample, the most probable causes are:

1. **Sample 1 (index=14)**:
   - **Hardcoded Tokens**: The model might be stuck in a loop where it repeatedly generates the same fixed sequence of tokens.
   - **Unicode Anomaly**: There could be an issue with how the model handles certain special characters or encodings.
   - **Char Class Shift**: The model may be transitioning between different character classes incorrectly due to a lack of proper context or training.

2. **Sample 2 (index=91)**:
   - **Unicode Anomaly**: This sample contains incorrect or unexpected Unicode characters, which might indicate an issue with the tokenizer or how the model processes non-ASCII characters.
   - **Char Class Shift**: The model might be switching between different character classes, leading to the insertion of symbols or numbers where they do not belong.

3. **Sample 3 (index=235)**:
   - **Ngram Repetition**: The model is likely repeating short sequences of tokens, possibly due to a lack of diversity in the generated text.
   - **Char Class Shift**: The model might be making transitions between different character classes, such as from mathematical notation to other types of text.

4. **Sample 4 (index=25)**:
   - **Hardcoded Tokens**: Similar to Sample 1, the model is stuck in a loop generating fixed sequences of tokens.
   - **Unicode Anomaly**: There could be an issue with the model's handling of special characters or encodings.
   - **Char Class Shift**: The model might be transitioning between different character classes incorrectly.

5. **Sample 5 (index=111)**:
   - **Hardcoded Tokens**: The model is generating a fixed sequence of tokens that does not align with the question or expected answer.
   - **Unicode Anomaly**: Incorrect or unexpected Unicode characters might be present due to a tokenizer issue.
   - **Char Class Shift**: The model might be making transitions between different character classes, such as from mathematical notation to other types of text.

## Suggested Fixes

### Specific Parameter Changes

1. **Temperature and Top_p**:
   - **Temperature**: Lowering the temperature can reduce randomness and help the model focus on more coherent outputs. Try setting `temperature` to 0.5 or even 0.3.
   - **Top_p**: Adjusting the `top_p` value can also influence the model's behavior. Reducing `top_p` to 0.7 or 0.6 might help mitigate some of the degenerate outputs.

2. **Presence Penalty**:
   - Increase the `presence_penalty` to discourage the model from repeating tokens. Start with a value of 0.5 and adjust as needed.

### Stop Token / EOS Configuration Changes

1. **Stop Sequences**:
   - Ensure that appropriate stop sequences are configured. For example, adding a stop sequence like `</s>` or `EOS` after the end of the question or problem statement can help the model know when to stop generating text.
   - Example: Add a stop sequence at the end of the question or problem statement.

### Post-Processing Strategies

1. **Output Filtering**:
   - Implement post-processing filters to remove hardcoded tokens, unicode anomalies, and char class shifts. For instance, you can use regular expressions to identify and remove patterns that match hardcoded sequences or unexpected characters.
   - Example: Use regex to filter out repeated sequences or specific hardcoded tokens.

2. **Truncation**:
   - Apply truncation to the model's output to ensure it does not generate excessive text. This can help prevent the model from getting stuck in degenerate loops.
   - Example: Limit the maximum length of the generated text to a reasonable number, e.g., 500 characters.

### Additional Considerations

1. **Model-Specific Known Issues**:
   - Check if there are any known issues with the `Qwen2.5-Math-7B-Instruct` model related to the specific degeneration types observed. Sometimes, models trained on large datasets may have specific quirks or limitations.

2. **Context Window Overflow**:
   - Ensure that the context window is not being overflowed, which can cause the model to lose track of the input and generate irrelevant or degenerate text. Adjust the context window size if necessary.

By applying these suggested fixes, you should be able to improve the quality of the generated text and reduce the occurrence of degenerate outputs.


## Recommendations

1. **Unicode anomalies frequent**: The model outputs CJK/emoji/unusual characters. This is common with multilingual models (Qwen, etc.) when sampling is too random. Consider adding a post-processing filter for non-Latin characters, or lowering temperature.


## Dataset: gaokao2023en

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [257f8rag](https://wandb.ai/nlpresearch.group/llm-tts-eval-gaokao2023en/runs/257f8rag)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 385 total, **7** affected (1.8%)
- **Total garbage token occurrences**: 3875
- **Methods triggered**: char_class_shift: 6, hardcoded_tokens: 6, unicode_anomaly: 7
- **Affected sample indices**: 14, 25, 91, 111, 154, 311, 359


## Dataset: math

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [ey8bwcyv](https://wandb.ai/nlpresearch.group/llm-tts-eval-math500/runs/ey8bwcyv)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 500 total, **6** affected (1.2%)
- **Total garbage token occurrences**: 4040
- **Methods triggered**: char_class_shift: 6, hardcoded_tokens: 6, unicode_anomaly: 6
- **Affected sample indices**: 42, 169, 241, 265, 353, 356


## Dataset: minerva_math

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [tcjzph7v](https://wandb.ai/nlpresearch.group/llm-tts-eval-minerva-math/runs/tcjzph7v)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 272 total, **9** affected (3.3%)
- **Total garbage token occurrences**: 5157
- **Methods triggered**: char_class_shift: 7, hardcoded_tokens: 7, ngram_repetition: 1, unicode_anomaly: 8
- **Affected sample indices**: 38, 95, 125, 128, 155, 196, 205, 223, 235

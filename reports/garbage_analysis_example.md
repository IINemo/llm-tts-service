# Garbage Generation Analysis Report

Generated: 2026-02-06 11:44:11

Garbage token set (14 tokens): `🌈`, `蹩`, `ebx`, `Leone`, `SEEK`, `cdr`, `legate`, `witty`, `mę`, `afi`, `uellen`, `ARRANT`, `ponsored`, `isor`

Detection methods: hardcoded_tokens, unicode_anomaly, ngram_repetition, line_repetition, char_class_shift

## Summary

| Dataset | Model | Strategy | Temp | top_p | Paths | Total | Affected | % |
|---------|-------|----------|------|-------|-------|-------|----------|---|
| gaokao2023en | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 385 | 6 | 1.6% |
| math | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 500 | 6 | 1.2% |
| minerva_math | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 272 | 7 | 2.6% |

## Detection Methods Overview

| Method | Samples Flagged | % of Garbage |
|--------|-----------------|--------------|
| hardcoded_tokens | 19 | 100.0% |
| unicode_anomaly | 19 | 100.0% |

## Garbage Token Frequency

| Token | Total Occurrences |
|-------|-------------------|
| cdr | 380 |
| Leone | 285 |
| legate | 209 |
| ebx | 171 |
| mę | 152 |
| afi | 133 |
| isor | 114 |
| '🌈' | 95 |
| '蹩' | 95 |
| SEEK | 76 |
| witty | 57 |
| ARRANT | 38 |
| uellen | 19 |
| ponsored | 19 |

## Garbage Onset Position

Where in the generated text does garbage first appear?

| Position | Count | % | Interpretation |
|----------|-------|---|----------------|
| first_25% | 19 | 100% ███████████████ | Early (prompt/tokenizer issue) |
| middle_50% | 0 | 0% ░░░░░░░░░░░░░░░ | Mid-generation (context overflow, attention drift) |
| last_25% | 0 | 0% ░░░░░░░░░░░░░░░ | Late (EOS/stop token issue) |

## Step-Level Degeneration

Which reasoning step tends to degenerate first?

| Step # | Garbage Count |
|--------|---------------|
| 1 | 19 |

Mean garbage step index: **1.0**

## Correctness Correlation

|  | Correct | Incorrect | Total | Accuracy |
|--|---------|-----------|-------|----------|
| Garbage | 0 | 19 | 19 | 0.0% |
| Clean | 757 | 381 | 1138 | 66.5% |

## Validity Score Comparison

| Group | Mean Validity Score |
|-------|---------------------|
| Garbage samples | 0.103 |
| Clean samples | 13.997 |

## Token Count Comparison

| Group | Mean Output Tokens |
|-------|---------------------|
| Garbage samples | 883 |
| Clean samples | 297 |

## Most Repeated N-grams (in garbage samples)

| N-gram | Max Repetitions |
|--------|-----------------|
| lculations are correct, | 1 |
| are correct, the | 1 |
| correct, the number | 1 |
| lculations are correct, the | 1 |
| are correct, the number | 1 |

## Cross-Run Correlation


### By Temperature

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| 0.7 | 19 | 1157 | 1.6% | ███████████████ |

### By Model

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| Qwen2.5-Math-7B-Instruct | 19 | 1157 | 1.6% | ███████████████ |

### By Strategy

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| self_consistency | 19 | 1157 | 1.6% | ███████████████ |

### By top_p

| Value | Affected | Total | Rate | |
|-------|----------|-------|------|-|
| 0.8 | 19 | 1157 | 1.6% | ███████████████ |

## Example Garbage Snippets

### Sample 379 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.18)

```
lculations are correct, the number of children is indeed \( C = 14 \). However, we need to recheck calculations to reeusector-highlightlegate)> -highlight;"  Leoneursors mę męebxursors witty Leonelegate sw;" ebx蹩 cdr starDMursors sw Leone seed;"  Leoneisor star蹩isor🌈legateARRANT cdr witty sw sw;"  s
```

### Sample 380 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.18)

```
lculations are correct, the number of children is indeed \( C = 14 \). However, we need to recheck calculations to reeusector-highlightlegate)> -highlight;"  Leoneursors mę męebxursors witty Leonelegate sw;" ebx蹩 cdr starDMursors sw Leone seed;"  Leoneisor star蹩isor🌈legateARRANT cdr witty sw sw;"  s
```

### Sample 381 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.18)

```
lculations are correct, the number of children is indeed \( C = 14 \). However, we need to recheck calculations to reeusector-highlightlegate)> -highlight;"  Leoneursors mę męebxursors witty Leonelegate sw;" ebx蹩 cdr starDMursors sw Leone seed;"  Leoneisor star蹩isor🌈legateARRANT cdr witty sw sw;"  s
```

### Sample 382 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.18)

```
lculations are correct, the number of children is indeed \( C = 14 \). However, we need to recheck calculations to reeusector-highlightlegate)> -highlight;"  Leoneursors mę męebxursors witty Leonelegate sw;" ebx蹩 cdr starDMursors sw Leone seed;"  Leoneisor star蹩isor🌈legateARRANT cdr witty sw sw;"  s
```

### Sample 383 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.18)

```
lculations are correct, the number of children is indeed \( C = 14 \). However, we need to recheck calculations to reeusector-highlightlegate)> -highlight;"  Leoneursors mę męebxursors witty Leonelegate sw;" ebx蹩 cdr starDMursors sw Leone seed;"  Leoneisor star蹩isor🌈legateARRANT cdr witty sw sw;"  s
```


## Recommendations

1. **Early degeneration detected**: >50% of garbage starts in the first 25% of text. Check prompt formatting and tokenizer compatibility. The model may not understand the input format.
2. **Unicode anomalies frequent**: The model outputs CJK/emoji/unusual characters. This is common with multilingual models (Qwen, etc.) when sampling is too random. Consider adding a post-processing filter for non-Latin characters, or lowering temperature.
3. **All garbage samples are incorrect**: Garbage generation always leads to wrong answers. Early detection and re-generation could improve accuracy.


## Dataset: gaokao2023en

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [257f8rag](https://wandb.ai/nlpresearch.group/llm-tts-eval-gaokao2023en/runs/257f8rag)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 385 total, **6** affected (1.6%)
- **Total garbage token occurrences**: 582
- **Methods triggered**: hardcoded_tokens: 6, unicode_anomaly: 6
- **Affected sample indices**: 379, 380, 381, 382, 383, 384


## Dataset: math

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [ey8bwcyv](https://wandb.ai/nlpresearch.group/llm-tts-eval-math500/runs/ey8bwcyv)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 500 total, **6** affected (1.2%)
- **Total garbage token occurrences**: 582
- **Methods triggered**: hardcoded_tokens: 6, unicode_anomaly: 6
- **Affected sample indices**: 494, 495, 496, 497, 498, 499


## Dataset: minerva_math

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: [tcjzph7v](https://wandb.ai/nlpresearch.group/llm-tts-eval-minerva-math/runs/tcjzph7v)
- **Strategy**: self_consistency (8 paths)
- **Samples**: 272 total, **7** affected (2.6%)
- **Total garbage token occurrences**: 679
- **Methods triggered**: hardcoded_tokens: 7, unicode_anomaly: 7
- **Affected sample indices**: 265, 266, 267, 268, 269, 270, 271

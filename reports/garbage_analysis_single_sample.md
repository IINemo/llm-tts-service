# Garbage Generation Analysis Report

Generated: 2026-02-06 14:53:09

Garbage token set (14 tokens): `🌈`, `蹩`, `ebx`, `Leone`, `SEEK`, `cdr`, `legate`, `witty`, `mę`, `afi`, `uellen`, `ARRANT`, `ponsored`, `isor`

Detection methods: hardcoded_tokens, unicode_anomaly, ngram_repetition, line_repetition, char_class_shift

## Summary

| Dataset | Model | Strategy | Temp | top_p | Paths | Total | Affected | % |
|---------|-------|----------|------|-------|-------|-------|----------|---|
| math | Qwen2.5-Math-7B-Instruct | self_consistency | 0.7 | 0.8 | 8 | 1 | 1 | 100.0% |

## Detection Methods Overview

| Method | Samples Flagged | % of Garbage |
|--------|-----------------|--------------|
| hardcoded_tokens | 1 | 100.0% |
| unicode_anomaly | 1 | 100.0% |

## Garbage Token Frequency

| Token | Total Occurrences |
|-------|-------------------|
| cdr | 20 |
| Leone | 15 |
| legate | 11 |
| ebx | 9 |
| mę | 8 |
| afi | 7 |
| isor | 6 |
| '🌈' | 5 |
| '蹩' | 5 |
| SEEK | 4 |
| witty | 3 |
| ARRANT | 2 |
| uellen | 1 |
| ponsored | 1 |

## Garbage Onset Position

Where in the generated text does garbage first appear?

| Position | Count | % | Interpretation |
|----------|-------|---|----------------|
| first_25% | 0 | 0% ░░░░░░░░░░░░░░░ | Early (prompt/tokenizer issue) |
| middle_50% | 1 | 100% ███████████████ | Mid-generation (context overflow, attention drift) |
| last_25% | 0 | 0% ░░░░░░░░░░░░░░░ | Late (EOS/stop token issue) |

## Step-Level Degeneration

Which reasoning step tends to degenerate first?

| Step # | Garbage Count |
|--------|---------------|
| 4 | 1 |

Mean garbage step index: **4.0**

## Correctness Correlation

|  | Correct | Incorrect | Total | Accuracy |
|--|---------|-----------|-------|----------|
| Garbage | 0 | 1 | 1 | 0.0% |
| Clean | 0 | 0 | 0 | 0.0% |

## Validity Score Comparison

| Group | Mean Validity Score |
|-------|---------------------|
| Garbage samples | 0.697 |

## Token Count Comparison

| Group | Mean Output Tokens |
|-------|---------------------|
| Garbage samples | 650 |

## Most Repeated N-grams (in garbage samples)

| N-gram | Max Repetitions |
|--------|-----------------|
| - 4 = | 4 |
| C = 14 | 4 |
| sum of the | 3 |
| sum of the ages | 3 |
| of the ages of | 3 |

## Example Garbage Snippets

### Sample 129 (methods: hardcoded_tokens, unicode_anomaly, onset: 0.65)

```
Let's denote the current sum of the ages of the husband and wife by \( P \) and the current sum of the ages of their children by \( C \). According to the problem, we have the following three equations:  1. \( P = 6C \) 2. \( P - 4 = 10(C - 6) \) 3. \( P + 12 = 3(C + 6) \)(since six years from now e
```


## Recommendations

1. **Unicode anomalies frequent**: The model outputs CJK/emoji/unusual characters. This is common with multilingual models (Qwen, etc.) when sampling is too random. Consider adding a post-processing filter for non-Latin characters, or lowering temperature.
2. **All garbage samples are incorrect**: Garbage generation always leads to wrong answers. Early detection and re-generation could improve accuracy.


## Dataset: math

### Qwen2.5-Math-7B-Instruct | temp=0.7 | top_p=0.8

- **Run**: single_sample_test
- **Strategy**: self_consistency (8 paths)
- **Samples**: 1 total, **1** affected (100.0%)
- **Total garbage token occurrences**: 97
- **Methods triggered**: hardcoded_tokens: 1, unicode_anomaly: 1
- **Affected sample indices**: 129

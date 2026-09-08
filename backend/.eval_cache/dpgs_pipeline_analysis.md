# Full DPGA Classification Pipeline Analysis

- Input workbook: `dpgs.csv.xlsx`
- Projects attempted: **2**
- Projects completed: **2**
- Runtime: **0.4 minutes**
- Methods: `{'groq': 2}`
- Projects with both classifier score vectors: **2**

## Executive Summary

This evaluation runs repository URL parsing, repository metadata/README/topic retrieval, LLM summarization, Groq JSON classification, and Aurora classification for every workbook row. The Excel output stores bounded confidence scores from 0 to 1 for every SDG and both models.

### Overall Metrics at 0.5

| Model | Projects | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Groq | 2 | 0.857 | 0.750 | 0.800 | 6 | 1 | 2 |
| Aurora | 2 | 1.000 | 0.250 | 0.400 | 2 | 0 | 6 |

## Per-SDG Comparison

| SDG | Truth support | Groq F1 | Aurora F1 | Groq recall | Aurora recall |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.000 | 1.000 | 0.000 | 1.000 |
| 2 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| 4 | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| 5 | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| 6 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 7 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| 9 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 11 | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 14 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | 1 | 1.000 | 0.000 | 1.000 | 0.000 |
| 17 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

## Confidence and Agreement Insights

- Every exported confidence is clamped to the inclusive range `[0, 1]`.
- `groq_sdgN_confidence` and `aurora_sdgN_confidence` are raw model scores after normalization, not calibrated probabilities.
- `groq_predictions` and `aurora_predictions` list the SDGs passing the production per-SDG threshold gate.
- Rows with an error retain the original workbook data and record the failure in `error` rather than fabricating scores.

## Operational Findings

- The pipeline is repository-aware: the user description and repository-derived summary are sent to both classifiers.
- A Groq failure causes Aurora to become the primary method for that row; both score vectors are retained whenever available.
- A low score does not mean the project is unrelated in an absolute sense; it means the model did not cross the selected decision threshold.
- The workbook's `act_sdg1` through `act_sdg17` columns are treated as multi-label ground truth, so one project may contribute to several SDGs.

## Output Files

- Scored workbook: `C:\Users\subha\proj_chaoss\c4gt\UNSDG-classifier-tool\backend\.eval_cache\dpgs_pipeline_scores.xlsx`
- This analysis: `C:\Users\subha\proj_chaoss\c4gt\UNSDG-classifier-tool\backend\.eval_cache\dpgs_pipeline_analysis.md`
- Resume cache: `C:\Users\subha\proj_chaoss\c4gt\UNSDG-classifier-tool\backend\.eval_cache\dpgs_pipeline_cache.json`

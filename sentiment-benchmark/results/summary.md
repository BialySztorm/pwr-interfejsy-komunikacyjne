# Sentiment/emotion benchmark summary

## Models

| Model | Task | Labels | License | Size | Base model |
|---|---|---|---|---|---|
| herbert_sentiment | sentiment | negative, neutral, positive | CC BY 4.0 | ~130M | allegro/herbert-base-cased |
| bardsai_sentiment | sentiment | negative, neutral, positive | CC BY 4.0 | ~130M | allegro/herbert-base-cased |
| horizon_sentiment | sentiment | negative, neutral, positive | Apache-2.0 | ~141M | jhu-clsp/mmBERT-small |
| emotion_roberta_pl6 | emotion | joy, sadness, anger, fear, none | MIT | ~355M | RoBERTa |
| goemotions_pl_gpt2 | emotion | joy, sadness, anger, fear, none | MIT | ~124M | sdadas/polish-gpt2-small |
| tabularisai_multilingual | emotion | joy, sadness, anger, fear, none | MIT | ~278M | FacebookAI/xlm-roberta-base |

## Quantitative results

| Model | Accuracy | Macro F1 | Weighted F1 | Mean time/text (ms) | Final score |
|---|---|---|---|---|---|
| herbert_sentiment | 0.583 | 0.577 | 0.576 | 23.05 | 0.549 |
| bardsai_sentiment | 0.646 | 0.631 | 0.643 | 22.78 | 0.607 |
| horizon_sentiment | 0.833 | 0.834 | 0.837 | 18.96 | 0.834 |
| emotion_roberta_pl6 | 0.707 | 0.581 | 0.680 | 22.98 | 0.579 |
| goemotions_pl_gpt2 | 0.512 | 0.486 | 0.506 | 24.34 | 0.467 |
| tabularisai_multilingual | 0.780 | 0.751 | 0.787 | 27.19 | 0.644 |

## Inter-model agreement with the reference

| Model | Agreement | Cohen's kappa |
|---|---|---|
| herbert_sentiment | 0.583 | 0.403 |
| bardsai_sentiment | 0.646 | 0.476 |
| horizon_sentiment | 0.833 | 0.746 |
| emotion_roberta_pl6 | 0.707 | 0.619 |
| goemotions_pl_gpt2 | 0.512 | 0.361 |
| tabularisai_multilingual | 0.780 | 0.711 |

## Pairwise model agreement

| Model A | Model B | Task | Agreement | Cohen's kappa |
|---|---|---|---|---|
| herbert_sentiment | bardsai_sentiment | sentiment | 0.729 | 0.566 |
| herbert_sentiment | horizon_sentiment | sentiment | 0.646 | 0.480 |
| bardsai_sentiment | horizon_sentiment | sentiment | 0.646 | 0.474 |
| emotion_roberta_pl6 | goemotions_pl_gpt2 | emotion | 0.512 | 0.371 |
| emotion_roberta_pl6 | tabularisai_multilingual | emotion | 0.585 | 0.468 |
| goemotions_pl_gpt2 | tabularisai_multilingual | emotion | 0.585 | 0.404 |

Error/disputed cases: 37 (see `errors.csv`).

See `plots/` for confusion matrices, per-class F1, subset results, and timing charts.
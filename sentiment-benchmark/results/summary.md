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
| herbert_sentiment | 0.545 | 0.536 | 0.540 | 22.20 | 0.587 |
| bardsai_sentiment | 0.618 | 0.607 | 0.616 | 33.35 | 0.587 |
| horizon_sentiment | 0.836 | 0.840 | 0.840 | 24.18 | 0.777 |
| emotion_roberta_pl6 | 0.717 | 0.595 | 0.685 | 24.06 | 0.591 |
| goemotions_pl_gpt2 | 0.522 | 0.501 | 0.525 | 28.58 | 0.556 |
| tabularisai_multilingual | 0.783 | 0.741 | 0.784 | 23.42 | 0.744 |

## Inter-model agreement with the reference

| Model | Agreement | Cohen's kappa |
|---|---|---|
| herbert_sentiment | 0.545 | 0.356 |
| bardsai_sentiment | 0.618 | 0.442 |
| horizon_sentiment | 0.836 | 0.749 |
| emotion_roberta_pl6 | 0.717 | 0.634 |
| goemotions_pl_gpt2 | 0.522 | 0.376 |
| tabularisai_multilingual | 0.783 | 0.715 |

## Pairwise model agreement

| Model A | Model B | Task | Agreement | Cohen's kappa |
|---|---|---|---|---|
| herbert_sentiment | bardsai_sentiment | sentiment | 0.745 | 0.586 |
| herbert_sentiment | horizon_sentiment | sentiment | 0.618 | 0.446 |
| bardsai_sentiment | horizon_sentiment | sentiment | 0.636 | 0.464 |
| emotion_roberta_pl6 | goemotions_pl_gpt2 | emotion | 0.522 | 0.383 |
| emotion_roberta_pl6 | tabularisai_multilingual | emotion | 0.609 | 0.496 |
| goemotions_pl_gpt2 | tabularisai_multilingual | emotion | 0.609 | 0.436 |

Error/disputed cases: 43 (see `errors.csv`).

See `plots/` for confusion matrices, per-class F1, subset results, and timing charts.
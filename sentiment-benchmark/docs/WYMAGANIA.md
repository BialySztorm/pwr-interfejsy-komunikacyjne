# Requirement-by-requirement status

Status against the exercise brief ("Ćwiczenie 3 — Ocena narzędzi do rozpoznawania
sentymentu i emocji"). 🟢 done · 🟡 partial / pending a real run · 🔴 not started.

| Requirement | Status | Notes |
|---|---|---|
| 2-3 free Polish-capable tools | 🟢 | 3 sentiment models + 3 emotion models (1 required + 2 optional), see `config.yaml` |
| Test set from student-prepared utterances, ≥ 50 texts | 🟢 | real Drive sync: 54 texts from 11 speakers parsed, no warnings — see `results/dataset_stats.json` |
| Same utterances run through all tools | 🟢 | `classify.py --model all` run against the real 54-text dataset; see `results/*/predictions.csv` |
| Reference comparison | 🟢 | `build_dataset.py` derives reference labels from filenames; `evaluate.py` compares against them |
| Accuracy / Precision / Recall / F1 / confusion matrix | 🟢 | `evaluate.py`, per model, per class + macro + weighted |
| Per-emotion-category results | 🟢 | per-class F1 in `evaluate.py`; `f1_per_class_*.png` plot |
| Robustness to negation / irony / ambiguity | 🟢 | subset evaluation (negation, irony, ambiguous) in `evaluate.py` |
| Error / disputed-case identification | 🟢 | `results/errors.csv` |
| Usefulness in a conversational interface | 🟢 | criteria 5, 7, 8 scored manually in `config.yaml` (`manual_scores`), with justification per model |
| Weighted final score | 🟢 | computed per model in `results/summary.md` (`Final score` column), now including the manual criteria 5/7/8 |
| Single utterance vs. larger-set analysis | 🟢 | `classify.py` measures per-text and whole-batch timing separately |
| Classification time | 🟢 | per-text mean/median + batch total in `timing.json` / `summary.md` |
| Report + presentation | 🔴 | not started |

## Known caveats

- The weighted final score (criterion synthesis) uses a documented but inherently
  judgment-based formula for the "computed" criteria that don't have an obvious 0-1 scale
  (robustness, single-vs-batch speedup, speed normalization) — see the docstring at the
  top of `evaluate.py`.
- `goemotions_pl_gpt2` and `tabularisai_multilingual`'s 28-/11-label-to-5-label mappings
  in `config.yaml` are a project judgment call, not derived from the models' own
  documentation — revisit if evaluation results look off for a specific emotion.
- This stage shares the repo's root `.venv` (Python 3.13) instead of a dedicated one —
  see the README's Setup section for why.
- `emotion_roberta_pl6`'s actual `config.json` has no named `id2label` (just
  `LABEL_0..LABEL_5`); the mapping in `config.yaml` was confirmed empirically (one
  unambiguous example sentence per emotion), not read from the model repo.

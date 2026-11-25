# Requirement-by-requirement status

Status against the exercise brief ("Ćwiczenie 3 — Ocena narzędzi do rozpoznawania
sentymentu i emocji"). 🟢 done · 🟡 partial / pending a real run · 🔴 not started.

| Requirement | Status | Notes |
|---|---|---|
| 2-3 free Polish-capable tools | 🟢 | 3 sentiment models + 3 emotion models (1 required + 2 optional), see `config.yaml` |
| Test set from student-prepared utterances, ≥ 50 texts | 🟡 | `sync_drive.py` + `build_dataset.py` built and tested on fixtures; needs a real Drive sync to confirm ≥ 50 |
| Same utterances run through all tools | 🟡 | `classify.py --model all` implemented, not yet run against the real dataset |
| Reference comparison | 🟢 | `build_dataset.py` derives reference labels from filenames; `evaluate.py` compares against them |
| Accuracy / Precision / Recall / F1 / confusion matrix | 🟢 | `evaluate.py`, per model, per class + macro + weighted |
| Per-emotion-category results | 🟢 | per-class F1 in `evaluate.py`; `f1_per_class_*.png` plot |
| Robustness to negation / irony / ambiguity | 🟢 | subset evaluation (negation, irony, ambiguous) in `evaluate.py` |
| Error / disputed-case identification | 🟢 | `results/errors.csv` |
| Usefulness in a conversational interface | 🟡 | criteria 5, 7, 8 are scored manually in `config.yaml` (`manual_scores`) — not filled in yet |
| Weighted final score | 🟡 | formula implemented in `evaluate.py`; weights proposed in `config.yaml`, awaiting approval; manual scores not filled in yet |
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
- Python 3.14 is used instead of this repo's usual 3.15 pin — see the README's Setup
  section for why.

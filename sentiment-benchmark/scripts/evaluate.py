"""Evaluate all classified models against the reference labels in data/dataset.csv.

Produces, per model: accuracy / precision / recall / F1 (per class + macro + weighted)
and a confusion matrix, for sentiment and/or emotion depending on what the model
predicts. Also computes:

  - subset results: negation, emoji, irony, short vs. long, neutral/none (ambiguous)
  - pairwise inter-model agreement (percent agreement + Cohen's kappa) and agreement
    with the reference
  - a weighted final score per model, combining the computed criteria below with the
    manual 0-1 scores from config.yaml (criteria 5, 7, 8)
  - results/errors.csv: every text where at least one model disagrees with the
    reference or with another model, for qualitative review

Final-score methodology (criteria_weights in config.yaml; see that file for the
weights themselves -- this is the formula applied to each, scaled to 0-1):
  1. correctness             = macro-F1 on the model's own task (sentiment or emotion)
  2. per_emotion_effectiveness = mean per-class F1, excluding the "none"/neutral class
  3. robustness               = mean of {negation subset accuracy, ambiguous subset
                                 accuracy, 1 - irony "fooled" rate}
  4. reference_agreement      = Cohen's kappa vs. the reference, rescaled (kappa+1)/2
  5. label_coverage           = manual score from config.yaml
  6. single_vs_batch          = heuristic from batch speedup: min(1, speedup / 5),
                                 speedup = single-pass total time / batch total time
  7. interpretability         = manual score from config.yaml
  8. integration               = manual score from config.yaml
  9. speed                     = inverted min-max normalization of mean per-text time
                                 across all classified models in this run

This is a proposed scoring scheme, not an official rubric -- see config.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["length_words"] = int(row["length_words"])
        row["has_emoji"] = row["has_emoji"] == "True"
        row["has_negation"] = row["has_negation"] == "True"
    return rows


def load_predictions(model_dir: Path) -> dict[str, dict]:
    predictions_path = model_dir / "predictions.csv"
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["score"] = float(row["score"])
        row["time_ms"] = float(row["time_ms"])
    return {row["id"]: row for row in rows}


def load_timing(model_dir: Path) -> dict:
    return json.loads((model_dir / "timing.json").read_text(encoding="utf-8"))


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": report["accuracy"],
        "per_class": {label: report[label] for label in labels},
        "macro": report["macro avg"],
        "weighted": report["weighted avg"],
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "n": len(y_true),
    }


def subset_ids(dataset_rows: list[dict], cfg: dict) -> dict[str, set[str]]:
    max_short_words = cfg["evaluation"]["short_utterance_max_words"]
    subsets: dict[str, set[str]] = {
        "negation": set(),
        "emoji": set(),
        "irony": set(),
        "short": set(),
        "long": set(),
        "ambiguous": set(),
    }
    for row in dataset_rows:
        if row["has_negation"]:
            subsets["negation"].add(row["id"])
        if row["has_emoji"]:
            subsets["emoji"].add(row["id"])
        if row["ref_emotion"] == "irony":
            subsets["irony"].add(row["id"])
        if row["length_words"] <= max_short_words:
            subsets["short"].add(row["id"])
        else:
            subsets["long"].add(row["id"])
        if row["ref_sentiment"] == "neutral" or row["ref_emotion"] == "none":
            subsets["ambiguous"].add(row["id"])
    return subsets


def task_rows(dataset_rows: list[dict], task: str) -> list[dict]:
    """Rows usable for standard classification metrics on a given task.

    Irony has no emotion-space label, so it is excluded from emotion metrics here and
    evaluated separately (see irony_breakdown).
    """
    if task == "emotion":
        return [row for row in dataset_rows if row["ref_emotion"] != "irony"]
    return dataset_rows


def ref_field(task: str) -> str:
    return "ref_sentiment" if task == "sentiment" else "ref_emotion"


def evaluate_model(
    model_name: str,
    task: str,
    dataset_rows: list[dict],
    predictions: dict[str, dict],
    labels: list[str],
    subsets: dict[str, set[str]],
) -> dict:
    rows = task_rows(dataset_rows, task)
    field = ref_field(task)

    y_true = [row[field] for row in rows]
    y_pred = [predictions[row["id"]]["mapped_label"] for row in rows]
    overall = classification_metrics(y_true, y_pred, labels)

    subset_results = {}
    for subset_name in ["negation", "emoji", "short", "long", "ambiguous"]:
        subset = [row for row in rows if row["id"] in subsets[subset_name]]
        if not subset:
            subset_results[subset_name] = None
            continue
        sub_true = [row[field] for row in subset]
        sub_pred = [predictions[row["id"]]["mapped_label"] for row in subset]
        subset_results[subset_name] = classification_metrics(sub_true, sub_pred, labels)

    irony_rows = [row for row in dataset_rows if row["id"] in subsets["irony"]]
    irony_breakdown = None
    if irony_rows:
        irony_breakdown = {
            "n": len(irony_rows),
            "predicted_label_counts": {},
        }
        for row in irony_rows:
            pred = predictions[row["id"]]["mapped_label"]
            irony_breakdown["predicted_label_counts"][pred] = (
                irony_breakdown["predicted_label_counts"].get(pred, 0) + 1
            )

    return {
        "task": task,
        "overall": overall,
        "subsets": subset_results,
        "irony": irony_breakdown,
    }


OPPOSITE_VALENCE = {"negative": "positive", "positive": "negative"}


def irony_fooled_rate(model_name: str, task: str, dataset_rows, predictions, subsets, cfg) -> float | None:
    """Fraction of the irony subset predicted as the opposite-valence (the "fooled") label.

    The expected irony reference is cfg["evaluation"]["irony_ref_sentiment"] (usually
    "negative"); "fooled" means the model predicted the opposite valence instead. Only
    defined for the sentiment task, and only when the reference has an opposite valence
    (not "neutral").
    """
    irony_rows = [row for row in dataset_rows if row["id"] in subsets["irony"]]
    if not irony_rows or task != "sentiment":
        return None
    irony_ref_sentiment = cfg["evaluation"]["irony_ref_sentiment"]
    fooled_label = OPPOSITE_VALENCE.get(irony_ref_sentiment)
    if fooled_label is None:
        return None
    fooled = sum(1 for row in irony_rows if predictions[row["id"]]["mapped_label"] == fooled_label)
    return fooled / len(irony_rows)


def pairwise_agreement(
    model_predictions: dict[str, dict[str, dict]],
    dataset_rows: list[dict],
    task_of_model: dict[str, str],
) -> dict:
    results = {"vs_reference": {}, "vs_each_other": []}

    for model_name, predictions in model_predictions.items():
        task = task_of_model[model_name]
        rows = task_rows(dataset_rows, task)
        field = ref_field(task)
        y_true = [row[field] for row in rows]
        y_pred = [predictions[row["id"]]["mapped_label"] for row in rows]
        agreement = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else None
        kappa = cohen_kappa_score(y_true, y_pred) if y_true else None
        results["vs_reference"][model_name] = {"agreement": agreement, "kappa": kappa}

    names = list(model_predictions.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            if task_of_model[a_name] != task_of_model[b_name]:
                continue
            task = task_of_model[a_name]
            rows = task_rows(dataset_rows, task)
            a_pred = [model_predictions[a_name][row["id"]]["mapped_label"] for row in rows]
            b_pred = [model_predictions[b_name][row["id"]]["mapped_label"] for row in rows]
            if not a_pred:
                continue
            agreement = sum(1 for a, b in zip(a_pred, b_pred) if a == b) / len(a_pred)
            kappa = cohen_kappa_score(a_pred, b_pred)
            results["vs_each_other"].append(
                {"model_a": a_name, "model_b": b_name, "task": task, "agreement": agreement, "kappa": kappa}
            )

    return results


def build_errors_table(
    dataset_rows: list[dict],
    model_predictions: dict[str, dict[str, dict]],
    task_of_model: dict[str, str],
    out_path: Path,
) -> int:
    model_names = list(model_predictions.keys())
    error_rows = []

    for row in dataset_rows:
        row_preds = {}
        mapped_by_task: dict[str, set[str]] = {}
        disagreement = False
        for model_name in model_names:
            task = task_of_model[model_name]
            if task == "emotion" and row["ref_emotion"] == "irony":
                row_preds[model_name] = "n/a (irony)"
                continue
            pred = model_predictions[model_name].get(row["id"])
            if pred is None:
                row_preds[model_name] = "n/a"
                continue
            mapped = pred["mapped_label"]
            row_preds[model_name] = mapped
            mapped_by_task.setdefault(task, set()).add(mapped)
            ref = row[ref_field(task)]
            if mapped != ref:
                disagreement = True

        # Models disagree only if they diverge within the SAME task's label space --
        # comparing a sentiment label to an emotion label would always look "different".
        if any(len(values) > 1 for values in mapped_by_task.values()):
            disagreement = True

        if disagreement:
            error_rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "ref_sentiment": row["ref_sentiment"],
                    "ref_emotion": row["ref_emotion"],
                    **{f"pred_{name}": row_preds[name] for name in model_names},
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "ref_sentiment", "ref_emotion"] + [f"pred_{name}" for name in model_names]
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)
    return len(error_rows)


def weighted_score(
    model_name: str,
    model_cfg: dict,
    metrics: dict,
    timing: dict,
    all_mean_times_ms: dict[str, float],
    agreement: dict,
    subsets: dict,
    dataset_rows: list[dict],
    predictions: dict,
    cfg: dict,
) -> dict:
    weights = cfg["evaluation"]["criteria_weights"]
    manual = cfg["evaluation"]["manual_scores"]
    task = metrics["task"]

    correctness = metrics["overall"]["macro"]["f1-score"]

    per_class = metrics["overall"]["per_class"]
    excluded = {"none", "neutral"}
    per_class_f1s = [v["f1-score"] for label, v in per_class.items() if label not in excluded]
    per_emotion_effectiveness = sum(per_class_f1s) / len(per_class_f1s) if per_class_f1s else correctness

    robustness_components = []
    if metrics["subsets"].get("negation"):
        robustness_components.append(metrics["subsets"]["negation"]["accuracy"])
    if metrics["subsets"].get("ambiguous"):
        robustness_components.append(metrics["subsets"]["ambiguous"]["accuracy"])
    fooled_rate = irony_fooled_rate(model_name, task, dataset_rows, predictions, subsets, cfg)
    if fooled_rate is not None:
        robustness_components.append(1 - fooled_rate)
    robustness = sum(robustness_components) / len(robustness_components) if robustness_components else None

    kappa = agreement["vs_reference"][model_name]["kappa"]
    reference_agreement = (kappa + 1) / 2 if kappa is not None else None

    label_coverage = manual["label_coverage"].get(model_name, {}).get("score")

    speedup = (
        timing["single_pass_total_s"] / timing["batch_total_s"] if timing.get("batch_total_s") else None
    )
    single_vs_batch = min(1.0, speedup / 5) if speedup else None

    interpretability = manual["interpretability"].get(model_name, {}).get("score")
    integration = manual["integration"].get(model_name, {}).get("score")

    mean_times = list(all_mean_times_ms.values())
    this_time = all_mean_times_ms[model_name]
    if len(mean_times) > 1 and max(mean_times) != min(mean_times):
        speed = 1 - (this_time - min(mean_times)) / (max(mean_times) - min(mean_times))
    else:
        speed = 1.0

    components = {
        "correctness": correctness,
        "per_emotion_effectiveness": per_emotion_effectiveness,
        "robustness": robustness,
        "reference_agreement": reference_agreement,
        "label_coverage": label_coverage,
        "single_vs_batch": single_vs_batch,
        "interpretability": interpretability,
        "integration": integration,
        "speed": speed,
    }

    weighted_sum = 0.0
    weight_total = 0.0
    for criterion, value in components.items():
        if value is None:
            continue
        weighted_sum += value * weights[criterion]
        weight_total += weights[criterion]
    final_score = weighted_sum / weight_total if weight_total else None

    return {"components": components, "final_score": final_score}


def plot_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4 + len(labels) * 0.5, 4 + len(labels) * 0.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_f1_per_class(all_metrics: dict[str, dict], labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(all_metrics), 1)
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        f1s = [metrics["overall"]["per_class"].get(label, {}).get("f1-score", 0) for label in labels]
        positions = [x + i * width for x in range(len(labels))]
        ax.bar(positions, f1s, width=width, label=model_name)
    ax.set_xticks([x + width * (len(all_metrics) - 1) / 2 for x in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1-score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_subset_accuracy(all_metrics: dict[str, dict], subset_names: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(len(all_metrics), 1)
    for i, (model_name, metrics) in enumerate(all_metrics.items()):
        accs = [
            metrics["subsets"][s]["accuracy"] if metrics["subsets"].get(s) else 0
            for s in subset_names
        ]
        positions = [x + i * width for x in range(len(subset_names))]
        ax.bar(positions, accs, width=width, label=model_name)
    ax.set_xticks([x + width * (len(all_metrics) - 1) / 2 for x in range(len(subset_names))])
    ax.set_xticklabels(subset_names)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_timing(timings: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(timings.keys())
    means = [timings[name]["single_pass_mean_ms"] for name in names]
    ax.bar(names, means)
    ax.set_ylabel("Mean per-text time (ms)")
    ax.set_title("Classification time per model")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_md(summary: dict, out_path: Path) -> None:
    lines = ["# Sentiment/emotion benchmark summary", ""]
    lines.append("## Models")
    lines.append("")
    lines.append("| Model | Task | Labels | License | Size | Base model |")
    lines.append("|---|---|---|---|---|---|")
    for model_name, info in summary["models"].items():
        labels = ", ".join(info["labels"])
        lines.append(
            f"| {model_name} | {info['task']} | {labels} | {info['license']} | "
            f"{info['params']} | {info['base_model']} |"
        )

    lines.append("")
    lines.append("## Quantitative results")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | Weighted F1 | Mean time/text (ms) | Final score |")
    lines.append("|---|---|---|---|---|---|")
    for model_name, overall in summary["results"].items():
        timing = summary["timing"][model_name]
        score = summary["final_scores"][model_name]["final_score"]
        score_str = f"{score:.3f}" if score is not None else "n/a"
        lines.append(
            f"| {model_name} | {overall['accuracy']:.3f} | {overall['macro']['f1-score']:.3f} | "
            f"{overall['weighted']['f1-score']:.3f} | {timing['single_pass_mean_ms']:.2f} | {score_str} |"
        )

    lines.append("")
    lines.append("## Inter-model agreement with the reference")
    lines.append("")
    lines.append("| Model | Agreement | Cohen's kappa |")
    lines.append("|---|---|---|")
    for model_name, agreement in summary["agreement"]["vs_reference"].items():
        lines.append(f"| {model_name} | {agreement['agreement']:.3f} | {agreement['kappa']:.3f} |")

    lines.append("")
    lines.append("## Pairwise model agreement")
    lines.append("")
    lines.append("| Model A | Model B | Task | Agreement | Cohen's kappa |")
    lines.append("|---|---|---|---|---|")
    for pair in summary["agreement"]["vs_each_other"]:
        lines.append(
            f"| {pair['model_a']} | {pair['model_b']} | {pair['task']} | "
            f"{pair['agreement']:.3f} | {pair['kappa']:.3f} |"
        )

    lines.append("")
    lines.append(f"Error/disputed cases: {summary['error_count']} (see `errors.csv`).")
    lines.append("")
    lines.append("See `plots/` for confusion matrices, per-class F1, subset results, and timing charts.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classified models against the reference.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "dataset.csv")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}. Run build_dataset.py first.", file=sys.stderr)
        return 1
    dataset_rows = load_dataset(args.dataset)
    subsets = subset_ids(dataset_rows, cfg)

    model_names = [
        name
        for name in cfg["models"]
        if (args.results_dir / name / "predictions.csv").exists()
    ]
    if not model_names:
        print(f"No predictions found under {args.results_dir}. Run classify.py first.", file=sys.stderr)
        return 1

    task_of_model = {name: cfg["models"][name]["task"] for name in model_names}
    model_predictions = {name: load_predictions(args.results_dir / name) for name in model_names}
    timings = {name: load_timing(args.results_dir / name) for name in model_names}
    mean_times_ms = {name: timings[name]["single_pass_mean_ms"] for name in model_names}

    results = {}
    for name in model_names:
        task = task_of_model[name]
        labels = cfg["sentiment_labels"] if task == "sentiment" else cfg["emotion_labels"]
        results[name] = evaluate_model(
            name, task, dataset_rows, model_predictions[name], labels, subsets
        )

    agreement = pairwise_agreement(model_predictions, dataset_rows, task_of_model)

    errors_path = args.results_dir / "errors.csv"
    error_count = build_errors_table(dataset_rows, model_predictions, task_of_model, errors_path)

    final_scores = {}
    for name in model_names:
        final_scores[name] = weighted_score(
            name,
            cfg["models"][name],
            results[name],
            timings[name],
            mean_times_ms,
            agreement,
            subsets,
            dataset_rows,
            model_predictions[name],
            cfg,
        )

    models_info = {}
    for name in model_names:
        model_cfg = cfg["models"][name]
        task = task_of_model[name]
        labels = cfg["sentiment_labels"] if task == "sentiment" else cfg["emotion_labels"]
        models_info[name] = {
            "task": task,
            "labels": labels,
            "license": model_cfg["license"],
            "params": model_cfg["params"],
            "base_model": model_cfg["base_model"],
            "hf_id": model_cfg["hf_id"],
        }

    summary = {
        "models": models_info,
        "results": {name: results[name]["overall"] for name in model_names},
        "subsets": {name: results[name]["subsets"] for name in model_names},
        "irony": {name: results[name]["irony"] for name in model_names},
        "timing": timings,
        "agreement": agreement,
        "final_scores": final_scores,
        "error_count": error_count,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_md(summary, args.results_dir / "summary.md")

    plots_dir = args.results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        task = task_of_model[name]
        overall = results[name]["overall"]
        plot_confusion_matrix(
            overall["confusion_matrix"],
            overall["labels"],
            f"{name} ({task}) confusion matrix",
            plots_dir / f"confusion_{name}.png",
        )

    for task in ["sentiment", "emotion"]:
        task_models = {name: results[name] for name in model_names if task_of_model[name] == task}
        if not task_models:
            continue
        labels = cfg["sentiment_labels"] if task == "sentiment" else cfg["emotion_labels"]
        plot_f1_per_class(task_models, labels, f"F1 per class ({task})", plots_dir / f"f1_per_class_{task}.png")
        plot_subset_accuracy(
            task_models,
            ["negation", "emoji", "short", "long", "ambiguous"],
            f"Subset accuracy ({task})",
            plots_dir / f"subset_accuracy_{task}.png",
        )

    plot_timing(timings, plots_dir / "timing.png")

    print(f"Wrote {args.results_dir / 'summary.json'}, {args.results_dir / 'summary.md'}, {errors_path}")
    print(f"Wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

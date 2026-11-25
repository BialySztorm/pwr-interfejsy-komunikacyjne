"""Run one or more Hugging Face models from config.yaml over data/dataset.csv.

For each model, loads it once (outside any timing), runs a warmup pass, then measures:
  - per-text latency: one forward pass per utterance, timed individually (simulates a
    conversational interface classifying a single incoming message)
  - batch latency: the whole dataset tokenized and classified in one batch (simulates
    offline/bulk analysis of a larger set)

Writes results/<model>/predictions.csv (id, raw_label, raw score distribution, mapped
label, per-text time) and results/<model>/timing.json (hardware info + aggregate timing).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing an expected model or field."""


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset(dataset_path: Path) -> list[dict]:
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def hardware_info(device: str) -> dict:
    info = {
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if device == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def top_label_and_distribution(
    probs: torch.Tensor, id2label: dict[int, str]
) -> tuple[str, float, dict[str, float]]:
    """Return (top raw label, its score, full {label: score} distribution)."""
    distribution = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    top_index = int(torch.argmax(probs).item())
    top_label = id2label[top_index]
    return top_label, float(probs[top_index]), distribution


def map_label(
    raw_label: str,
    score: float,
    label_mapping: dict,
    threshold: float | None,
    model_name: str,
    unmapped_warned: set[str],
) -> str:
    if raw_label in label_mapping:
        mapped = label_mapping[raw_label]
    elif raw_label.lower() in label_mapping:
        mapped = label_mapping[raw_label.lower()]
    else:
        if raw_label not in unmapped_warned:
            print(
                f"WARNING: [{model_name}] raw label '{raw_label}' is not in config.yaml's "
                "label_mapping -- falling back to 'none'. The model's id2label probably "
                "doesn't match what's hardcoded in config.yaml; fix the mapping there."
            )
            unmapped_warned.add(raw_label)
        mapped = "none"
    if threshold is not None and score < threshold:
        return "none"
    return mapped


def classify_model(
    model_name: str,
    model_cfg: dict,
    rows: list[dict],
    device: str,
    results_dir: Path,
) -> None:
    hf_id = model_cfg["hf_id"]
    multilabel = model_cfg.get("multilabel", False)
    threshold = model_cfg.get("multilabel_threshold") if multilabel else None
    label_mapping = model_cfg["label_mapping"]

    print(f"\n=== {model_name} ({hf_id}) ===")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(hf_id)
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    texts = [row["text"] for row in rows]
    activation = torch.sigmoid if multilabel else (lambda x: torch.softmax(x, dim=-1))
    unmapped_warned: set[str] = set()

    # Warmup: not timed, lets lazy kernel/graph init happen outside the measurement.
    if texts:
        with torch.no_grad():
            warmup_start = time.perf_counter()
            inputs = tokenizer(texts[0], truncation=True, padding=True, return_tensors="pt").to(device)
            model(**inputs)
            warmup_s = time.perf_counter() - warmup_start
    else:
        warmup_s = 0.0

    # Per-text timing: one forward pass per utterance.
    predictions: list[dict] = []
    single_pass_times_ms: list[float] = []
    with torch.no_grad():
        for row, text in zip(rows, texts):
            start = time.perf_counter()
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
            logits = model(**inputs).logits[0]
            probs = activation(logits)
            elapsed_ms = (time.perf_counter() - start) * 1000
            single_pass_times_ms.append(elapsed_ms)

            raw_label, score, distribution = top_label_and_distribution(probs, id2label)
            mapped_label = map_label(
                raw_label, score, label_mapping, threshold, model_name, unmapped_warned
            )

            predictions.append(
                {
                    "id": row["id"],
                    "raw_label": raw_label,
                    "score": score,
                    "distribution": json.dumps(distribution, ensure_ascii=False),
                    "mapped_label": mapped_label,
                    "time_ms": elapsed_ms,
                }
            )

    # Batch timing: the whole dataset classified in one batched forward pass.
    batch_total_s = 0.0
    if texts:
        with torch.no_grad():
            batch_start = time.perf_counter()
            batch_inputs = tokenizer(
                texts, truncation=True, padding=True, return_tensors="pt"
            ).to(device)
            model(**batch_inputs)
            batch_total_s = time.perf_counter() - batch_start

    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = model_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "raw_label", "score", "distribution", "mapped_label", "time_ms"]
        )
        writer.writeheader()
        writer.writerows(predictions)

    sorted_times = sorted(single_pass_times_ms)
    n = len(sorted_times)
    timing = {
        **hardware_info(device),
        "model": model_name,
        "hf_id": hf_id,
        "n_texts": n,
        "warmup_s": warmup_s,
        "single_pass_total_s": sum(single_pass_times_ms) / 1000,
        "single_pass_mean_ms": sum(single_pass_times_ms) / n if n else None,
        "single_pass_median_ms": sorted_times[n // 2] if n else None,
        "batch_total_s": batch_total_s,
    }
    (model_dir / "timing.json").write_text(
        json.dumps(timing, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {predictions_path} and {model_dir / 'timing.json'}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify data/dataset.csv with one or all models.")
    parser.add_argument("--model", default="all", help="Model name from config.yaml, or 'all'.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "dataset.csv")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu; cuda is optional and requires a GPU build of torch).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    config = load_config(args.config)
    models = config["models"]

    if args.model == "all":
        selected = list(models.keys())
    elif args.model in models:
        selected = [args.model]
    else:
        print(f"Unknown model '{args.model}'. Known models: {', '.join(models)}", file=sys.stderr)
        return 1

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}. Run build_dataset.py first.", file=sys.stderr)
        return 1
    rows = load_dataset(args.dataset)
    if not rows:
        print(f"Dataset {args.dataset} is empty.", file=sys.stderr)
        return 1

    for model_name in selected:
        classify_model(model_name, models[model_name], rows, args.device, args.results_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

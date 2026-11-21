"""CLI runner: python -m src.evaluate --backend ollama --model llama3:8b"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backends import HuggingFaceBackend, OllamaBackend, OpenAIBackend
from src.metrics import classify_error, compute_perturbation_consistency, compute_summary_metrics
from src.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, TASK_DEFINITION, build_prompt


def build_backend(backend_type: str, model_name: str, base_url: str | None = None):
    if backend_type == "ollama":
        return OllamaBackend(model_name=model_name)
    if backend_type == "hf":
        return HuggingFaceBackend(model_id=model_name)
    if backend_type == "openai":
        return OpenAIBackend(model_name=model_name, base_url=base_url)
    raise ValueError(f"Unknown backend: {backend_type}")


def _json_safe(value):
    """Convert numpy/pandas scalars and NaN/NA to JSON-native Python values."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def run_split(df: pd.DataFrame, backend, model_label: str, strategies: list[str],
              id_col: str, cache_path: Path | None = None) -> pd.DataFrame:
    """Predict every (strategy, item). Finished predictions are appended to a JSONL cache
    so an interrupted run (API error, crash) resumes instead of starting over."""
    cache: dict[tuple[str, str], dict] = {}
    if cache_path is not None and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r.get("strategy"), r.get(id_col))
            if key[0] is not None and key[1] is not None:
                cache[key] = r

    records = []
    for strategy in strategies:
        is_cot = strategy == "cot"
        for _, row in df.iterrows():
            key = (strategy, row[id_col])
            cached = cache.get(key)
            if cached is not None:
                # Guard against stale cache entries if benchmark content changed.
                if cached.get("premise") == row["premise"] and cached.get("hypothesis") == row["hypothesis"]:
                    records.append(cached)
                    continue
            pred = backend.predict(build_prompt(row["premise"], row["hypothesis"], strategy), SYSTEM_PROMPT, is_cot=is_cot)
            rec = row.to_dict()
            rec.update(model=model_label, strategy=strategy, predicted_label=pred.predicted_label,
                       prob_yes=pred.prob_yes, prob_no=pred.prob_no, logprob_margin=pred.logprob_margin,
                       raw_output=pred.raw_text)
            if "category" in rec and "subtype" in rec:
                rec["error_type"] = classify_error(pd.Series(rec))
            rec = _json_safe(rec)
            records.append(rec)
            if cache_path is not None:
                with cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return pd.DataFrame(records)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL)
        return out.strip() or None
    except Exception:
        return None


def _prompt_hash() -> str:
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "task_definition": TASK_DEFINITION,
        "few_shot_examples": FEW_SHOT_EXAMPLES,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _write_manifest(out_dir: Path, args, model_label: str, data_path: Path, pert_path: Path) -> None:
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_label,
        "backend": args.backend,
        "strategies": args.strategies,
        "limit": args.limit,
        "data_path": str(data_path),
        "perturbations_path": str(pert_path),
        "data_sha256": _sha256_file(data_path),
        "perturbations_sha256": _sha256_file(pert_path),
        "prompt_sha256": _prompt_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(Path.cwd()),
        "command": sys.argv,
        "metrics_designation": {
            "primary": ["accuracy", "macro_f1", "balanced_accuracy", "acc_ci_low", "acc_ci_high", "per_category_accuracy_macro_f1", "mcnemar_p_value"],
            "exploratory": ["false_positive_rate", "false_negative_rate", "invalid_rate", "acc_excl_contested", "error_attribution", "robustness_consistency", "logprob_margin"],
        },
    }
    safe = model_label.replace("/", "_").replace(":", "_")
    suffix = f"_limit{args.limit}" if args.limit else ""
    path = out_dir / f"manifest_{safe}{suffix}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMs on simple linguistic inferences.")
    parser.add_argument("--backend", choices=["ollama", "hf", "openai"], required=True)
    parser.add_argument("--model", required=True, help="e.g. llama3:8b, mistral:7b, gpt-4o")
    parser.add_argument("--data", default="data/inference_benchmark.csv")
    parser.add_argument("--perturbations", default="data/perturbation_benchmark.csv")
    parser.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot", "cot"])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="only first N items per split (smoke test)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (or set OPENAI_BASE_URL)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(args.backend, args.model, base_url=args.base_url)
    safe = args.model.replace("/", "_").replace(":", "_")
    suffix = f"_limit{args.limit}" if args.limit else ""

    main_df = pd.read_csv(args.data)
    if args.limit:
        main_df = main_df.head(args.limit)
    preds = run_split(main_df, backend, args.model, args.strategies, "id", out_dir / f".cache_{safe}{suffix}.jsonl")
    preds.to_csv(out_dir / f"predictions_{safe}{suffix}.csv", index=False)
    summary = compute_summary_metrics(preds)
    summary.to_csv(out_dir / f"summary_{safe}{suffix}.csv", index=False)
    print(summary.to_string(index=False))

    pert_path = Path(args.perturbations)
    if pert_path.exists():
        pert_df = pd.read_csv(pert_path)
        if args.limit:
            pert_df = pert_df.head(args.limit)
        pert = run_split(pert_df, backend, args.model, args.strategies, "base_id",
                         out_dir / f".cache_pert_{safe}{suffix}.jsonl")
        pert.to_csv(out_dir / f"perturbation_predictions_{safe}{suffix}.csv", index=False)
        rob = compute_perturbation_consistency(preds, pert)
        rob.to_csv(out_dir / f"robustness_{safe}{suffix}.csv", index=False)

    _write_manifest(out_dir, args, args.model, Path(args.data), pert_path)


if __name__ == "__main__":
    main()

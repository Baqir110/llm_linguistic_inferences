# Evaluating Large Language Models on Simple Linguistic Inferences

**Author:** Muhammad Baqir (Matriculation No. 2184791)  
**Course:** GdI-Proj-B/M: LLMs and Linguistic Theory (WS 2026/27) — University of Bamberg  
**Supervisor:** Dr. Luke Burke

## Overview

This repository implements a controlled benchmark and evaluation pipeline for testing whether language models distinguish strict linguistic inference from plausible but unsupported conclusions. The benchmark contains **180 balanced premise–hypothesis pairs** (60 per category, 30 Yes/30 No each) and **30 truth-preserving lexical perturbations**.

### Categories

1. **Lexical entailment (`lexical_entailment`)** — hypernyms, manner verbs, intersective/privative modifiers, and downward-monotone negation controls.
2. **Presupposition projection (`presupposition`)** — change-of-state verbs, factives, iteratives, clefts, projection under negation/questions, and non-factive/filter controls.
3. **Plausible non-entailment (`plausible_non_entailment`)** — world-plausible but logically unsupported continuations, paired with strictly entailed controls.

## Repository structure

```text
llm_linguistic_inferences/
├── data/
│   ├── inference_benchmark.csv       # 180 gold-annotated, balanced items
│   └── perturbation_benchmark.csv    # 30 truth-preserving paraphrases
├── scripts/
│   ├── run_all.sh                    # Linux/macOS batch runner (resumable)
│   └── run_all.ps1                   # Windows PowerShell batch runner (resumable)
├── src/
│   ├── analyze.py                    # Aggregation, primary/exploratory summaries, plots
│   ├── backends.py                   # Ollama, Hugging Face, and OpenAI backends
│   ├── evaluate.py                   # CLI runner, JSONL cache, run manifest
│   ├── metrics.py                    # Metrics, CIs, McNemar, baselines, error attribution
│   └── prompts.py                    # Shared task definition and prompt builders
├── tests/
│   └── test_pipeline.py              # Unit tests + benchmark integrity + regressions
├── DATA_CARD.md                      # Dataset documentation and contested-item review
├── REPORT.md                         # Detailed project report (draft)
├── Project_Proposal_corrected.md     # Clean corrected proposal
├── pyproject.toml                    # Package metadata and optional extras
└── requirements.txt                  # Convenience dependency list
```

## Setup

### Option A: pip + requirements

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B: pyproject extras

```bash
pip install -e .
pip install -e .[openai]   # OpenAI backend
pip install -e .[hf]       # Hugging Face backend
pip install -e .[dev]      # pytest
pip install -e .[all]      # everything
```

## Design notes

- **Shared task definition:** all strategies use the same task definition so prompt comparisons are not confounded by extra hints.
- **Balanced labels:** 30 Yes / 30 No per category; constant guesser = 50%.
- **Contested items:** 4 theory-dependent items are flagged with `contested` and `contested_note`; report `acc_excl_contested` alongside main metrics.
- **Invalid outputs:** malformed/non-Yes/No outputs count as errors and are reported as `invalid_rate`.
- **CoT parsing:** CoT requires an explicit `Final Answer: Yes|No`; reasoning text alone is scored `Invalid`.
- **Log-probabilities:** direct prompts record `prob_yes`, `prob_no`, and `logprob_margin` where backend support exists. HF gives exact next-token probabilities; Ollama/OpenAI use top-logprob approximations.
- **Resumable runs:** predictions are cached in `results/.cache_*.jsonl`; reruns resume after interruptions. Cache entries are validated against premise/hypothesis content.
- **Run manifest:** each run writes `results/manifest_<model>.json` with command, timestamp, prompt hash, dataset hashes, platform, and git commit when available.
- **Primary vs exploratory metrics:** `src.analyze` writes `primary_summary.csv`, `exploratory_summary.csv`, and `metrics_designation.json`. Primary metrics are intended for main comparisons; exploratory metrics are diagnostic.

## Running experiments

### 1. Ollama (local open-weight models)

```bash
ollama pull llama3:8b
python -m src.evaluate --backend ollama --model llama3:8b
```

### 2. Hugging Face (exact direct log-probabilities)

```bash
python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct
```

### 3. OpenAI API

```bash
# Linux/macOS
export OPENAI_API_KEY="your-key-here"
# Windows PowerShell
# $env:OPENAI_API_KEY="your-key-here"
python -m src.evaluate --backend openai --model gpt-4o
```

### 4. Aggregate results

```bash
python -m src.analyze
```

### 5. Full batch

```bash
# Linux/macOS
bash scripts/run_all.sh
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
```

## Outputs

`src.evaluate` writes per-model prediction/summary/robustness CSVs and a manifest. `src.analyze` writes:

- `master_summary.csv`, `primary_summary.csv`, `exploratory_summary.csv`
- `majority_baselines.csv`, `significance_tests.csv`
- `error_attribution.csv`, `robustness_summary.csv`
- `metrics_designation.json`
- `category_accuracy_comparison.png`, `fp_fn_rates.png`

## Tests

```bash
pytest -v
```

The test suite covers prompt behavior, CoT parsing, logprob parsing, metrics, benchmark integrity, perturbation integrity, cache round-trip/type safety, HF guard behavior, and OpenAI whitespace-token handling.

## Documentation

- `DATA_CARD.md` — dataset composition, labeling protocol, contested-item second-pass review, limitations.
- `REPORT.md` — detailed draft report with methodology, reproducibility checklist, and execution note.
- `Project_Proposal_corrected.md` — cleaned and corrected proposal.

## Limitations

- Real runs included so far: `llama3:latest`, `mistral:7b`, `mixtral:8x7b`, and `qwen2.5:7b` (extra) via Ollama (full 180 items × 3 strategies + 30 perturbations). `llama3.3:70b` was attempted but failed with Ollama 500 (likely resource limit). OpenAI GPT models blocked by API quota; optional HF exact-logit check not run.
- No independent human inter-annotator agreement study is included; contested items are documented as a sensitivity protocol.
- Backend log-probability comparability differs (exact logits vs. top-logprob approximations).
- The dataset is small and should not be used as a general certification of model capability.

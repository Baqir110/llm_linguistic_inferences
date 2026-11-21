# Project Proposal: Evaluating LLMs on Simple Linguistic Inferences

**Muhammad Baqir (2184791)** | M.Sc. ISSS, University of Bamberg | GdI-Proj-B/M: LLMs and Linguistic Theory (WS 2026/27) | Supervisor: Dr. Luke Burke

## Problem & Goal
LLMs can sound fluent but still confuse what must be true from a sentence with what is only likely. This project tests whether models can make simple linguistic inferences correctly, based on Basmov et al. (2024).

## Research Questions
1. Can selected LLMs tell strict inference apart from plausible but unsupported claims?
2. Do zero-shot, few-shot, and chain-of-thought prompts change results?
3. Where do errors come from: world-knowledge bias, negation/presupposition failure, or lexical mistakes?

## Dataset (180 balanced items + 30 perturbations)
60 items/category, 30 Yes/30 No each. 4 theory-dependent items marked `contested` and reported with/without sensitivity.

| Category | Premise | Hypothesis | Gold | Why |
|---|---|---|---:|---|
| Lexical entailment | "The shopper purchased a sedan." | "The shopper purchased an automobile." | Yes | Hypernym relation |
| Lexical entailment | "The child did not eat a fruit." | "The child did not eat an apple." | No | Negation blocks inference |
| Presupposition | "John did not stop smoking." | "John used to smoke." | Yes | Presupposition survives negation |
| Plausible non-entailment | "The runner collapsed after crossing the finish line." | "The runner had a heart attack." | No | Plausible, not required |

## Models & Method
**Models:** Llama 3 8B/70B, Mistral 7B, Mixtral 8x7B (Ollama); GPT-4o, GPT-3.5-turbo (OpenAI); optional HF Llama-3-8B-Instruct for exact Yes/No logits.  
**Prompts:** same task definition for zero-shot, few-shot (4 demos outside benchmark), CoT (`Final Answer: Yes/No`). Temp=0, fixed seed where supported. Invalid/missing final answer = error.  
**Outputs:** raw predictions, JSONL cache, run manifest (hashes, command, model, time, git commit).

## Metrics & Plan
**Primary:** Accuracy + 95% bootstrap CI, Macro-F1, balanced accuracy, per-category Acc/F1, exact McNemar. **Exploratory:** FP/FN, invalid rate, acc_excl_contested, heuristic error attribution, perturbation consistency.  
**Timeline:** Nov 18–30 setup/data; Dec 1–24 pipeline; Jan 1–20 runs; Jan 21–Feb 5 analysis/report.

**References:** Basmov, Goldberg, Tsarfaty (2024), arXiv:2305.14785; Fox & Katzir (2024), *Theoretical Linguistics* 50(1-2); Futrell & Mahowald (2026), *BBS* 49, e198; Piantadosi (2023), Lingbuzz 007180.
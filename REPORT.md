# Project Report: Testing Language Models on Simple Language Inferences

**Student:** Muhammad Baqir (2184791)  
**Program:** M.Sc. International Software Systems Science  
**Course:** GdI-Proj-B/M: LLMs and Linguistic Theory (WS 2026/27)  
**Supervisor:** Dr. Luke Burke  
**University:** University of Bamberg  
**Date:** 21 May 2026

## 1. What this project is about

Language models can write text that sounds very good. But sounding good is not the same as understanding what a sentence really means. In this project, I wanted to check a simple thing: when a model is given one sentence (the premise) and another sentence (the hypothesis), can it tell whether the second sentence must be true, or whether it only sounds likely?

This matters because many model mistakes come from guessing what is common in the real world instead of following what the words actually say. I built a small benchmark and a Python testing pipeline to check this in a clear and repeatable way.

## 2. Research questions

I worked with these main questions:

1. Can selected language models tell the difference between a real inference and a plausible but wrong one?
2. Does the prompt style change the result? I tested zero-shot, few-shot, and chain-of-thought prompts.
3. Do bigger or proprietary models do better than smaller open models?
4. What kind of mistakes do models make? For example, do they over-accept likely statements, or do they miss presuppositions and negation?

## 3. The dataset I used

The main benchmark has **180 items**. Each item has a premise, a hypothesis, and a gold label (`Yes` or `No`).

The data is balanced:
- 60 lexical entailment items (30 Yes, 30 No)
- 60 presupposition items (30 Yes, 30 No)
- 60 plausible non-entailment items (30 Yes, 30 No)

I also made **30 perturbation items**. These are small rewrites of original items. The meaning stays the same, so the gold label should stay the same. This helps check if a model is stable when wording changes.

### Examples

| Category | Premise | Hypothesis | Gold | Simple reason |
|---|---|---|---:|---|
| Lexical entailment | The shopper purchased a sedan. | The shopper purchased an automobile. | Yes | A sedan is an automobile. |
| Lexical entailment | The child did not eat a fruit. | The child did not eat an apple. | No | Not eating any fruit does not mean not eating an apple. |
| Presupposition | John did not stop smoking. | John used to smoke. | Yes | "Stopped" still suggests he smoked before. |
| Presupposition | Tom believes that the shop is open. | The shop is open. | No | Believing something does not make it true. |
| Plausible non-entailment | The runner collapsed after crossing the finish line. | The runner had a heart attack. | No | It is possible, but not certain. |
| Plausible non-entailment | The firefighters extinguished the blaze. | There was a fire. | Yes | You cannot extinguish a blaze if there was no fire. |

### Contested items

Four items are marked as contested because language experts may argue about them. I kept them in the data, but I marked them clearly. The report should show results with all items and also without these contested items.

| ID | Category | Gold | Why it is debated |
|---|---|---:|---|
| INF-017 | Lexical entailment | Yes | "Every" can be tricky when there are no students at all. |
| INF-066 | Presupposition | Yes | "The king of France" is a famous philosophy/language case. |
| INF-076 | Presupposition | Yes | "Managed to" may show difficulty, but some people call it an implicature, not a strict entailment. |
| INF-118 | Presupposition | No | Counterfactual sentences often suggest the opposite happened, but that suggestion can be canceled. |

## 4. How I built the evaluation

### Prompt styles

I used the same task definition in all three prompt styles so the comparison is fair:

- **Zero-shot:** give the task rule and ask for Yes/No.
- **Few-shot:** give the same rule plus 4 example answers. These examples are not in the benchmark.
- **Chain-of-thought:** give the same rule, ask for short reasoning, and require the last line to be `Final Answer: Yes` or `Final Answer: No`.

### Models and backends

The code supports these backends:

- **Ollama** for local open-weight models like Llama 3 and Mistral.
- **Hugging Face** for local transformer models, with exact Yes/No token probability for direct prompts.
- **OpenAI API** for models like GPT-4o.

I ran real local tests through Ollama on four models: `llama3:latest`, `mistral:7b`, `mixtral:8x7b`, and `qwen2.5:7b` (qwen was an extra model, not in the original proposal). I also ran `openai/gpt-oss-20b` via Groq (OpenAI-compatible API). I tried a free OpenRouter model, but the current key in `apikeys.env` did not authenticate for OpenRouter chat completions. I ran the full 180-item benchmark with all three prompt styles, plus the 30 perturbation items. I did not run all models named in the original proposal (still missing `llama3.3:70b` run, OpenAI GPT models due to API quota, and optional HF exact-logit check pending approval). The numbers below come from those saved result files.

### How answers are scored

- For normal prompts, the code reads the Yes/No answer.
- For chain-of-thought, only `Final Answer: Yes` or `Final Answer: No` is accepted. Anything else is marked `Invalid`.
- Invalid answers count as wrong, but they are also counted separately so we can see format problems.

## 5. Metrics

I split metrics into two groups so the main results stay clear.

### Main metrics
These are the main numbers I would use to compare models:
- Accuracy
- 95% bootstrap confidence interval for accuracy
- Macro-F1
- Balanced accuracy
- Per-category accuracy and Macro-F1
- McNemar test for paired model/prompt comparisons

### Extra diagnostic metrics
These are useful, but they should be read with care:
- False positive rate (model says Yes too often)
- False negative rate (model says No too often)
- Invalid rate
- Accuracy without contested items
- Heuristic error type labels
- Perturbation consistency

## 6. What I checked and fixed in the code

I ran the tests before and after my fixes. The final result was:

```text
13 passed
```

The main bugs I found and fixed were:

1. **Cache type bug:** Old cache saving could turn some values into strings. I fixed it so booleans, numbers, and missing values stay correct.
2. **Stale cache bug:** If a benchmark item changed but the ID stayed the same, old cache could be reused by mistake. I added a check for premise and hypothesis.
3. **Hugging Face crash:** If a tokenizer had no single-token Yes/No option, the code crashed in a confusing way. I added a clear error message.
4. **Ollama logprob bug:** Sometimes the chosen Yes/No token was not inside top logprobs. I fixed the parser to include it.
5. **OpenAI logprob bug:** The first token can be a blank line or space. I changed the code to skip empty tokens.
6. **Contested item bug:** "False" as a string can be read as true in Python. I fixed the contested check so it reads values safely.
7. **Metrics and analysis cleanup:** I used `nanmean` where needed, removed an unused import, made analysis ignore duplicate rows, and made it safer when `error_type` is missing.

I also added regression tests for these cases so the same bugs are less likely to come back.

## 7. Real model results

I ran four local Ollama models on the full benchmark. These are real saved results, not guesses.

### Main results (primary metrics)

| Model | Strategy | Accuracy | 95% CI | Macro-F1 | Bal. Acc | Lex Acc | Presup Acc | Plausible Non-Ent Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3:latest | zero_shot | 0.6778 | 0.6111-0.7444 | 0.6459 | 0.6778 | 0.7833 | 0.7167 | 0.5333 |
| llama3:latest | few_shot | 0.7722 | 0.7056-0.8333 | 0.7662 | 0.7722 | 0.8500 | 0.7667 | 0.7000 |
| llama3:latest | cot | 0.5944 | 0.5278-0.6611 | 0.6006 | 0.5944 | 0.7333 | 0.6500 | 0.4000 |
| mistral:7b | zero_shot | 0.6889 | 0.6222-0.7500 | 0.6651 | 0.6889 | 0.8333 | 0.7167 | 0.5167 |
| mistral:7b | few_shot | 0.7389 | 0.6722-0.8001 | 0.7215 | 0.7389 | 0.8667 | 0.7333 | 0.6167 |
| mistral:7b | cot | 0.7611 | 0.6944-0.8222 | 0.7552 | 0.7611 | 0.9000 | 0.7000 | 0.6833 |
| mixtral:8x7b | zero_shot | 0.7778 | 0.7165-0.8333 | 0.8006 | 0.7778 | 0.8833 | 0.7667 | 0.6833 |
| mixtral:8x7b | few_shot | 0.8556 | 0.8056-0.9056 | 0.8572 | 0.8556 | 0.9000 | 0.8333 | 0.8333 |
| mixtral:8x7b | cot | 0.7667 | 0.7000-0.8222 | 0.7631 | 0.7667 | 0.9000 | 0.7167 | 0.6833 |
| qwen2.5:7b | zero_shot | 0.8667 | 0.8111-0.9111 | 0.8663 | 0.8667 | 0.8333 | 0.8667 | 0.9000 |
| qwen2.5:7b | few_shot | 0.9056 | 0.8610-0.9444 | 0.9054 | 0.9056 | 0.8833 | 0.9000 | 0.9333 |
| qwen2.5:7b | cot | 0.9000 | 0.8556-0.9389 | 0.8998 | 0.9000 | 0.8667 | 0.9000 | 0.9333 |
| openai/gpt-oss-20b (Groq) | zero_shot | 0.4389 | 0.3611-0.5056 | 0.5986 | 0.4389 | 0.6167 | 0.3833 | 0.3167 |
| openai/gpt-oss-20b (Groq) | few_shot | 0.3667 | 0.2944-0.4389 | 0.4764 | 0.3667 | 0.4833 | 0.4000 | 0.2167 |
| openai/gpt-oss-20b (Groq) | cot | 0.9056 | 0.8556-0.9444 | 0.9184 | 0.9056 | 0.9500 | 0.9000 | 0.8667 |

### What I noticed

- **Qwen2.5:7b was the strongest overall** (about 87-91%). Mixtral:8x7b was next (about 77-86%), then Mistral:7b (about 69-76%), then Llama3:latest (about 59-77%).
- **Plausible non-entailment was the hardest category** for most models, especially Llama3:latest (40-70%) and Mistral:7b zero-shot (51.67%). This suggests world-knowledge over-acceptance is a real problem.
- **Few-shot helped most models.** Llama3 improved a lot with few-shot; Mixtral also jumped in few-shot. Mistral got its best score in CoT, but CoT hurt Llama3.
- **Invalid answers were rare** overall for most models, but Mixtral zero-shot had 6.67% invalid outputs and Llama3 CoT had 4.44%. `openai/gpt-oss-20b` (Groq, reasoning model) was very strong in CoT (90.56%) but had very high invalid rates in zero-shot/few-shot (53.33%/47.78%) because it often put answers in reasoning content instead of a clean final label.

### Extra diagnostic results

| Model | Strategy | FP rate | FN rate | Invalid rate | Acc. excl. contested |
|---|---|---:|---:|---:|---:|
| llama3:latest | zero_shot | 0.6222 | 0.0222 | 0.0000 | 0.6705 |
| llama3:latest | few_shot | 0.3889 | 0.0667 | 0.0000 | 0.7670 |
| llama3:latest | cot | 0.5333 | 0.1889 | 0.0444 | 0.5966 |
| mistral:7b | zero_shot | 0.5778 | 0.0444 | 0.0000 | 0.6875 |
| mistral:7b | few_shot | 0.5111 | 0.0111 | 0.0000 | 0.7386 |
| mistral:7b | cot | 0.4222 | 0.0444 | 0.0056 | 0.7557 |
| mixtral:8x7b | zero_shot | 0.2556 | 0.0556 | 0.0667 | 0.7727 |
| mixtral:8x7b | few_shot | 0.2000 | 0.0778 | 0.0056 | 0.8523 |
| mixtral:8x7b | cot | 0.3556 | 0.1111 | 0.0000 | 0.7614 |
| qwen2.5:7b | zero_shot | 0.1889 | 0.0778 | 0.0000 | 0.8636 |
| qwen2.5:7b | few_shot | 0.1333 | 0.0556 | 0.0000 | 0.9034 |
| qwen2.5:7b | cot | 0.1444 | 0.0556 | 0.0000 | 0.8977 |
| openai/gpt-oss-20b (Groq) | zero_shot | 0.0111 | 0.0444 | 0.5333 | 0.4432 |
| openai/gpt-oss-20b (Groq) | few_shot | 0.2667 | 0.0444 | 0.4778 | 0.3580 |
| openai/gpt-oss-20b (Groq) | cot | 0.0889 | 0.0444 | 0.0278 | 0.9148 |

The false positive rate shows over-entailment. Llama3 and Mistral had high false positive rates, which fits the idea that they accepted too many plausible statements.

### Perturbation robustness (30 items)

| Model | Strategy | Consistency | Joint Acc. | Orig. Acc. | Perturbed Acc. |
|---|---|---:|---:|---:|---:|
| llama3:latest | zero_shot | 0.9667 | 0.5667 | 0.6000 | 0.5667 |
| llama3:latest | few_shot | 0.9000 | 0.7333 | 0.8000 | 0.7667 |
| llama3:latest | cot | 0.8000 | 0.4333 | 0.5333 | 0.5333 |
| mistral:7b | zero_shot | 0.9667 | 0.6667 | 0.6667 | 0.7000 |
| mistral:7b | few_shot | 0.9667 | 0.6333 | 0.6333 | 0.6667 |
| mistral:7b | cot | 0.9667 | 0.8000 | 0.8333 | 0.8000 |
| mixtral:8x7b | zero_shot | 0.9333 | 0.8000 | 0.8667 | 0.8000 |
| mixtral:8x7b | few_shot | 1.0000 | 0.8667 | 0.8667 | 0.8667 |
| mixtral:8x7b | cot | 0.7667 | 0.6000 | 0.8000 | 0.6333 |
| qwen2.5:7b | zero_shot | 0.8667 | 0.8000 | 0.8333 | 0.9000 |
| qwen2.5:7b | few_shot | 0.8333 | 0.8000 | 0.9000 | 0.8667 |
| qwen2.5:7b | cot | 0.8667 | 0.7667 | 0.8333 | 0.8333 |
| openai/gpt-oss-20b (Groq) | zero_shot | 0.6333 | 0.0000 | 0.3333 | 0.0000 |
| openai/gpt-oss-20b (Groq) | few_shot | 0.3667 | 0.0000 | 0.5000 | 0.0000 |
| openai/gpt-oss-20b (Groq) | cot | 0.9667 | 0.9333 | 0.9333 | 0.9667 |

These are small-sample robustness checks (only 30 pairs), so I would not make strong claims from them alone.

### Statistical significance (McNemar, paired exact test)

I used paired exact McNemar tests because each model answers the same items. Some differences that look small are still meaningful, and some are not.

- **Llama3:latest**: few-shot was significantly better than zero-shot (p=0.0023). Zero-shot was significantly better than CoT (p=0.0237).
- **Mistral:7b**: CoT (p=0.0023) and few-shot (p=0.0225) were significantly better than zero-shot.
- **Mixtral:8x7b**: few-shot was significantly better than zero-shot (p=0.0094); CoT vs zero-shot was not significant (p=0.8555).
- **Qwen2.5:7b**: few-shot was significantly better than zero-shot (p=0.0391); CoT vs zero-shot was not significant (p=0.1460).
- **Model comparisons**: Qwen2.5:7b was significantly better than Llama3 and Mistral in all tested strategies, and better than Mixtral in zero-shot and CoT; Mixtral vs Qwen few-shot was close but not significant at 0.05 (p=0.0931). Mistral vs Mixtral CoT was not significant (p=1.0).

Full results are in `results/significance_tests.csv`.

### Error analysis (exploratory)

The error labels are heuristic (rule-based), so I treat them as exploratory, not final truth. Still, the pattern is clear:

- **World-knowledge over-entailment** was the most common error for weaker models, especially Llama3 and Mistral. This matches the high false positive rates.
- **Negation/presupposition/factivity errors** appeared across models, showing that logical scope and trigger handling are still hard.
- **Format errors** were rare, but Mixtral zero-shot had 12 format errors in the heuristic attribution and 6.67% invalid rate overall; this should be checked manually.

Full table: `results/error_attribution.csv`. Figures: `results/category_accuracy_comparison.png` and `results/fp_fn_rates.png`.

## 8. How to fix the limitations

You asked for a clear answer to the earlier limitations. Here is how I would fix them.

| Limitation | Why it is a problem | How to fix it |
|---|---|---|
| Some proposal models still not run | The proposal listed more models; current results are still incomplete. | Run remaining planned models (`llama3.3:70b` or exact 70B tag, GPT-4o, a current GPT-3.5 replacement such as `gpt-4o-mini`, and optional HF exact-logit check) when resources/API access are available. Save raw predictions and manifests for every run. I attempted `llama3.3:70b` locally but got an Ollama 500 Internal Server Error (likely memory/resource limit). Try a smaller quantized 70B tag, a machine with more RAM/VRAM, or a cloud GPU. |
| No independent human inter-annotator agreement | One person or one team can miss label problems. | Ask at least 2 independent annotators to label items using the same written rules. Start with all 4 contested items and a sample of tricky items; ideally label all 180. Compute agreement (for example Cohen's kappa for two people, Fleiss' kappa for more). Resolve disagreements by discussion and document final decisions. |
| Log-probability scores are not fully comparable across backends | HF gives exact token logits, while Ollama/OpenAI often give top-logprob approximations. | Do not compare raw logprob margins across different backends as if they were identical. For fair comparison, use the same backend where possible, or report exact-logit HF results separately. Also report plain Yes/No accuracy for all models. |
| Plagiarism risk in writing/code | Copying text or code can cause serious academic problems. | Write text in your own words, cite sources for ideas, keep notes of where ideas came from, and do not copy benchmark text from papers. For code, write your own implementation, keep commit history, and run a plagiarism/originality check before submission. |
| Small dataset | 180 items is good for a class project, but not enough for big claims. | Add more items, more categories, and more annotators. Test on other benchmarks too. Be careful with claims in the final report. |
| Theory-dependent labels | Some cases depend on linguistics theory, not just simple right/wrong. | Keep contested labels visible, report both all-items and no-contested results, and explain the theory issue in the report. |

## 9. Reproducibility

I tried to make the project easy to run again. The repo now includes:

- `pyproject.toml` for clean installs with extras
- `requirements.txt` for simple install
- `scripts/run_all.sh` for Linux/macOS
- `scripts/run_all.ps1` for Windows
- JSONL cache files so runs can continue after a crash
- run manifest files with hashes, command, time, and git commit
- tests for data, metrics, parsing, and bug fixes

### Basic commands

```bash
pip install -e .[dev]
pytest -v
python -m src.evaluate --backend ollama --model llama3:8b
python -m src.analyze
```

## 10. Limits of this report

This report is honest about what is done and what is not done. I ran real tests for four local Ollama models (`llama3:latest`, `mistral:7b`, `mixtral:8x7b`, `qwen2.5:7b`) and one Groq API model (`openai/gpt-oss-20b`), and the code/data checks pass. But I did not run all models from the original proposal (Llama 70B run, OpenAI GPT models, and optional HF exact-logit check are still missing). OpenRouter free-model run failed due to authentication with the current key. I also did not collect independent human agreement scores. The dataset is still small, and some labels are theory-dependent. These parts should be improved before a final submission.

## 11. Final thoughts

The main value of this project is that it gives a clear and simple way to test language models on meaning, not just fluency. The benchmark is small, but it is balanced and easy to inspect. The code now has better tests, better caching, run manifests, and clearer metric reporting. The real runs show something useful: `qwen2.5:7b` performed best overall, Mixtral was strong in few-shot, and Llama3/Mistral made many over-entailment mistakes, especially on plausible non-entailment. With more models and human label checks, this setup can give an even stronger and more trustworthy comparison.

## 12. Submission checklist

- [x] Clean 1-page proposal PDF/MD.
- [x] Detailed plain-language report PDF/MD.
- [x] Balanced benchmark (180) + perturbations (30) + data card.
- [x] Reproducible code with tests (`13 passed`).
- [x] Run manifests, caches, raw predictions, summaries, plots.
- [x] Real runs for 5 models (llama3, mistral, mixtral, qwen2.5, gpt-oss-20b via Groq).
- [x] Significance tests and error attribution.
- [x] Annotation protocol/template for human agreement.
- [x] RTX 4060 GPU detected; CUDA PyTorch install attempted for optional HF exact-logit run.
- [ ] Independent human inter-annotator agreement (kappa) completed by real annotators.
- [ ] Remaining proposal models run (70B/API/HF exact-logit) or explicitly justified.
- [ ] Final plagiarism/originality check before submission.

## Appendix A: Human annotation package

Prepared files for independent agreement study:
- `annotation/annotation_template.csv` (blind, 180 items)
- `annotation/ANNOTATION_PROTOCOL.md`
- `annotation/disagreement_resolution_template.csv`
- `scripts/compute_agreement.py`

This step needs real human annotators and cannot be honestly automated.

## References

- Basmov, V., Goldberg, Y., & Tsarfaty, R. (2024). Simple Linguistic Inferences of Large Language Models (LLMs): Blind Spots and Blinds. arXiv:2305.14785.
- Fox, D., & Katzir, R. (2024). Large Language Models and theoretical linguistics. Theoretical Linguistics, 50(1-2), 71-76.
- Futrell, R., & Mahowald, K. (2026). How linguistics learned to stop worrying and love the language models. Behavioral and Brain Sciences, 49, e198.
- Piantadosi, S. T. (2023). Modern language models refute Chomsky's approach to language. Lingbuzz preprint 007180.

# Review vs. project proposal

Coverage: 180 pairs / 3 categories / 30 perturbations / zero-, few-shot, CoT / Accuracy + Macro-F1 / error attribution all present. Gaps and fixes:

## Fixed
| Area | Problem | Fix |
|---|---|---|
| Prompt | System prompt demanded "no world assumptions", yet gold marks negated presuppositions (e.g. "John did not stop smoking" -> "used to smoke") as Yes; a strictly logical model is penalised | One shared task definition covering presuppositions, excluding merely plausible inferences |
| Prompt | CoT alone got hints (negation scope, presupposition triggers) -> strategy confounded | Same definition for all strategies |
| Few-shot | Demo overlapped test subtypes | New demos, test asserts they are not in the benchmark |
| CoT parsing | Fallback took the last yes/no inside the reasoning; 180 tokens truncated answers | Strict `Final Answer:`, else `Invalid`; 400 tokens |
| HF backend | No chat template for instruct models; only " Yes"/" No" token; wasteful `generate` | Chat template, summed token variants, logits only for direct prompts |
| OpenAI | Variant tokens overwrote each other; one-sided top-k dropped probabilities; `top_logprobs=None` | logsumexp over variants, floor for missing side, conditional kwargs, seed |
| Metrics | No Macro-F1 *per category* (proposal), Invalid handled inconsistently, no baselines/CIs/significance | Added per-category F1 + balanced accuracy, `invalid_rate`, majority baselines, bootstrap CI, exact McNemar |
| Error attribution | Non-factive/counterfactive false positives fell into "other"; under-entailment mislabelled | `factivity_overgeneralization`, `negation_scope_failure`, `overly_strict_on_control` |
| Perturbations | old `INF-146` paraphrase belonged to INF-158 and was ungrammatical; "plastic pistol" is not privative; "cardiac arrest" != "heart attack" | Repaired; test checks base_id/paraphrase overlap |
| Data | `a alleged`; INF-032 needed an unstated assumption (front door); INF-034 was world knowledge, not lexical; "tall" mislabelled intersective | Corrected |
| Infra | Plain `pytest` failed (import path); mixed import styles; no resume | `pytest.ini`, `python -m src.*`, JSONL cache, `--limit` |
| Analysis | Only zero-shot vs CoT plotted; robustness/FP-FN not aggregated | All strategies, FP/FN plot, robustness summary |

## Open decisions - now resolved
1. **Label skew**: rebalanced inside the 180-pair cap to 30 Yes / 30 No per category (chance = 50%). 48 near-duplicate items (mostly hypernym Yes, factive Yes, script No) were replaced by 18 lexical, 15 presupposition and 15 plausible-category items; the plausible category now has minimal-pair entailed controls. Perturbation base_ids were remapped; IDs were renumbered (old->new mapping is not needed for results).
2. **Log-probabilities**: Ollama supports `logprobs`/`top_logprobs`; the backend now requests them for direct prompts (graceful fallback on old servers). All three backends give P(Yes)/P(No).
3. **Contested labels**: marked in the CSV (`contested`, `contested_note`): "every -> some" (existential import), "king of France" (Russell vs Strawson), "managed -> required effort" (implicature), counterfactual "If he had studied ... -> He studied". `acc_excl_contested` is reported; discuss in the report.
4. **Proposal text**: rewritten (see `Project_Proposal_corrected`). References verified: Basmov et al. = "Simple Linguistic Inferences of Large Language Models (LLMs): Blind Spots and Blinds", arXiv:2305.14785 (v1 2023, v2 Apr 2024); Fox & Katzir = "Large Language Models and theoretical linguistics", Theoretical Linguistics 50(1-2):71-76; Futrell & Mahowald (2026), BBS 49, e198.

## Still true
- Llama 3 70B / Mixtral are heavy; run them via Ollama (quantised).
- Nothing was run against real models here (no Ollama/GPU/API key); tests and a mock-backend run pass.

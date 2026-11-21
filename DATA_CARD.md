# Data Card: Simple Linguistic Inferences Benchmark

## 1. Dataset summary

- **Name:** Simple Linguistic Inferences Benchmark
- **Version:** 1.0 (project submission draft)
- **Author:** Muhammad Baqir (Matriculation No. 2184791)
- **Course:** GdI-Proj-B/M: LLMs and Linguistic Theory (WS 2026/27), University of Bamberg
- **Supervisor:** Dr. Luke Burke
- **Files:**
  - `data/inference_benchmark.csv` — 180 premise–hypothesis items
  - `data/perturbation_benchmark.csv` — 30 truth-preserving lexical paraphrases
- **Task:** Binary inference judgment (Yes/No) for whether a hypothesis follows from a premise.

## 2. Motivation and intended use

The benchmark evaluates whether language models distinguish strict semantic/pragmatic inference from plausible but unsupported conclusions. It is designed for controlled course-project evaluation, error analysis, and prompt-strategy comparison.

**Intended uses**
- Evaluate LLMs on lexical entailment, presupposition projection, and plausible non-entailment.
- Compare prompting strategies (zero-shot, few-shot, chain-of-thought).
- Run robustness checks with lexical paraphrases.
- Analyze false positives (over-entailment) and false negatives (under-entailment).

**Out-of-scope uses**
- High-stakes decisions about people, law, medicine, or safety.
- Claims about general linguistic competence from this dataset alone.
- Training/fine-tuning without additional validation and license review.

## 3. Composition

### Main benchmark (`inference_benchmark.csv`)

| Column | Description |
|---|---|
| `id` | Unique item identifier (`INF-001` ... `INF-180`). |
| `category` | One of `lexical_entailment`, `presupposition`, `plausible_non_entailment`. |
| `subtype` | Fine-grained linguistic/control subtype. |
| `premise` | Premise sentence. |
| `hypothesis` | Hypothesis sentence. |
| `gold_label` | `Yes` if hypothesis follows; `No` otherwise. |
| `contested` | `True` if the gold label is theoretically debatable. |
| `contested_note` | Short reason for contestation; empty for uncontested items. |

**Balance**
- 180 items total.
- 60 items per category.
- 30 `Yes` and 30 `No` per category (chance = 50%).
- 4 contested items.

### Perturbations (`perturbation_benchmark.csv`)

| Column | Description |
|---|---|
| `base_id` | ID of the original benchmark item. |
| `premise` | Paraphrased premise. |
| `hypothesis` | Paraphrased hypothesis. |
| `gold_label` | Preserved gold label. |

**Properties**
- 30 items total.
- Each `base_id` maps to an existing benchmark item.
- Paraphrases are intended to preserve the inference relation.

## 4. Labeling and annotation protocol

### Label definition used by the project

A hypothesis receives `Yes` when it must hold given the premise in its ordinary reading, including relevant lexical meaning relations and presuppositions that are normally taken for granted by the premise wording. A hypothesis receives `No` when it is merely plausible, typical, suggested by world knowledge, or left open by the premise.

### Construction process

1. Drafted items by category and subtype, inspired by the project proposal and prior linguistic-inference evaluation work.
2. Balanced labels within each category to avoid majority-class shortcuts.
3. Added plausible non-entailment controls and strict entailed controls in similar scenarios.
4. Added contested flags for theoretically debated cases.
5. Added lexical perturbations with preserved truth conditions.
6. Validated with automated integrity tests (`tests/test_pipeline.py`).

### Contested-label protocol

Items are marked `contested=True` when competent linguistic theories or common analyses may disagree. Primary reporting should include:
- **Main metric:** accuracy/F1 over all items.
- **Sensitivity metric:** `acc_excl_contested` excluding contested items.
- **Qualitative discussion:** explicit discussion of every contested item.

## 5. Second-pass linguistic review (documented)

This review was performed as a systematic second pass over contested items and category design. It is **not** independent human inter-annotator agreement. No external annotator study was run in this repository; collecting independent annotations and computing agreement (e.g., Cohen's kappa / Fleiss' kappa) is a required limitation to address before external publication.

### Contested items

| ID | Category | Gold | Review judgment | Reason | Recommendation |
|---|---|---|---:|---|---|
| INF-017 | lexical_entailment | Yes | Debatable | "Every student passed the exam" can be analyzed as entailing "Some students passed the exam" under existential import; modern logic often treats universal statements as compatible with empty domains. | Keep contested flag; report both with/without item; avoid using as a core lexical example. |
| INF-066 | presupposition | Yes | Debatable | Definite description "The king of France..." raises classic Russell vs. Strawson disagreement. | Keep contested; discuss as theory-dependent presupposition/uniqueness case. |
| INF-076 | presupposition | Yes | Debatable | "Managed to finish" strongly implies difficulty/effort, but some analyses treat effort as conventional implicature rather than entailment/presupposition. | Keep contested; if revised, consider moving to implicature-focused control or relabeling with stricter criterion. |
| INF-118 | presupposition | No | Debatable | Counterfactuals commonly implicate antecedent falsity, but implicature can be cancellable; calling it a strict non-entailment is plausible under the project definition. | Keep contested; report sensitivity and discuss implicature vs. entailment boundary. |

### Additional review notes

- **Plausible non-entailment items** should remain world-plausible but not logically necessary. Any item where the hypothesis becomes necessary under ordinary lexical meaning should be moved or relabeled.
- **Presupposition projection items** should be checked for trigger strength and cancellation behavior. Negation and question controls are useful but need clear annotation rationale.
- **Lexical items** should avoid world-knowledge shortcuts and ambiguous polysemy.
- **Perturbations** should preserve truth conditions, not merely topic similarity.

## 6. Automated validation

The test suite checks:
- Prompt format and CoT parsing rules.
- Few-shot examples are not present in the benchmark.
- Metric behavior, invalid-label handling, and McNemar/baseline calculations.
- Benchmark size, balance, uniqueness, contested-note presence, and no missing values outside `contested_note` where expected.
- Perturbation base IDs, label preservation, and lexical overlap.

Run:

```bash
pytest -v
```

## 7. Limitations

- The dataset is small (180 main + 30 perturbation items) and not representative of all linguistic phenomena.
- Labels reflect one project annotation protocol; some items are theory-dependent.
- No independent human inter-annotator agreement is reported.
- Subtype names are useful for organization but do not replace expert linguistic annotation.
- Automated error attribution in code is heuristic and should be treated as exploratory.

## 8. Ethical and responsible use considerations

- **Bias and cultural assumptions:** plausible-event scripts may reflect common-sense assumptions that vary across cultures.
- **Overclaim risk:** good performance on this benchmark does not prove robust semantic understanding.
- **Misuse risk:** do not use this benchmark as a certification of model reliability.
- **Transparency:** report contested-item handling, prompt templates, model versions, and run manifests.

## 9. Reproducibility checklist

- Keep `data/*.csv` unchanged for a reported run; record SHA-256 hashes.
- Record prompt hash, model ID, backend, decoding settings, date/time, and git commit (manifest written by `src.evaluate`).
- Report primary and exploratory metrics separately.
- Release raw predictions and analysis outputs.

## 10. Licensing and attribution

This data card describes project-internal benchmark files. Before public release, confirm licensing/attribution for any adapted ideas and add an explicit dataset license. Cite Basmov et al. (2024) as conceptual inspiration for simple linguistic inference evaluation, while noting that the present dataset, labels, and code are project-specific artifacts.

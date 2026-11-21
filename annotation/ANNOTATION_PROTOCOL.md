# Annotation Protocol (Human Agreement)

## Goal
Check whether the gold labels in `data/inference_benchmark.csv` are clear and reliable. This is needed because the project currently has no independent human inter-annotator agreement score.

## Who should annotate
- At least **2 independent annotators** (not the same person who created the labels).
- Annotators should have basic linguistics or NLP background if possible.
- They must not see `gold_label`, `contested`, or `contested_note` during first-pass labeling.

## Files
- `annotation_template.csv`: blind template (no gold labels).
- After annotation, save each annotator file as `annotation/annotator_A.csv`, `annotation/annotator_B.csv`, etc.

## Labeling rules
For each item, decide whether the **Hypothesis** follows from the **Premise**.

- **Yes**: The hypothesis must be true if the premise is understood in its normal meaning. This includes word meaning relations and what the premise wording takes for granted.
- **No**: The hypothesis is only likely, typical, possible, or based on world knowledge, or the premise leaves it open.

If unsure, still choose Yes/No and explain in `notes`; use `confidence_1to5` to show uncertainty.

## Process
1. Give annotators this protocol and the blind template.
2. First pass: annotators label independently (no discussion).
3. Compute agreement:
   - 2 annotators: Cohen's kappa + percent agreement.
   - 3+ annotators: Fleiss' kappa + percent agreement.
4. Review disagreements, especially contested items and low-confidence items.
5. Document decisions in a resolution table: keep label, change label, or mark contested.

## Minimum acceptable for submission
- All 4 contested items + at least 50 additional items annotated by 2 people.
- Report kappa, percent agreement, and disagreement examples.

## Recommended for strong grade
- All 180 items annotated by 2 independent annotators.
- Disagreement resolution log included in appendix.

## Simple kappa computation
You can use sklearn:

```python
import pandas as pd
from sklearn.metrics import cohen_kappa_score
a = pd.read_csv('annotation/annotator_A.csv')
b = pd.read_csv('annotation/annotator_B.csv')
m = a.merge(b, on='id', suffixes=('_a','_b'))
print('percent', (m.annotator_label_a == m.annotator_label_b).mean())
print('kappa', cohen_kappa_score(m.annotator_label_a, m.annotator_label_b))
```

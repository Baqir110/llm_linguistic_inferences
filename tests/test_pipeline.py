"""Unit + data-integrity tests: prompts, parsing, metrics, benchmark consistency."""

import pandas as pd

from src.backends import parse_binary_label, yes_no_from_top_logprobs
from src.metrics import (classify_error, compute_perturbation_consistency,
                         compute_summary_metrics, mcnemar_exact, majority_baselines)
from src.prompts import FEW_SHOT_EXAMPLES, build_prompt

BENCH = pd.read_csv("data/inference_benchmark.csv")
PERT = pd.read_csv("data/perturbation_benchmark.csv")


def test_parse_binary_label():
    assert parse_binary_label("Yes.") == "Yes"
    assert parse_binary_label("No, it does not follow.") == "No"
    assert parse_binary_label("Unclear", is_cot=False) == "Invalid"
    assert parse_binary_label("Step 1: ...\nFinal Answer: No", is_cot=True) == "No"
    assert parse_binary_label("**Final Answer: Yes**", is_cot=True) == "Yes"


def test_cot_does_not_use_yes_no_from_reasoning():
    # Truncated / unformatted CoT must be Invalid, not whatever yes/no appears in the reasoning.
    assert parse_binary_label("No negation is present, so the answer is yes", is_cot=True) == "Invalid"


def test_prompt_builders_share_task_definition():
    zs, fs, cot = (build_prompt("A dog barked.", "An animal barked.", s) for s in ("zero_shot", "few_shot", "cot"))
    assert fs.count("Premise:") == len(FEW_SHOT_EXAMPLES) + 1
    assert "Final Answer: Yes" in cot
    first = zs.split("\n")[0]
    assert first in fs and first in cot


def test_few_shot_examples_not_in_benchmark():
    pairs = set(zip(BENCH.premise, BENCH.hypothesis))
    assert not any((p, h) in pairs for p, h, _ in FEW_SHOT_EXAMPLES)


def test_error_attribution_and_metrics():
    df = pd.DataFrame([
        dict(model="m", strategy="zero_shot", id="a", category="plausible_non_entailment",
             subtype="medical_plausibility", gold_label="No", predicted_label="Yes"),
        dict(model="m", strategy="zero_shot", id="b", category="presupposition",
             subtype="factive_verb", gold_label="Yes", predicted_label="No"),
        dict(model="m", strategy="zero_shot", id="c", category="lexical_entailment",
             subtype="hypernym", gold_label="Yes", predicted_label="Yes"),
        dict(model="m", strategy="zero_shot", id="d", category="presupposition",
             subtype="non_factive_verb", gold_label="No", predicted_label="Yes"),
        dict(model="m", strategy="zero_shot", id="e", category="presupposition",
             subtype="negated_factive", gold_label="Yes", predicted_label="Invalid"),
    ])
    df["error_type"] = df.apply(classify_error, axis=1)
    assert list(df.error_type) == ["world_knowledge_overentailment", "presupposition_projection_failure",
                                   "correct", "factivity_overgeneralization", "format_error"]
    s = compute_summary_metrics(df, n_boot=50)
    assert s.loc[0, "false_positive_rate"] == 1.0
    assert abs(s.loc[0, "false_negative_rate"] - 1 / 3) < 1e-3  # Invalid is not a "No"
    assert abs(s.loc[0, "invalid_rate"] - 0.2) < 1e-6
    assert abs(s.loc[0, "accuracy"] - 0.2) < 1e-6


def test_ollama_logprob_parsing():
    import math
    entries = [{"token": "Yes", "logprob": math.log(0.6),
                "top_logprobs": [{"token": "Yes", "logprob": math.log(0.5)}, {"token": " yes", "logprob": math.log(0.1)},
                                 {"token": "No", "logprob": math.log(0.3)}]}]
    py, pn, margin = yes_no_from_top_logprobs(entries)
    assert abs(py - 0.6 / 0.9) < 1e-6 and abs(pn - 0.3 / 0.9) < 1e-6 and margin > 0
    assert yes_no_from_top_logprobs(None) == (None, None, None)
    assert yes_no_from_top_logprobs([{"token": "Maybe", "logprob": -1, "top_logprobs": []}])[0] is None


def test_mcnemar_and_baselines():
    import numpy as np
    a = np.array([True] * 10 + [False] * 2)
    b = np.array([False] * 10 + [True] * 2)
    assert mcnemar_exact(a, b)[:2] == (10, 2)
    assert mcnemar_exact(a, a)[2] == 1.0
    mb = majority_baselines(BENCH)
    assert mb.shape[0] == 4 and (mb.always_yes_acc == 0.5).all()  # balanced: constant guesser = 50%


def test_perturbation_consistency():
    base = pd.DataFrame([dict(model="m", strategy="s", id="x", gold_label="Yes", predicted_label="Yes")])
    pert = pd.DataFrame([dict(model="m", strategy="s", base_id="x", predicted_label="No")])
    r = compute_perturbation_consistency(base, pert)
    assert r.loc[0, "consistency_rate"] == 0.0 and r.loc[0, "acc_perturbed"] == 0.0


# ---- benchmark integrity -------------------------------------------------
def test_benchmark_structure():
    assert len(BENCH) == 180 and BENCH.id.is_unique
    assert set(BENCH.gold_label) == {"Yes", "No"}
    assert (BENCH.groupby("category").size() == 60).all()
    assert (BENCH.groupby(["category", "gold_label"]).size() == 30).all()   # balanced 30/30
    assert BENCH.contested.sum() == 4 and BENCH[BENCH.contested].contested_note.notna().all()
    assert not BENCH.duplicated(["premise", "hypothesis"]).any()
    assert not BENCH.drop(columns='contested_note').isna().any().any()


def test_perturbations_match_base_items():
    m = PERT.merge(BENCH, left_on="base_id", right_on="id", suffixes=("_p", ""))
    assert len(m) == len(PERT)                      # every base_id exists
    assert (m.gold_label_p == m.gold_label).all()   # truth-preserving
    assert (m.premise_p != m.premise).all() or (m.hypothesis_p != m.hypothesis).all()
    # paraphrase should share most content words with its base item (catches wrong base_id)
    def toks(s): return {w.strip(".,").lower() for w in s.split() if len(w) > 3}
    for r in m.itertuples():
        base = toks(r.premise) | toks(r.hypothesis)
        pert = toks(r.premise_p) | toks(r.hypothesis_p)
        assert len(base & pert) >= 1, r.base_id


# ---- regression tests -----------------------------------------------------
def test_cache_round_trip_preserves_types(tmp_path):
    import json
    import numpy as np

    from src.evaluate import _json_safe, run_split

    class FakeBackend:
        def predict(self, prompt, system_prompt, is_cot=False):
            from src.backends import ModelPrediction
            return ModelPrediction(raw_text="Yes", predicted_label="Yes", prob_yes=0.9, prob_no=0.1, logprob_margin=2.1972)

    df = pd.DataFrame([{
        "id": "X1", "category": "lexical_entailment", "subtype": "hypernym",
        "premise": "A dog ran.", "hypothesis": "An animal ran.",
        "gold_label": "Yes", "contested": np.bool_(False), "contested_note": "",
    }])
    cache = tmp_path / "cache.jsonl"
    backend = FakeBackend()
    out1 = run_split(df, backend, "fake", ["zero_shot"], "id", cache)
    # simulate old corrupted cache value on disk should not break metrics; new cache should be clean
    raw = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1
    rec = json.loads(raw[0])
    assert rec["contested"] is False
    assert isinstance(rec["prob_yes"], float)

    out2 = run_split(df, backend, "fake", ["zero_shot"], "id", cache)
    assert len(out2) == 1
    assert out2.loc[0, "predicted_label"] == "Yes"
    assert bool(out2.loc[0, "contested"]) is False

    # stale cache guard: changed premise should force recompute
    df2 = df.copy()
    df2.loc[0, "premise"] = "A cat ran."
    out3 = run_split(df2, backend, "fake", ["zero_shot"], "id", cache)
    assert out3.loc[0, "premise"] == "A cat ran."
    assert len(cache.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_hf_empty_token_ids_raises_clear_error():
    import pytest

    from src.backends import HuggingFaceBackend, ModelPrediction

    class DummyTok:
        chat_template = None
        eos_token_id = 0
        def encode(self, v, add_special_tokens=False):
            return [1, 2, 3]  # never single token
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    class DummyModel:
        device = "cpu"

    b = HuggingFaceBackend.__new__(HuggingFaceBackend)
    b.tokenizer = DummyTok()
    b.model = DummyModel()
    b.yes_ids = []
    b.no_ids = []
    b.torch = None
    b._encode = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not encode"))
    # direct prediction path should fail before encoding only after logits? we need to hit the guard
    # Build a minimal fake torch/model path by monkeypatching predict internals via subclassing.
    class FakeHF(HuggingFaceBackend):
        def _encode(self, prompt, system_prompt):
            import torch
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

        def predict(self, prompt, system_prompt, is_cot=False):
            import torch
            class Out:
                logits = torch.zeros((1, 1, 4))

            self.model = type("M", (), {"device": "cpu", "__call__": lambda self, **kw: Out()})()
            self.torch = torch
            return HuggingFaceBackend.predict(self, prompt, system_prompt, is_cot=is_cot)

    fake = FakeHF.__new__(FakeHF)
    fake.tokenizer = DummyTok()
    fake.yes_ids = []
    fake.no_ids = []
    with pytest.raises(ValueError, match="single-token Yes/No"):
        fake.predict("p", "s", is_cot=False)


def test_openai_whitespace_first_logprob_token():
    from types import SimpleNamespace

    from src.backends import OpenAIBackend

    # Build fake OpenAI response objects with attribute access.
    top = [SimpleNamespace(token="Yes", logprob=-0.1), SimpleNamespace(token="No", logprob=-2.3)]
    content = [SimpleNamespace(token="\n", top_logprobs=[]), SimpleNamespace(token="Yes", top_logprobs=top)]
    choice = SimpleNamespace(message=SimpleNamespace(content="Yes"), logprobs=SimpleNamespace(content=content))
    resp = SimpleNamespace(choices=[choice])

    b = OpenAIBackend.__new__(OpenAIBackend)
    b.model_name = "fake"
    b.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    pred = b.predict("p", "s", is_cot=False)
    assert pred.predicted_label == "Yes"
    assert pred.prob_yes is not None and pred.prob_no is not None
    assert pred.prob_yes > pred.prob_no

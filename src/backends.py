"""Model inference backends for Ollama, Hugging Face Transformers, and OpenAI."""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

COT_MAX_TOKENS = 400  # 180 truncated verbose models before "Final Answer"
SEED = 1


@dataclass
class ModelPrediction:
    raw_text: str
    predicted_label: str
    prob_yes: Optional[float] = None
    prob_no: Optional[float] = None
    logprob_margin: Optional[float] = None  # log P(Yes) - log P(No)


def parse_binary_label(text: str, is_cot: bool = False) -> str:
    """Extract a Yes/No label; return 'Invalid' if none can be found.

    CoT: only an explicit 'Final Answer: Yes|No' counts (the last one). Falling back to
    any yes/no inside the reasoning would mis-score outputs such as "No negation here ...".
    Direct: the first Yes/No token, which must be the answer itself.
    """
    cleaned = text.strip()
    if is_cot:
        matches = re.findall(r"Final Answer\s*:\s*\**\s*(Yes|No)\b", cleaned, flags=re.IGNORECASE)
        return matches[-1].capitalize() if matches else "Invalid"
    match = re.search(r"\b(Yes|No)\b", cleaned, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else "Invalid"


def _logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _softmax_yes_no(logp_yes: float, logp_no: float) -> tuple[float, float, float]:
    max_lp = max(logp_yes, logp_no)
    exp_y = math.exp(logp_yes - max_lp)
    exp_n = math.exp(logp_no - max_lp)
    total = exp_y + exp_n
    return exp_y / total, exp_n / total, logp_yes - logp_no


def yes_no_from_top_logprobs(entries, floor: float = -20.0):
    """First *answer* token's top-k alternatives -> (P(Yes), P(No), log-odds), or (None,)*3.

    `entries` is a list of {token, logprob, top_logprobs:[{token, logprob}, ...]}. Leading
    whitespace/newline tokens are skipped. Case/space variants are summed (logsumexp); a side
    missing from the top-k gets `floor` so one-sided mass is not discarded.
    """
    if not entries:
        return None, None, None
    first = next((e for e in entries if (e.get("token") or "").strip()), None)
    if first is None:
        return None, None, None
    alts = list(first.get("top_logprobs") or [])
    if not alts:
        alts = [{"token": first["token"], "logprob": first["logprob"]}]
    else:
        # Some APIs omit the selected token from top_logprobs; include it so mass is not lost.
        first_tok = (first.get("token") or "").strip().lower()
        if first_tok in {"yes", "no"} and not any((a.get("token") or "").strip().lower() == first_tok for a in alts):
            alts = alts + [{"token": first["token"], "logprob": first["logprob"]}]
    yes = [a["logprob"] for a in alts if (a.get("token") or "").strip().lower() == "yes"]
    no = [a["logprob"] for a in alts if (a.get("token") or "").strip().lower() == "no"]
    if not yes and not no:
        return None, None, None
    return _softmax_yes_no(_logsumexp(yes) if yes else floor, _logsumexp(no) if no else floor)


class OllamaBackend:
    """Local open-weights models via the Ollama REST API.

    Direct prompts request first-token log-probabilities (`logprobs`/`top_logprobs`, supported by
    current Ollama releases; older servers simply omit them and prob_* stay None).
    """

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")

    def predict(self, prompt: str, system_prompt: str, is_cot: bool = False) -> ModelPrediction:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "seed": SEED,
                "num_predict": COT_MAX_TOKENS if is_cot else 8,
            },
        }
        if not is_cot:
            payload.update(logprobs=True, top_logprobs=20)
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        raw = body.get("response", "")
        p_yes = p_no = margin = None
        if not is_cot:
            p_yes, p_no, margin = yes_no_from_top_logprobs(body.get("logprobs"))
        return ModelPrediction(raw_text=raw, predicted_label=parse_binary_label(raw, is_cot=is_cot),
                               prob_yes=p_yes, prob_no=p_no, logprob_margin=margin)


class HuggingFaceBackend:
    """Local causal LMs; direct strategies are scored by exact next-token P(Yes) vs P(No)."""

    YES_VARIANTS = ("Yes", " Yes", "yes", " yes")
    NO_VARIANTS = ("No", " No", "no", " no")

    def __init__(self, model_id: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self.model.eval()
        self.yes_ids = self._single_token_ids(self.YES_VARIANTS)
        self.no_ids = self._single_token_ids(self.NO_VARIANTS)

    def _single_token_ids(self, variants) -> list[int]:
        ids = set()
        for v in variants:
            toks = self.tokenizer.encode(v, add_special_tokens=False)
            if len(toks) == 1:  # keep only variants that are a single token
                ids.add(toks[0])
        return sorted(ids)

    def _encode(self, prompt: str, system_prompt: str):
        """Use the model's chat template when it has one (instruct models), else raw text."""
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
            return {k: v.to(self.model.device) for k, v in ids.items()}
        return self.tokenizer(f"{system_prompt}\n\n{prompt}", return_tensors="pt").to(self.model.device)

    def predict(self, prompt: str, system_prompt: str, is_cot: bool = False) -> ModelPrediction:
        inputs = self._encode(prompt, system_prompt)

        if is_cot:
            with self.torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=COT_MAX_TOKENS, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            raw = self.tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return ModelPrediction(raw_text=raw, predicted_label=parse_binary_label(raw, is_cot=True))

        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1, :]
        log_probs = self.torch.log_softmax(logits.float(), dim=-1)
        if not self.yes_ids or not self.no_ids:
            raise ValueError(
                "Tokenizer has no single-token Yes/No variants; cannot compute direct log-probabilities."
            )
        lp_yes = _logsumexp(float(log_probs[i]) for i in self.yes_ids)
        lp_no = _logsumexp(float(log_probs[i]) for i in self.no_ids)
        p_yes, p_no, margin = _softmax_yes_no(lp_yes, lp_no)
        label = "Yes" if p_yes >= p_no else "No"
        return ModelPrediction(raw_text=label, predicted_label=label, prob_yes=p_yes, prob_no=p_no,
                               logprob_margin=margin)


class OpenAIBackend:
    """OpenAI-compatible chat models (OpenAI, Groq, OpenRouter, etc.) with first-token logprobs for direct prompts."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def predict(self, prompt: str, system_prompt: str, is_cot: bool = False) -> ModelPrediction:
        kwargs = dict(
            model=self.model_name,
            temperature=0.0,
            seed=SEED,
            max_tokens=COT_MAX_TOKENS if is_cot else 64,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        )
        if not is_cot:
            kwargs.update(logprobs=True, top_logprobs=10)
        for attempt in range(6):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                msg = str(e).lower()
                if not is_cot and ("logprobs" in msg and ("not supported" in msg or "unsupported" in msg)):
                    kwargs.pop("logprobs", None)
                    kwargs.pop("top_logprobs", None)
                    response = self.client.chat.completions.create(**kwargs)
                    break
                if "seed" in msg and any(x in msg for x in ["not supported", "unsupported", "unknown", "invalid", "must be"]):
                    kwargs.pop("seed", None)
                    response = self.client.chat.completions.create(**kwargs)
                    break
                if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                    if attempt == 5:
                        raise
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise
        choice = response.choices[0]
        raw = choice.message.content or ""
        if not raw.strip():
            # Some reasoning models return content in reasoning/reasoning_content fields.
            msg = choice.message
            raw = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or (msg.model_extra or {}).get("reasoning_content") or (msg.model_extra or {}).get("reasoning") or ""

        p_yes = p_no = margin = None
        if not is_cot and choice.logprobs and choice.logprobs.content:
            content = choice.logprobs.content
            first = next((e for e in content if (e.token or "").strip()), None)
            if first is not None and first.top_logprobs:
                yes_lps, no_lps = [], []
                for item in first.top_logprobs:
                    tok = item.token.strip().lower()
                    if tok == "yes":
                        yes_lps.append(item.logprob)
                    elif tok == "no":
                        no_lps.append(item.logprob)
                if yes_lps or no_lps:  # a missing side gets a floor so one-sided mass is not discarded
                    floor = -20.0
                    p_yes, p_no, margin = _softmax_yes_no(
                        _logsumexp(yes_lps) if yes_lps else floor,
                        _logsumexp(no_lps) if no_lps else floor,
                    )

        return ModelPrediction(raw_text=raw, predicted_label=parse_binary_label(raw, is_cot=is_cot),
                               prob_yes=p_yes, prob_no=p_no, logprob_margin=margin)

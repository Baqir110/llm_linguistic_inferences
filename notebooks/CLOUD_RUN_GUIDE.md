# Cloud / Free Compute Run Guide

This guide explains how to run the project on Kaggle, Google Colab, Hugging Face, and how to ask University of Bamberg for compute resources.

## Repo

GitHub: https://github.com/Baqir110/llm_linguistic_inferences

> Do **not** upload `apikeys.env` or any real API keys to GitHub/Kaggle/Colab public notebooks.

---

## 1) University of Bamberg (ask for student compute)

I could not verify live pages from this environment, so confirm details on Uni Bamberg IT pages. Ask your course/IT support:

### What to ask
- "I am an MSc ISSS student (BA number: <your BA number>). For course GdI-Proj-B/M LLMs and Linguistic Theory, I need to run LLM evaluations (Python, PyTorch/Hugging Face/Ollama). Is there student access to GPU servers, JupyterHub, HPC, or lab machines?"
- "Can I get temporary access to a machine with NVIDIA GPU (>=16GB VRAM preferred) for running 8B model inference?"
- "Is there any course-provided compute from the chair/supervisor?"

### Where to check
- Uni Bamberg IT Service Desk / Rechenzentrum student services.
- Your course VC page / supervisor Dr. Luke Burke.
- Student portal IT service catalog.

### What to include in request
- Project title, course, supervisor.
- What you need: Python 3.10+, GPU, ~20GB disk, internet for model download.
- Estimated runtime: few hours for 8B model.
- Data is public/synthetic benchmark CSV; no personal data.

---

## 2) Kaggle (recommended free GPU option)

### Pros/limits
- Free GPU quota (commonly ~30 hrs/week; can change).
- T4 16GB is usually enough for 8B HF model with `device_map="auto"` / offload; 70B not feasible.
- Sessions can disconnect; save outputs often.

### Steps
1. Go to https://www.kaggle.com and create/login account.
2. Verify phone number (required for GPU).
3. Create a new Notebook.
4. **Settings**:
   - Accelerator: **GPU T4 x2** or **GPU P100** (T4 is enough).
   - Internet: **On** (needed for pip/model download; for gated HF models also need token).
   - Persistence: Enable if you want files kept.
5. Add code cell to clone repo:
   ```python
   !git clone https://github.com/Baqir110/llm_linguistic_inferences.git
   %cd llm_linguistic_inferences
   ```
   If repo is private, use a Kaggle Secret/GitHub token, never hardcode.
6. Install deps:
   ```python
   !pip -q install -r requirements.txt
   ```
7. For gated HF models (Meta Llama), add HF token as Kaggle Secret:
   - Add-ons > Secrets > `HF_TOKEN`
   - In notebook:
   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   ```
8. Run evaluation:
   ```python
   !python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct --strategies zero_shot few_shot
   !python -m src.analyze
   ```
9. Download results:
   - Right side panel > Output > download `results/` files.
   - Or zip:
   ```python
   !zip -r results_kaggle.zip results
   ```
10. Copy downloaded results back into local repo `results/` and rerun `python -m src.analyze` locally.

### Kaggle troubleshooting
- **OOM**: restart kernel, use `device_map="auto"`, close other notebooks. 8B fp16 may offload to CPU RAM on T4.
- **Gated model 403**: accept Meta license on HF and ensure HF_TOKEN is for same account.
- **Session dies**: results in `results/` are saved incrementally; cache files allow resume if you re-upload workspace.

---

## 3) Google Colab (free, more limited)

### Pros/limits
- Easy to use, free T4 sometimes available.
- Session limits/disconnects; not reliable for long runs.
- Save to Google Drive frequently.

### Steps
1. Open https://colab.research.google.com
2. New notebook.
3. Runtime > Change runtime type > **T4 GPU**.
4. Mount Drive (optional but recommended):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
5. Clone repo:
   ```python
   !git clone https://github.com/Baqir110/llm_linguistic_inferences.git
   %cd llm_linguistic_inferences
   ```
6. Install deps:
   ```python
   !pip -q install -r requirements.txt
   ```
7. Set HF token securely (do not hardcode in shared notebook):
   ```python
   import os
   from getpass import getpass
   os.environ["HF_TOKEN"] = getpass("HF_TOKEN: ")
   ```
8. Run:
   ```python
   !python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct --strategies zero_shot few_shot
   !python -m src.analyze
   ```
9. Copy results to Drive:
   ```python
   !cp -r results /content/drive/MyDrive/llm_project_results
   ```

### Colab troubleshooting
- If disconnected, re-run; cache/resume behavior depends on where cache was saved. Save `results/` to Drive.
- If no GPU available, try later or use Kaggle.

---

## 4) Hugging Face (models/tokens)

### What you need
- HF account: https://huggingface.co
- Token: https://huggingface.co/settings/tokens
- For Meta Llama models: accept license on model page while logged in.

### Local/GPU HF run
```powershell
$env:HF_TOKEN="hf_..."
python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct --strategies zero_shot few_shot
python -m src.analyze
```

### If gated model pending
- Wait for approval; or document as pending.
- Use ungated substitute only if clearly labeled as substitute.

### HF Inference / free endpoints
- Not unlimited. May be useful for smoke tests, but full 540-call eval can hit limits.
- Prefer Kaggle/Colab/local GPU for full run.

---

## 5) Groq / OpenRouter (free-tier APIs, limited)

Already supported by `--base-url` / `OPENAI_BASE_URL`.

### Groq
```powershell
$env:OPENAI_API_KEY="gsk_..."
python -m src.evaluate --backend openai --model openai/gpt-oss-20b --base-url https://api.groq.com/openai/v1
```

### OpenRouter
```powershell
$env:OPENAI_API_KEY="sk-or-v1-..."
python -m src.evaluate --backend openai --model nvidia/nemotron-3.5-lightning:free --base-url https://openrouter.ai/api/v1
```

Expect rate limits/daily caps on free tiers. Rerun resumes from cache.

---

## 6) What to copy back after cloud run

Copy these from cloud `results/` into local `results/`:
- `predictions_<model>.csv`
- `summary_<model>.csv`
- `perturbation_predictions_<model>.csv`
- `robustness_<model>.csv`
- `manifest_<model>.json`

Then run locally:
```powershell
python -m src.analyze
```

This regenerates master/primary/exploratory summaries, significance tests, and plots.

---

## 7) Secrets safety

- Never commit `apikeys.env`.
- Never hardcode keys in notebooks you share.
- Use Kaggle Secrets / Colab getpass / environment variables.
- Revoke any key pasted into chat or public place.

# Runs every model named in the project proposal. Resumable: re-run after a failure.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1
$ErrorActionPreference = "Stop"

$ollamaModels = @("llama3:8b", "mistral:7b", "llama3:70b", "mixtral:8x7b")
foreach ($m in $ollamaModels) {
    ollama pull $m
    python -m src.evaluate --backend ollama --model $m
}

# Requires OPENAI_API_KEY
$openaiModels = @("gpt-4o", "gpt-3.5-turbo")
foreach ($m in $openaiModels) {
    python -m src.evaluate --backend openai --model $m
}

# Exact P(Yes)/P(No) for the open-weights 8B model (needs a GPU):
python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct --strategies zero_shot few_shot

python -m src.analyze

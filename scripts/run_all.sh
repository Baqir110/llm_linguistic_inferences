#!/usr/bin/env bash
# Runs every model named in the project proposal. Resumable: re-run after a failure.
set -e
for m in llama3:8b mistral:7b llama3:70b mixtral:8x7b; do
  ollama pull "$m"
  python -m src.evaluate --backend ollama --model "$m"
done
# needs OPENAI_API_KEY
for m in gpt-4o gpt-3.5-turbo; do
  python -m src.evaluate --backend openai --model "$m"
done
# Exact P(Yes)/P(No) for the open-weights 8B model (needs a GPU):
python -m src.evaluate --backend hf --model meta-llama/Meta-Llama-3-8B-Instruct --strategies zero_shot few_shot
python -m src.analyze

#!/usr/bin/env bash
set -euo pipefail

target_dir=${1:-/data/SurgPub/models/uAI-NEXUS-MedVLM-1.0a-7B-RL}
mkdir -p "$target_dir"
cd "$target_dir"
base_url="https://hf-mirror.com/UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL/resolve/main"

files=(
  config.json generation_config.json added_tokens.json chat_template.jinja
  merges.txt model.safetensors.index.json preprocessor_config.json
  special_tokens_map.json tokenizer.json tokenizer_config.json
  video_preprocessor_config.json vocab.json
  model-00001-of-00004.safetensors model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors model-00004-of-00004.safetensors
)

for file in "${files[@]}"; do
  curl -fL --retry 10 --retry-delay 5 --connect-timeout 30 \
    -C - -o "$file" "$base_url/$file"
done

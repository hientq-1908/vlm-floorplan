#!/usr/bin/env bash
# Serve Qwen3.5-4B (or our LoRA on top of it) behind an OpenAI-compatible API with vLLM.
#
# vLLM is pinned to 0.33.0 on purpose:
#   - 0.33.0 is the first release with the fix for Qwen3.5 LoRA adapters being silently
#     ignored (vLLM PR #62224), and
#   - it is the newest release the pinned TRL (1.18.1) accepts.
# Bump it together with TRL, never alone.
#
# The three parser flags are the Qwen3.5 model card's recommended setup:
#   --reasoning-parser qwen3        splits the <think> block into `reasoning_content`
#   --enable-auto-tool-choice       lets the model call tools (the editing agent, M5)
#   --tool-call-parser qwen3_coder  reads Qwen's XML-style tool calls
#
# Configure through environment variables (defaults in brackets):
#   MODEL [Qwen/Qwen3.5-4B]   SERVED_NAME [qwen3.5-4b]   SERVE_HOST [127.0.0.1]   SERVE_PORT [8000]
#   MAX_MODEL_LEN [20480]     GPU_MEMORY_UTILIZATION [0.90]
#   LORA_MODULES []           e.g. "floorplan=/path/to/adapter" (enables --enable-lora)
#
# Usage: deploy/vllm/serve.sh [extra vllm flags...]
set -euo pipefail

VLLM_VERSION="0.33.0"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-4b}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
# 20480 = a 1280 px plan image (about 1.2K tokens) + prompt + the 12K completion budget
# that thinking mode needs. Raise it if the harness's --max-tokens goes up.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Use an installed vLLM if it is exactly the pinned version, otherwise fetch it with uvx.
if command -v vllm >/dev/null 2>&1 && [ "$(vllm --version 2>/dev/null | tail -n1)" = "$VLLM_VERSION" ]; then
  VLLM=(vllm)
else
  VLLM=(uvx --from "vllm==${VLLM_VERSION}" vllm)
fi

args=(
  serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host "$SERVE_HOST" --port "$SERVE_PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --limit-mm-per-prompt '{"image": 1}'
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)
if [ -n "${LORA_MODULES:-}" ]; then
  args+=(--enable-lora --lora-modules "$LORA_MODULES")
fi

echo "serving ${MODEL} as ${SERVED_NAME} on ${SERVE_HOST}:${SERVE_PORT} (vLLM ${VLLM_VERSION})" >&2
exec "${VLLM[@]}" "${args[@]}" "$@"

# Serving Qwen3.5-4B with vLLM

`serve.sh` starts an OpenAI-compatible server for the base model, and later for our LoRA.

```bash
deploy/vllm/serve.sh                       # base model on 127.0.0.1:8000
LORA_MODULES="floorplan=/path/to/adapter" deploy/vllm/serve.sh
```

## Why these settings

| Setting | Reason |
|---------|--------|
| vLLM `0.33.0` | The first release that applies Qwen3.5 LoRA adapters (vLLM PR 62224; before it adapters loaded but changed nothing) and the newest release the pinned TRL accepts. Bump it together with TRL. |
| `--reasoning-parser qwen3` | Splits the `<think>` block into `reasoning_content`, so `content` holds only the answer. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | The setup the Qwen3.5 model card gives for tool calling, needed by the editing agent. |
| `--limit-mm-per-prompt '{"image": 1}'` | One plan image per request. |
| `MAX_MODEL_LEN=88064` | The longest text-to-graph agent-loop transcript at `t2g-run`'s default `--max-tokens` 8192 (`t2g_harness.context_tokens`); it also covers a 1280 px image, the prompt and the 12K completion budget of the perception harness. The ClearML baseline job sets it from its `--max-tokens`. |

All settings can be changed with environment variables; the list is at the top of the script.
The script uses an installed `vllm` only if it is exactly 0.33.0, and otherwise runs it with `uvx`.

The ClearML agent installs its packages from `requirements-baseline.txt`.
vLLM brings its own torch, so this environment is separate from the project's `ml` dependency group.

## Checks before the first GPU run

The flags follow the Qwen3.5 model card and were not run, because no GPU was available.
Run `vllm serve --help` for 0.33.0 once and confirm the flags above.
Then check the server with:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

When the LoRA arrives, compare a few vLLM outputs against HF PEFT outputs on the same inputs, to catch an adapter that loads but does nothing.

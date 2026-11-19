"""Settings of one evaluation run. Saved as `config.json` in the run directory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vlm_floorplan.evaluation.prompt import PROMPT_VERSION


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    split: str
    """JSONL file of `PlanRecord`s; image paths inside are relative to its directory."""
    base_url: str = "http://localhost:8000/v1"
    model: str
    api_key_env: str | None = None
    """Name of the environment variable holding the key. The key itself is never stored."""
    max_long_side: int = Field(default=1024, gt=0)
    """Longest image side in pixels after client-side resizing. The knob M2 compares."""
    max_tokens: int = Field(default=12288, gt=0)
    """Completion budget. With thinking on, reasoning tokens spend it too."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = 20
    seed: int | None = 0
    enable_thinking: bool | None = True
    """Sent as chat_template_kwargs.enable_thinking; None leaves the server default."""
    concurrency: int = Field(default=4, gt=0)
    limit: int | None = None
    """Evaluate only the first N records (smoke tests)."""
    timeout_s: float = 600.0
    prompt_version: str = PROMPT_VERSION

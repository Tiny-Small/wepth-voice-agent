"""Environment configuration for user-facing replies.

Replies are opt-in and off by default. The deterministic pipeline already produces a
correct outcome; a reply composer only decides how it is *said*, and the template
composer needs no model at all.

    AGENTIC_REPLY_COMPOSER   template (default) | llm | none
    AGENTIC_REPLY_MODEL      HuggingFace id for the llm composer
    AGENTIC_REPLY_DEVICE     auto (default) | cpu | cuda | cuda:0

`template` needs no model and is the default; `none` silences the spoken sentence
without changing the turn's outcome. Only `llm` loads a model.

Defaults are chosen so that nothing new is required to run the spine or the chat UI,
and so that a model is never downloaded by surprise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Kinds accepted for AGENTIC_REPLY_COMPOSER.
TEMPLATE = "template"
LLM = "llm"
NONE = "none"
VALID_KINDS = (TEMPLATE, LLM, NONE)

# Default reply model. Measured, not assumed: on this task a 270M model rephrased only
# 2/6 sentences acceptably and produced fluent but wrong output for others ("Play some
# Jazz" succeeded, yet it answered "Okay, I'm ready. Please provide the text you want me
# to rephrase."). A 0.6B model rephrased 4/6 and stayed on-topic. The deterministic
# template remains the real default; this only matters when replies are switched on.
DEFAULT_REPLY_MODEL = "Qwen/Qwen3-0.6B"

# Evaluated alternative. `google/gemma-3-270m-it` is GATED on HuggingFace (`gated=manual`),
# so it needs licence acceptance and authentication before it will download, and it
# under-performed the 0.6B model above. Supported via AGENTIC_REPLY_MODEL if wanted.
ALT_REPLY_MODEL = "google/gemma-3-270m-it"

# Models that emit a reasoning trace; requires `enable_thinking=False` to suppress it,
# otherwise the raw reply contains the model's deliberation.
THINKING_MODELS = ("Qwen/Qwen3", "Qwen3")

DEFAULT_DEVICE = "auto"


@dataclass(frozen=True)
class ReplySettings:
    """Resolved reply configuration."""

    composer: str = TEMPLATE
    model: str = DEFAULT_REPLY_MODEL
    device: str = DEFAULT_DEVICE

    @property
    def enabled(self) -> bool:
        return self.composer != NONE

    @property
    def uses_model(self) -> bool:
        return self.composer == LLM

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "ReplySettings":
        env = os.environ if environ is None else environ
        kind = (env.get("AGENTIC_REPLY_COMPOSER") or TEMPLATE).strip().casefold()
        if kind == "local_llm":  # accept the alias used in the design notes
            kind = LLM
        if kind not in VALID_KINDS:
            raise ValueError(
                f"AGENTIC_REPLY_COMPOSER must be one of {', '.join(VALID_KINDS)}; got '{kind}'")
        model = (env.get("AGENTIC_REPLY_MODEL") or DEFAULT_REPLY_MODEL).strip()
        if not model:
            raise ValueError("AGENTIC_REPLY_MODEL must not be blank")
        device = (env.get("AGENTIC_REPLY_DEVICE") or DEFAULT_DEVICE).strip().casefold()
        if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("AGENTIC_REPLY_DEVICE must be auto, cpu, cuda, or cuda:N")
        return cls(composer=kind, model=model, device=device)

    def describe(self) -> str:
        if not self.enabled:
            return "replies disabled (AGENTIC_REPLY_COMPOSER=none)"
        if self.uses_model:
            return f"replies via {self.model} on {self.device}"
        return "replies via deterministic template"

"""Environment configuration for the Jev and extraction backends.

Two independent axes, both opt-in so the offline slice keeps working with no
credentials and no model downloads:

    AGENTIC_JEV_BACKEND      rules (default) | live
    AGENTIC_GLOBAL_JEV_MODEL HF/OpenRouter id for Global Jev   (live only)
    AGENTIC_LOCAL_JEV_MODEL  HF/OpenRouter id for Local Jev    (live only)
    AGENTIC_EXTRACTOR        heuristic (default) | minilm | nuextract | luna | llama
    AGENTIC_EXTRACTOR_DEVICE auto (default) | cpu | cuda | cuda:0
    AGENTIC_LLAMA_MODEL      OpenRouter model id for llama (default: meta-llama/llama-3.1-8b-instruct)
    AGENTIC_BROWSER_CONTROLLER luna (default) | llama

`live` uses the `DecisionsProvider` transport (`typesafe/jev-1.13` by default), which
requires `OPENROUTER_API_KEY`. `rules` selects the deterministic stand-ins used by tests
and credential-free demos. The resolved backend is always reported by the chat server so
which Jev answered is never a matter of inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Jev backends.
RULES = "rules"
LIVE = "live"
VALID_JEV_BACKENDS = (RULES, LIVE)

# Extraction backends.
HEURISTIC = "heuristic"
MINILM = "minilm"
NUEXTRACT = "nuextract"
LUNA = "luna"
LLAMA = "llama"
VALID_EXTRACTORS = (HEURISTIC, MINILM, NUEXTRACT, LUNA, LLAMA)
VALID_BROWSER_CONTROLLERS = (LUNA, LLAMA)

# Default Jev model for the voice runtime.
DEFAULT_JEV_MODEL = "typesafe/jev-1.13"

DEFAULT_EXTRACTOR_DEVICE = "auto"
DEFAULT_LUNA_MODEL = "openai/gpt-4.1"
DEFAULT_LLAMA_MODEL = "meta-llama/llama-3.1-8b-instruct"


@dataclass(frozen=True)
class BackendSettings:
    """Resolved Jev + extractor configuration."""

    jev_backend: str = RULES
    global_jev_model: str = DEFAULT_JEV_MODEL
    local_jev_model: str = DEFAULT_JEV_MODEL
    extractor: str = HEURISTIC
    extractor_device: str = DEFAULT_EXTRACTOR_DEVICE
    luna_model: str = DEFAULT_LUNA_MODEL
    llama_model: str = DEFAULT_LLAMA_MODEL
    browser_controller: str = LUNA

    @property
    def live(self) -> bool:
        return self.jev_backend == LIVE

    @property
    def needs_api_key(self) -> bool:
        return self.live or self.extractor in (LUNA, LLAMA)

    @property
    def loads_extraction_model(self) -> bool:
        return self.extractor in (MINILM, NUEXTRACT)

    @property
    def browser_controller_model(self) -> str:
        return self.llama_model if self.browser_controller == LLAMA else self.luna_model

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "BackendSettings":
        env = os.environ if environ is None else environ
        backend = (env.get("AGENTIC_JEV_BACKEND") or RULES).strip().casefold()
        if backend in {"rule", "rule_based", "rule-based", "offline"}:
            backend = RULES
        if backend in {"llm", "openrouter", "decisions"}:
            backend = LIVE
        if backend not in VALID_JEV_BACKENDS:
            raise ValueError(
                f"AGENTIC_JEV_BACKEND must be one of {', '.join(VALID_JEV_BACKENDS)}; got '{backend}'")

        # Fall back to the older names so an existing `.envrc` still selects the model.
        global_model = (env.get("AGENTIC_GLOBAL_JEV_MODEL")
                        or env.get("GLOBAL_ROUTER_MODEL") or DEFAULT_JEV_MODEL).strip()
        local_model = (env.get("AGENTIC_LOCAL_JEV_MODEL")
                       or env.get("JEV_MODEL") or DEFAULT_JEV_MODEL).strip()
        if not global_model or not local_model:
            raise ValueError("Jev model ids must not be blank")

        extractor = (env.get("AGENTIC_EXTRACTOR") or HEURISTIC).strip().casefold()
        if extractor not in VALID_EXTRACTORS:
            raise ValueError(
                f"AGENTIC_EXTRACTOR must be one of {', '.join(VALID_EXTRACTORS)}; got '{extractor}'")

        device = (env.get("AGENTIC_EXTRACTOR_DEVICE") or DEFAULT_EXTRACTOR_DEVICE).strip().casefold()
        if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("AGENTIC_EXTRACTOR_DEVICE must be auto, cpu, cuda, or cuda:N")

        luna_model = (env.get("AGENTIC_LUNA_MODEL") or env.get("JEV_BROWSER_MODEL")
                      or DEFAULT_LUNA_MODEL).strip()
        if not luna_model:
            raise ValueError("AGENTIC_LUNA_MODEL must not be blank")

        llama_model = (env.get("AGENTIC_LLAMA_MODEL") or DEFAULT_LLAMA_MODEL).strip()
        if not llama_model:
            raise ValueError("AGENTIC_LLAMA_MODEL must not be blank")

        browser_controller = (env.get("AGENTIC_BROWSER_CONTROLLER")
                              or env.get("BROWSER_CONTROLLER") or LUNA).strip().casefold()
        if browser_controller not in VALID_BROWSER_CONTROLLERS:
            raise ValueError(
                f"AGENTIC_BROWSER_CONTROLLER must be one of {', '.join(VALID_BROWSER_CONTROLLERS)}; "
                f"got '{browser_controller}'")

        return cls(jev_backend=backend, global_jev_model=global_model,
                   local_jev_model=local_model, extractor=extractor,
                   extractor_device=device, luna_model=luna_model, llama_model=llama_model,
                   browser_controller=browser_controller)

    def describe(self) -> str:
        if self.live:
            jev = f"live Jev [{self.global_jev_model} global / {self.local_jev_model} local]"
        else:
            jev = "rule-based Jev (regex)"
        if self.extractor == LUNA:
            extractor = f"{self.extractor}[{self.luna_model}]"
        elif self.extractor == LLAMA:
            extractor = f"{self.extractor}[{self.llama_model}]"
        else:
            extractor = f"{self.extractor} ({self.extractor_device})"
        return f"{jev}, extractor={extractor}"

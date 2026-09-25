"""Convenience wiring for the voice-to-action spine.

Keeps the vertical slice runnable without credentials by pairing the real Jev
implementations (which need a `DecisionsProvider`) with a provider-free
`RuleBasedLocalJev`/`RuleBasedGlobalJev` used in tests and offline demos. Production
wiring supplies `JevGlobalRouter`/`JevLocalJev` over an OpenRouter Decisions provider.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from .capabilities.browser import BrowserCapability
from .capabilities.spotify import SpotifyCapability
from .capabilities.transfer import TransferCapability, transfer_world_schema
from .goal_builder import GoalBuilder
from .jev import GlobalRoute, LocalRoute
from .registry import CapabilityDescriptor, CapabilityRegistry
from .span import (ArgumentExtractor, ChainExtractor, ExtractiveQAExtractor,
                   HeuristicSpanExtractor, LlamaStructuredSlotExtractor,
                   LunaStructuredSlotExtractor, NuExtractExtractor)
from .world import WorldState


class RuleBasedGlobalJev:
    """Deterministic stand-in for `JevGlobalRouter`; same contract, no provider.

    Used for tests and credential-free demos. It receives utterance and
    `active_local` only - never world state.
    """

    def __init__(self, *, keywords: Mapping[str, tuple[str, ...]] | None = None,
                 model: str = "rule-based") -> None:
        self.model = model
        self.keywords = dict(keywords or {
            "Spotify": ("spotify", "track", "song", "play", "music", "jazz", "album", "artist", "pause",
                        "resume", "skip", "playlist", "spotify"),
            "Browser": ("browser", "google", "search", "search the web", "web", "navigate", "go to", "website",
                        "url", "tab", "back", "forward", "github"),
            "Transfer": ("transfer", "send money", "send", "pay", "balance", "transfer money"),
        })

    async def route(self, utterance: str, *, active_local: str | None = None,
                    context: Mapping[str, Any] | None = None) -> GlobalRoute:
        text = utterance.casefold()
        scores = {name: 0 for name in self.keywords}
        for name, words in self.keywords.items():
            scores[name] = sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", text))
        best = max(scores, key=lambda name: scores[name]) if scores else None
        if best is None or scores[best] == 0:
            # No capability keyword: keep the active local as the prior.
            return GlobalRoute(capability=active_local, confidence=0.2, reason="no_keyword")
        if active_local == best:
            return GlobalRoute(capability=best, confidence=0.9, reason="active_match")
        return GlobalRoute(capability=best, confidence=0.85, reason="keyword")


class RuleBasedLocalJev:
    """Deterministic stand-in for `JevLocalJev`; returns goal types, never actions."""

    def __init__(self, *, goal_patterns: Mapping[str, tuple[str, ...]], model: str = "rule-based") -> None:
        self.model = model
        self._compiled = {goal: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
                          for goal, patterns in goal_patterns.items()}

    async def interpret(self, utterance: str, *, descriptor: CapabilityDescriptor) -> LocalRoute:
        # Patterns are matched in declaration order, so a capability lists its more
        # specific goal types first (e.g. PlayResult before Play). Ordering by schema
        # insertion order instead would let "play it" match the general "play" pattern.
        for goal_type, patterns in self._compiled.items():
            if goal_type not in descriptor.goal_schemas:
                continue
            for pattern in patterns:
                if pattern.search(utterance):
                    return LocalRoute(capability=descriptor.name, goal_type=goal_type,
                                      confidence=0.9, raw_choice=goal_type)
        return LocalRoute(capability=descriptor.name, goal_type=None, confidence=0.2, raw_choice="None")


SPOTIFY_GOALS = {
    # "Play it/that/this" (an anaphor) before the general "play" pattern.
    "PLAY_RESULT": (r"\bplay\s+(?:it|that|this)\b|\bplay\s+the\s+(?:first|top)\b",),
    "PLAY": (r"\bplay\b|\bput on\b|\blisten to\b",),
    "SEARCH": (r"\bsearch\b|\blook for\b",),
    "PAUSE": (r"\bpause\b|\bhold on\b|\bstop\b",),
    "RESUME": (r"\bresume\b|\bcontinue\b|\bunpause\b",),
    "SKIP": (r"\bskip\b|\bnext\b",),
    "OPEN": (r"\bopen\s+spotify\b|\blaunch\s+spotify\b",),
}
BROWSER_GOALS = {
    "FIND": (r"\bfind\b|\blocate\b|\blook\s+for\b",),
    "SEARCH": (r"\bsearch\b|\bgoogle\b|\blook up\b",),
    "NAVIGATE": (r"\bgo to\b|\bnavigate\b|\bopen\b.*\bsite\b|\bvisit\b",),
    "BACK": (r"\bgo back\b|\bback\b",),
    "FORWARD": (r"\bforward\b",),
    "OPEN": (r"\bopen\s+(?:the\s+)?browser\b|\blaunch\s+(?:the\s+)?browser\b",),
}
TRANSFER_GOALS = {
    "TRANSFER": (r"\btransfer\b|\bsend\b|\bpay\b",),
    "CHECK_BALANCE": (r"\bbalance\b",),
    "CANCEL": (r"\bcancel\b|\bforget that\b|\bnever mind\b",),
    "CONFIRM": (r"\bconfirm\b|\byes,? go ahead\b|\bgo ahead\b",),
}


class RuleBasedLocalJevs:
    """Provides one `RuleBasedLocalJev` per capability."""

    def __init__(self) -> None:
        self._by_capability = {
            "Spotify": RuleBasedLocalJev(goal_patterns=SPOTIFY_GOALS),
            "Browser": RuleBasedLocalJev(goal_patterns=BROWSER_GOALS),
            "Transfer": RuleBasedLocalJev(goal_patterns=TRANSFER_GOALS),
        }

    def for_capability(self, name: str) -> RuleBasedLocalJev:
        return self._by_capability[name]


def default_extractor(*, prefer_model: bool = True) -> ArgumentExtractor:
    """Heuristic extractor, optionally fronted by the extractive QA model.

    `deepset/minilm-uncased-squad2` is used when its optional dependencies are
    installed; the heuristic extractor keeps the slice runnable otherwise.
    """
    heuristic = HeuristicSpanExtractor()
    if not prefer_model:
        return heuristic
    return ChainExtractor((ExtractiveQAExtractor(), heuristic))


def build_extractor(settings, *, luna_provider=None, llama_provider=None) -> ArgumentExtractor:
    """Realise the extractor named by `BackendSettings`.

    NuExtract is deliberately **not** wrapped in a `ChainExtractor`. `ChainExtractor`
    only implements the single-slot `extract` interface, not `extract_many`, so
    chaining it would hide the multi-slot batching from the goal builder and turn one
    generation into one call per slot - exactly the cost NuExtract is chosen to avoid.
    """
    from .backend_config import BackendSettings, HEURISTIC, LLAMA, LUNA, MINILM, NUEXTRACT

    if not isinstance(settings, BackendSettings):
        raise TypeError("build_extractor expects BackendSettings")
    if settings.extractor == HEURISTIC:
        return HeuristicSpanExtractor()
    if settings.extractor == MINILM:
        # The QA reader is CPU-only here (device is an int, -1 = CPU), so the chain's
        # heuristic fallback still applies when the reader finds no answer.
        return ChainExtractor((ExtractiveQAExtractor(), HeuristicSpanExtractor()))
    if settings.extractor == NUEXTRACT:
        return NuExtractExtractor(device=settings.extractor_device)
    if settings.extractor == LUNA:
        owns_provider = luna_provider is None
        if luna_provider is None:
            from ping_ponder.providers.openrouter import OpenRouterProvider
            luna_provider = OpenRouterProvider()
        extractor = LunaStructuredSlotExtractor(luna_provider, model=settings.luna_model)
        if owns_provider:
            extractor._owned_provider = luna_provider
        return extractor
    if settings.extractor == LLAMA:
        owns_provider = llama_provider is None
        if llama_provider is None:
            from ping_ponder.providers.openrouter import OpenRouterProvider
            llama_provider = OpenRouterProvider()
        extractor = LlamaStructuredSlotExtractor(llama_provider, model=settings.llama_model)
        if owns_provider:
            extractor._owned_provider = llama_provider
        return extractor
    raise ValueError(f"unknown extractor '{settings.extractor}'")


def build_default_world() -> WorldState:
    """Merged world schema for the capabilities in the default slice."""
    merged: dict[str, Any] = {}
    for capability in (SpotifyCapability, BrowserCapability):
        merged.update(capability.world_schema)
    merged.update(transfer_world_schema())
    return WorldState.from_schema(merged)


def build_default_registry(local_jevs: RuleBasedLocalJevs | None = None, *, adapters=None,
                           execution=None, browser_find=None) -> CapabilityRegistry:
    if adapters is None:
        from .execution_config import build_adapter_bundle
        adapters = build_adapter_bundle(execution)
    jevs = local_jevs or RuleBasedLocalJevs()
    return CapabilityRegistry({
        "Spotify": SpotifyCapability(jevs.for_capability("Spotify"), adapter=adapters.spotify).descriptor(),
        "Browser": BrowserCapability(jevs.for_capability("Browser"), adapter=adapters.browser,
                                     browser_find=browser_find).descriptor(),
        "Transfer": TransferCapability(jevs.for_capability("Transfer")).descriptor(),
    })


def build_default_spine(*, extractor: ArgumentExtractor | None = None,
                        registry: CapabilityRegistry | None = None, execution=None,
                        browser_find=None):
    """Assemble the offline vertical slice: registry + Jevs + extractor + planner + executor."""
    from .spine import VoiceActionSpine

    registry = registry or build_default_registry(execution=execution, browser_find=browser_find)
    return VoiceActionSpine(
        registry=registry,
        global_jev=RuleBasedGlobalJev(),
        goal_builder=GoalBuilder(extractor or default_extractor(prefer_model=False)),
        world=build_default_world(),
    )


def build_reply_composer():
    """Reply composer selected by environment, defaulting to the deterministic template.

    Reads AGENTIC_REPLY_COMPOSER / AGENTIC_REPLY_MODEL / AGENTIC_REPLY_DEVICE. Replies
    are opt-in, so nothing here downloads a model unless it is asked to.
    """
    from .reply import build_reply_composer as _build
    from .reply_config import ReplySettings

    settings = ReplySettings.from_env()
    # `none` must reach the factory: it selects the silent composer, which is not the
    # same thing as the template composer.
    return _build(kind=settings.composer, model_name=settings.model, device=settings.device)


def build_chat_session(auto_confirm: bool = False, replies=None, **kwargs):
    """Build a chat session, with recording Jevs.

    Kept here so the server and the tests assemble the session identically.

    Backends come from `AGENTIC_JEV_BACKEND` / `AGENTIC_EXTRACTOR` unless passed
    explicitly. Both default to the offline choice, so a bare call still needs no
    credentials and downloads nothing.
    """
    from .backend_config import BackendSettings
    from .chat import ChatSession, RecordingGlobalJev, RecordingLocalJev
    from .spine import VoiceActionSpine

    settings = kwargs.pop("settings", None) or BackendSettings.from_env()
    execution = kwargs.pop("execution", None)
    from .execution_config import ExecutionSettings, build_adapter_bundle
    execution = execution or ExecutionSettings.from_env()
    adapters = build_adapter_bundle(execution)
    browser_find = kwargs.pop("browser_find", None)
    extractor = kwargs.pop("extractor", None)
    if extractor is None:
        extractor = build_extractor(settings)

    registry, global_jev = _build_jevs(settings, adapters=adapters, browser_find=browser_find)
    # Wrap each capability's Local Jev so the trace can show what it was given.
    wrapped = {descriptor.name: RecordingLocalJev(descriptor.local_jev) for descriptor in registry}
    recording = CapabilityRegistry({
        name: CapabilityDescriptor(
            name=descriptor.name, description=descriptor.description,
            local_jev=wrapped[descriptor.name], goal_schemas=descriptor.goal_schemas,
            operators=descriptor.operators, world_schema=descriptor.world_schema,
            confirmation_subject=descriptor.confirmation_subject,
            observer=descriptor.observer)
        for name, descriptor in ((item.name, item) for item in registry)
    })
    global_jev = RecordingGlobalJev(global_jev)
    spine = VoiceActionSpine(registry=recording, global_jev=global_jev,
                             goal_builder=GoalBuilder(extractor),
                             world=build_default_world())
    if replies is None:
        replies = build_reply_composer()
    session = ChatSession(spine, global_jev=global_jev, local_jevs=wrapped, replies=replies)
    session.auto_confirm = auto_confirm
    session.backend = settings
    session.extractor_name = extractor.name
    session.slot_extractor = extractor
    session.browser_find = browser_find
    return session


def _build_jevs(settings, *, adapters=None, browser_find=None):
    """Registry + Global Jev for the configured backend.

    `live` shares a single provider across the capabilities and closes over the
    registry, so Global Jev's choices come from whatever is registered.
    """
    if not settings.live:
        return build_default_registry(adapters=adapters, browser_find=browser_find), RuleBasedGlobalJev()

    from ping_ponder.providers.openrouter_decisions import OpenRouterDecisionsProvider

    provider = OpenRouterDecisionsProvider()
    registry, router = build_jev_registry(
        global_provider=provider, local_provider=provider,
        global_model=settings.global_jev_model, local_model=settings.local_jev_model,
        adapters=adapters, browser_find=browser_find)
    # Retained so the owner can close the HTTP client; the router holds no lifecycle.
    router._owned_provider = provider
    return registry, router


async def aclose_jevs(global_jev, extractor=None) -> None:
    """Close providers owned by live Jev and Luna extraction wiring."""
    inner = getattr(global_jev, "_inner", global_jev)
    provider = getattr(inner, "_owned_provider", None)
    if provider is not None:
        await provider.aclose()
    extractor_provider = getattr(extractor, "_owned_provider", None)
    if extractor_provider is not None:
        await extractor_provider.aclose()


def build_jev_registry(*, global_provider, local_provider, global_model: str,
                       local_model: str, adapters=None, browser_find=None
                       ) -> tuple[CapabilityRegistry, "JevGlobalRouter"]:
    """Production wiring: one `JevGlobalRouter` and a per-capability `JevLocalJev`.

    Global Jev's choices come from the registry, so `local_provider`/`local_model`
    are all a new capability needs in order to participate.
    """
    from .capabilities.browser import BrowserCapability
    from .capabilities.spotify import SpotifyCapability
    from .jev import JevGlobalRouter, JevLocalJev

    local_jev = JevLocalJev(local_provider, model=local_model)
    if adapters is None:
        from .execution_config import build_adapter_bundle
        adapters = build_adapter_bundle()
    registry = CapabilityRegistry({
        "Spotify": SpotifyCapability(local_jev, adapter=adapters.spotify).descriptor(),
        "Browser": BrowserCapability(local_jev, adapter=adapters.browser,
                                     browser_find=browser_find).descriptor(),
        "Transfer": TransferCapability(local_jev).descriptor(),
    })
    return registry, JevGlobalRouter(global_provider, registry, model=global_model)

"""Backend selection: Jev and extractor come from the environment, offline by default."""

import pytest

from ping_ponder.agentic.backend_config import (DEFAULT_JEV_MODEL, DEFAULT_LLAMA_MODEL,
                                                BackendSettings, HEURISTIC, LIVE, RULES)


def test_defaults_are_offline_and_download_nothing():
    settings = BackendSettings.from_env({})
    assert settings.jev_backend == RULES and not settings.live
    assert settings.extractor == HEURISTIC and not settings.loads_extraction_model
    assert not settings.needs_api_key


def test_live_backend_reads_the_pinned_jev_model():
    settings = BackendSettings.from_env({"AGENTIC_JEV_BACKEND": "live"})
    assert settings.live
    assert settings.global_jev_model == DEFAULT_JEV_MODEL == "typesafe/jev-1.13"
    assert settings.local_jev_model == DEFAULT_JEV_MODEL


def test_live_backend_falls_back_to_the_older_env_var_names():
    """An existing `.envrc` exporting JEV_MODEL/GLOBAL_ROUTER_MODEL still selects them."""
    settings = BackendSettings.from_env({
        "AGENTIC_JEV_BACKEND": "live",
        "JEV_MODEL": "local-pinned", "GLOBAL_ROUTER_MODEL": "global-pinned"})
    assert settings.global_jev_model == "global-pinned"
    assert settings.local_jev_model == "local-pinned"


def test_explicit_agentic_vars_win_over_the_older_names():
    settings = BackendSettings.from_env({
        "AGENTIC_GLOBAL_JEV_MODEL": "new-global", "GLOBAL_ROUTER_MODEL": "old-global",
        "AGENTIC_LOCAL_JEV_MODEL": "new-local", "JEV_MODEL": "old-local"})
    assert settings.global_jev_model == "new-global" and settings.local_jev_model == "new-local"


def test_extractor_selection_and_validation():
    assert BackendSettings.from_env({"AGENTIC_EXTRACTOR": "nuextract"}).extractor == "nuextract"
    luna = BackendSettings.from_env({"AGENTIC_EXTRACTOR": "luna", "AGENTIC_LUNA_MODEL": "luna/model"})
    assert luna.extractor == "luna" and luna.luna_model == "luna/model"
    assert luna.needs_api_key and not luna.loads_extraction_model
    llama = BackendSettings.from_env({"AGENTIC_EXTRACTOR": "llama"})
    assert llama.extractor == "llama" and llama.llama_model == DEFAULT_LLAMA_MODEL
    assert llama.needs_api_key and not llama.loads_extraction_model
    assert BackendSettings.from_env({"AGENTIC_EXTRACTOR": "llama",
                                     "AGENTIC_LLAMA_MODEL": "test/custom-llama"}).llama_model == "test/custom-llama"
    assert BackendSettings.from_env({"AGENTIC_EXTRACTOR": "nuextract"}).loads_extraction_model
    assert BackendSettings.from_env({"AGENTIC_EXTRACTOR": "minilm"}).extractor == "minilm"
    with pytest.raises(ValueError):
        BackendSettings.from_env({"AGENTIC_EXTRACTOR": "banana"})
    with pytest.raises(ValueError):
        BackendSettings.from_env({"AGENTIC_JEV_BACKEND": "banana"})
    with pytest.raises(ValueError):
        BackendSettings.from_env({"AGENTIC_EXTRACTOR_DEVICE": "tpu"})


def test_aliases_resolve_to_the_two_backends():
    for alias in ("rules", "rule", "rule_based", "offline"):
        assert BackendSettings.from_env({"AGENTIC_JEV_BACKEND": alias}).jev_backend == RULES
    for alias in ("live", "llm", "openrouter", "decisions"):
        assert BackendSettings.from_env({"AGENTIC_JEV_BACKEND": alias}).jev_backend == LIVE


def test_describe_names_the_jev_and_extractor():
    offline = BackendSettings.from_env({}).describe()
    assert "rule-based" in offline and "heuristic" in offline
    live = BackendSettings.from_env({"AGENTIC_JEV_BACKEND": "live",
                                     "AGENTIC_EXTRACTOR": "nuextract"}).describe()
    assert "typesafe/jev-1.13" in live and "nuextract" in live


def test_build_extractor_selects_the_named_backend(monkeypatch):
    """NuExtract must not be chained: ChainExtractor has no extract_many."""
    from ping_ponder.agentic.span import (ChainExtractor, HeuristicSpanExtractor,
                                          NuExtractExtractor)
    from ping_ponder.agentic.wiring import build_extractor

    assert isinstance(build_extractor(BackendSettings.from_env({})), HeuristicSpanExtractor)
    minilm = build_extractor(BackendSettings.from_env({"AGENTIC_EXTRACTOR": "minilm"}))
    assert isinstance(minilm, ChainExtractor)
    nuextract = build_extractor(BackendSettings.from_env({"AGENTIC_EXTRACTOR": "nuextract"}))
    assert isinstance(nuextract, NuExtractExtractor)
    # The whole point of choosing NuExtract is one generation per goal, so it must
    # advertise the multi-slot interface.
    from ping_ponder.agentic.span import MultiSlotExtractor

    assert isinstance(nuextract, MultiSlotExtractor)

    from ping_ponder.agentic.span import LunaStructuredSlotExtractor
    luna_settings = BackendSettings.from_env({"AGENTIC_EXTRACTOR": "luna", "AGENTIC_LUNA_MODEL": "test/luna"})
    luna = build_extractor(luna_settings, luna_provider=object())
    assert isinstance(luna, LunaStructuredSlotExtractor)
    assert luna.model == "test/luna"

    from ping_ponder.agentic.span import LlamaStructuredSlotExtractor
    llama_settings = BackendSettings.from_env({"AGENTIC_EXTRACTOR": "llama"})
    llama = build_extractor(llama_settings, llama_provider=object())
    assert isinstance(llama, LlamaStructuredSlotExtractor)
    assert llama.model == DEFAULT_LLAMA_MODEL


def test_chat_session_defaults_to_rule_based_jevs():
    from ping_ponder.agentic.wiring import build_chat_session

    session = build_chat_session(settings=BackendSettings.from_env({}))
    assert session.backend.jev_backend == RULES
    assert session.global_jev.name == "RuleBasedGlobalJev"
    assert session.extractor_name == "heuristic"


def test_chat_session_reports_the_selected_backends():
    from ping_ponder.agentic.wiring import build_chat_session

    state = build_chat_session(settings=BackendSettings.from_env({})).state()
    assert state["backends"]["jev"] == RULES
    assert state["backends"]["extractor"] == "heuristic"
    assert "rule-based" in state["backends"]["describe"]


def test_live_backend_builds_jev_models_from_settings(monkeypatch):
    """`live` must reach JevGlobalRouter/JevLocalJev with the configured model ids.

    A stub provider stands in for OpenRouter so this runs offline.
    """
    from ping_ponder.agentic.wiring import build_chat_session
    import ping_ponder.agentic.wiring as wiring

    captured = {}

    class StubProvider:
        async def decide(self, *, model, state, questions):
            raise AssertionError("the stub is only here to be constructed")

        async def aclose(self):
            captured["closed"] = True

    monkeypatch.setattr(
        "ping_ponder.providers.openrouter_decisions.OpenRouterDecisionsProvider",
        lambda: StubProvider())

    settings = BackendSettings.from_env({
        "AGENTIC_JEV_BACKEND": "live",
        "AGENTIC_GLOBAL_JEV_MODEL": "g-model",
        "AGENTIC_LOCAL_JEV_MODEL": "l-model"})
    session = build_chat_session(settings=settings)
    assert session.global_jev.name == "global:g-model"
    assert next(iter(session.local_jevs.values())).name == "local:l-model"
    assert session.state()["backends"]["jev"] == LIVE


def test_available_reports_a_missing_dependency_as_false(monkeypatch):
    """A missing dependency is reportable up front, not only at extraction time.

    Without this, a backend that cannot load surfaces only as the slot reason
    `error:ExtractorUnavailable`, which reads as "the user said no query".
    """
    from ping_ponder.agentic.span import ExtractorUnavailable, NuExtractExtractor

    extractor = NuExtractExtractor()

    def missing():
        raise ExtractorUnavailable("torch is not installed")

    monkeypatch.setattr(extractor, "_load", missing)
    assert extractor.available() is False


def test_available_propagates_a_network_error(monkeypatch):
    """`available()` only answers "are the dependencies installed".

    A blocked network while resolving a remote-code model is not the same as an
    uninstalled backend, so it must not be reported as unavailable. Callers that
    preflight have to tolerate it, which the chat script does.
    """
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()

    def boom():
        raise ConnectionError("no network")

    monkeypatch.setattr(extractor, "_load", boom)
    with pytest.raises(ConnectionError):
        extractor.available()

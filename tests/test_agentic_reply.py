"""Reply layer: deterministic phrasing, and the guards on the opt-in LLM rephraser."""

import pytest

from ping_ponder.agentic.reply import (LocalLLMReplyComposer, ReplyContext, TemplateReplyComposer,
                                       TurnSituation, build_reply_composer, classify,
                                       meaning_preserved, rephrase_is_acceptable, sanitize,
                                       speakable_reason, violates_safety)
from ping_ponder.agentic.reply_config import ReplySettings
from ping_ponder.agentic.wiring import build_chat_session

TEMPLATE = TemplateReplyComposer()


def context(situation, **kwargs):
    return ReplyContext(situation=situation, **kwargs)


# --- classification ------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,expected", [
    (dict(outcome="SATISFIED", capability="Spotify", goal="Spotify.PLAY()"), TurnSituation.COMPLETED),
    (dict(outcome="SPECULATED", capability="Spotify", goal="Spotify.PLAY()"), TurnSituation.SPECULATED),
    (dict(outcome="INFEASIBLE", capability="Transfer", goal="Transfer.TRANSFER()"), TurnSituation.INFEASIBLE),
    (dict(outcome="BLOCKED", capability="Browser", goal="Browser.BACK()"), TurnSituation.INFEASIBLE),
    (dict(outcome="FAILED", capability="Spotify", goal="Spotify.PLAY()"), TurnSituation.FAILED),
    (dict(outcome=None, capability=None, goal=None), TurnSituation.NO_CAPABILITY),
    (dict(outcome=None, capability="Spotify", goal="Spotify.PLAY()"), TurnSituation.NO_GOAL),
    (dict(outcome=None, capability="Spotify", goal="Spotify.PLAY()", missing_slots=("query",)),
     TurnSituation.MISSING_ARGUMENT),
    (dict(outcome=None, capability="Spotify", goal=None, discarded="switched_to_Browser"),
     TurnSituation.DISCARDED),
])
def test_classify_maps_outcomes_to_situations(kwargs, expected):
    assert classify(**kwargs) is expected


def test_blank_slots_are_accepted_from_the_chat_trace():
    assert classify(outcome=None, capability="Spotify", goal="Spotify.PLAY()",
                    missing_slots=[]) is TurnSituation.NO_GOAL


# --- template composer ---------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("situation,reason,executed,fragment", [
    (TurnSituation.COMPLETED, None, ("PlayResult",), "Done"),
    (TurnSituation.COMPLETED, None, (), "already the case"),
    (TurnSituation.NO_CAPABILITY, None, (), "not able to help"),
    (TurnSituation.MISSING_ARGUMENT, None, (), "still need"),
    (TurnSituation.FAILED, None, (), "went wrong"),
    (TurnSituation.SPECULATED, None, (), "moment"),
])
async def test_template_replies_are_short_and_speakable(situation, reason, executed, fragment):
    reply = await TEMPLATE.compose(
        context(situation, reason=reason, executed=executed, missing_slots=("query",)))
    assert fragment in reply and len(reply) < 120


@pytest.mark.asyncio
async def test_infeasible_reply_states_the_cause_in_user_terms():
    reply = await TEMPLATE.compose(context(
        TurnSituation.INFEASIBLE,
        reason="CheckLimit requires transfer.amount_within_limit predicate"))
    assert reply == "I can't do that - that amount is over the limit."
    # The internal key must never reach the user.
    assert "predicate" not in reply and "transfer." not in reply


@pytest.mark.asyncio
async def test_missing_argument_names_the_slot_in_plain_words():
    reply = await TEMPLATE.compose(context(TurnSituation.MISSING_ARGUMENT, missing_slots=("query",)))
    assert "what you'd like me to use" in reply and "query" not in reply


def test_speakable_reason_returns_none_for_unknown_vocabulary():
    assert speakable_reason(None) is None
    assert speakable_reason("some unknown internal thing") is None
    assert speakable_reason("spotify.current_track is true") == "nothing is playing at the moment"


def test_sanitize_strips_internal_identifiers():
    cleaned = sanitize("cannot achieve Transfer.TRANSFER(): CheckLimit failed: "
                       "OperatorError: amount 5000 exceeds the limit")
    assert "OperatorError" not in cleaned and "CheckLimit" not in cleaned


# --- safety guards -------------------------------------------------------------------

@pytest.mark.parametrize("reply", [
    "Done, I've sent the transfer.",
    "I've opened Spotify and it is playing.",
    "The transfer completed successfully.",
])
def test_completion_claims_are_rejected_for_failure_situations(reply):
    assert violates_safety(reply, TurnSituation.INFEASIBLE) is not None


def test_completion_claims_are_allowed_only_for_completed():
    assert violates_safety("Done.", TurnSituation.COMPLETED) is None


def test_polarity_inversion_is_rejected():
    """The observed failure: a fluent rephrase that dropped the negation."""
    locked = "I couldn't work out what you wanted - could you say that another way?"
    assert meaning_preserved("I could say that another way.", locked) is False


@pytest.mark.parametrize("candidate", [
    "I'm not sure what you wanted - could you rephrase that?",
    "I didn't follow that - could you say it differently?",
])
def test_faithful_rephrases_are_accepted(candidate):
    locked = "I couldn't work out what you wanted - could you say that another way?"
    assert meaning_preserved(candidate, locked) is True


def test_third_person_narration_is_rejected():
    problem = rephrase_is_acceptable("The user played some Jazz.", "Done.", TurnSituation.COMPLETED)
    assert problem is not None and "narration" in problem


def test_added_question_is_rejected():
    problem = rephrase_is_acceptable("Done. Anything else?", "Done.", TurnSituation.COMPLETED)
    assert problem is not None


# --- LLM composer: fallback behaviour without loading a model ------------------------

@pytest.mark.asyncio
async def test_llm_composer_falls_back_when_the_model_cannot_load(monkeypatch):
    composer = LocalLLMReplyComposer("does-not-exist/model")

    def boom():
        raise RuntimeError("no model")

    monkeypatch.setattr(composer, "_load", boom)
    reply = await composer.compose(context(TurnSituation.NO_CAPABILITY))
    assert reply == await TEMPLATE.compose(context(TurnSituation.NO_CAPABILITY))


@pytest.mark.asyncio
async def test_llm_composer_rejects_a_reply_that_claims_completion(monkeypatch):
    composer = LocalLLMReplyComposer("stub")
    monkeypatch.setattr(composer, "_generate", lambda ctx, locked: "Done, I've sent it.")
    ctx = context(TurnSituation.INFEASIBLE,
                  reason="CheckLimit requires transfer.amount_within_limit predicate")
    reply = await composer.compose(ctx)
    # The unsafe candidate is discarded and the deterministic sentence is used.
    assert reply == await TEMPLATE.compose(ctx)


@pytest.mark.asyncio
async def test_llm_composer_accepts_a_faithful_rephrase(monkeypatch):
    composer = LocalLLMReplyComposer("stub")
    monkeypatch.setattr(composer, "_generate",
                        lambda ctx, locked: "Sorry, I can't do that - that amount is over the limit.")
    ctx = context(TurnSituation.INFEASIBLE,
                  reason="CheckLimit requires transfer.amount_within_limit predicate")
    reply = await composer.compose(ctx)
    assert reply.startswith("Sorry") and "amount is over the limit" in reply


# --- configuration -------------------------------------------------------------------

def test_replies_default_to_the_deterministic_template():
    settings = ReplySettings.from_env({})
    assert settings.composer == "template" and not settings.uses_model


def test_reply_configuration_reads_env_and_validates():
    settings = ReplySettings.from_env({"AGENTIC_REPLY_COMPOSER": "llm",
                                       "AGENTIC_REPLY_MODEL": "some/model",
                                       "AGENTIC_REPLY_DEVICE": "cuda"})
    assert settings.uses_model and settings.model == "some/model" and settings.device == "cuda"
    assert ReplySettings.from_env({"AGENTIC_REPLY_COMPOSER": "local_llm"}).composer == "llm"
    assert ReplySettings.from_env({"AGENTIC_REPLY_COMPOSER": "none"}).enabled is False
    with pytest.raises(ValueError):
        ReplySettings.from_env({"AGENTIC_REPLY_COMPOSER": "banana"})
    with pytest.raises(ValueError):
        ReplySettings.from_env({"AGENTIC_REPLY_DEVICE": "tpu"})


def test_composer_factory():
    assert build_reply_composer(kind="template", model_name="m", device="cpu").name == "template"
    assert build_reply_composer(kind="none", model_name="m", device="cpu").name == "none"
    assert isinstance(build_reply_composer(kind="llm", model_name="m", device="cpu"),
                      LocalLLMReplyComposer)
    with pytest.raises(ValueError):
        build_reply_composer(kind="nope", model_name="m", device="cpu")


@pytest.mark.asyncio
async def test_none_is_silent_not_an_alias_for_template():
    """`none` must actually suppress the sentence, not fall back to phrasing it."""
    from ping_ponder.agentic.reply import SilentReplyComposer

    composer = build_reply_composer(kind="none", model_name="m", device="cpu")
    assert isinstance(composer, SilentReplyComposer)
    assert await composer.compose(context(TurnSituation.COMPLETED, capability="Spotify")) is None


@pytest.mark.asyncio
async def test_none_setting_reaches_the_chat_session(monkeypatch):
    """The env var must survive wiring; it used to be swallowed before the factory."""
    monkeypatch.setenv("AGENTIC_REPLY_COMPOSER", "none")
    session = build_chat_session()
    trace = await session.say("play some jazz")
    assert session.replies.name == "none"
    assert trace.reply is None
    # Suppressing the sentence must not change what the system actually did.
    assert trace.outcome == "SATISFIED"


@pytest.mark.asyncio
async def test_template_setting_still_phrases_the_turn(monkeypatch):
    """Control for the test above: the default path is unaffected."""
    monkeypatch.setenv("AGENTIC_REPLY_COMPOSER", "template")
    session = build_chat_session()
    trace = await session.say("play some jazz")
    assert session.replies.name == "template"
    assert trace.reply == "Done."


# --- end to end ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_turn_gets_a_reply_and_the_outcome_is_unchanged():
    service = build_chat_session()
    for text, expected_outcome in [("Play some Jazz", "SATISFIED"),
                                   ("Play some", None),
                                   ("What's the weather in Paris?", None),
                                   ("Transfer money to Sarah for 5000", "INFEASIBLE")]:
        trace = await service.say(text)
        payload = trace.as_dict()
        assert payload["outcome"] == expected_outcome
        assert payload["reply"], f"no reply for {text!r}"
        assert payload["reply_composer"] == "template"


@pytest.mark.asyncio
async def test_reply_for_an_infeasible_transfer_is_truthful():
    service = build_chat_session()
    trace = await service.say("Transfer money to Sarah for 5000")
    assert trace.outcome == "INFEASIBLE"
    assert trace.reply == "I can't do that - that amount is over the limit."
    # Nothing may have been prepared or confirmed.
    assert service.spine.world.get("transfer.prepared") is False
    assert service.spine.world.get("transfer.confirmed") is False


# --- guards: bugs found by running a real 270M model ---------------------------------
# gemma-3-270m-it exposed all three of these. They are regression tests for the guard
# itself, not for any particular model.

def test_negated_completion_phrase_is_not_a_claim():
    """'nothing is playing' is a truthful failure message, not a success claim.

    A naive substring match flagged the banned phrase 'is playing' inside it, which
    rejected every legitimate rephrase of that sentence.
    """
    from ping_ponder.agentic.reply import violates_safety

    truthful = [
        "I can't do that - nothing is playing at the moment.",
        "I cannot do that, and nothing has been changed.",
        "Nothing has been sent.",
    ]
    for reply in truthful:
        assert violates_safety(reply, TurnSituation.INFEASIBLE) is None, reply


def test_asserted_completion_phrases_are_still_caught():
    from ping_ponder.agentic.reply import violates_safety

    for reply in ["I have sent the transfer.", "It is playing now.",
                  "The transfer completed successfully.", "Done, I've opened it."]:
        assert violates_safety(reply, TurnSituation.INFEASIBLE) is not None, reply


def test_hypothetical_completion_phrase_is_not_a_claim():
    from ping_ponder.agentic.reply import violates_safety

    assert violates_safety("If it is playing, I will stop it.", TurnSituation.INFEASIBLE) is None


def test_a_short_locked_reply_cannot_be_replaced_with_fluent_nonsense():
    """Observed: 'Done.' became 'Okay, I'm ready. Please provide the text...'.

    A two-word locked sentence carries too little content to check overlap against, so
    it cleared every guard while saying something unrelated.
    """
    bad = "Okay, I'm ready. Please provide the text you want me to rephrase."
    problem = rephrase_is_acceptable(bad, "Done.", TurnSituation.COMPLETED)
    assert problem is not None and "added content" in problem


@pytest.mark.parametrize("candidate", ["Done.", "All done.", "Okay, that's done."])
def test_faithful_rephrase_of_a_short_reply_is_accepted(candidate):
    assert rephrase_is_acceptable(candidate, "Done.", TurnSituation.COMPLETED) is None


def test_off_topic_replacement_of_a_negative_reply_is_rejected():
    bad = "Okay, I understand. I can help you with that."
    problem = rephrase_is_acceptable(bad, "I can't do that - that amount is over the limit.",
                                     TurnSituation.INFEASIBLE)
    assert problem is not None


# --- attribution: a broken backend must be visible ------------------------------------

@pytest.mark.asyncio
async def test_trace_reports_the_fallback_when_the_model_fails(monkeypatch):
    """A total model failure must not look like the model succeeded.

    This is how a mis-authenticated, never-loading backend went unnoticed: the template
    supplied the text while the trace still claimed the LLM did.
    """
    composer = LocalLLMReplyComposer("stub")

    def boom():
        raise OSError("401 gated repo")

    monkeypatch.setattr(composer, "_load", boom)
    service = build_chat_session(replies=composer)
    trace = await service.say("Transfer money to Sarah for 5000")
    payload = trace.as_dict()
    assert payload["reply"]  # a reply is still produced
    assert "fallback" in (payload["reply_composer"] or ""), payload["reply_composer"]
    assert "model_error" in composer.last_source


@pytest.mark.asyncio
async def test_trace_reports_the_llm_when_it_actually_rephrased(monkeypatch):
    composer = LocalLLMReplyComposer("stub")
    monkeypatch.setattr(composer, "_generate", lambda ctx, locked: locked)
    service = build_chat_session(replies=composer)
    trace = await service.say("Play some Jazz")
    assert trace.reply_composer == "llm:stub" and composer.last_source == "llm"


def test_health_reports_an_unloadable_model(monkeypatch):
    composer = LocalLLMReplyComposer("stub")
    monkeypatch.setattr(composer, "_load", lambda: (_ for _ in ()).throw(OSError("gated")))
    ok, detail = composer.health()
    assert ok is False and "gated" in detail


def test_default_reply_model_is_the_measured_better_one():
    """The default reflects measurement: 0.6B rephrased 4/6 acceptably, the 270M 2/6."""
    from ping_ponder.agentic.reply_config import ALT_REPLY_MODEL, DEFAULT_REPLY_MODEL

    assert DEFAULT_REPLY_MODEL == "Qwen/Qwen3-0.6B"
    assert ALT_REPLY_MODEL == "google/gemma-3-270m-it"

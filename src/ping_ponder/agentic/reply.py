"""User-facing replies when a turn produces no goal or cannot be carried out.

The architecture separates *what happened* from *how it is said*:

    deterministic pipeline -> ReplyContext (facts)
    ReplyComposer          -> a spoken sentence

The composer is presentation only. It can never decide that an action happened: the
outcome is settled by the planner and executor before a reply is composed. `COMPLETED`
is the only situation that permits a success claim, and both backends enforce that, so a
small model cannot talk the system into reporting a transfer it did not make.

Two backends:

* `TemplateReplyComposer` (default) - deterministic phrasing from structured facts. No
  dependencies, no model, no hallucination risk, and fast enough for the partial-speech
  path.
* `LocalLLMReplyComposer` (opt-in) - phrases the same facts with a small local instruct
  model. Better for the open-ended "no goal produced" case; see `reply_config` for the
  environment variables that enable it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from ping_ponder.observability import emit

logger = logging.getLogger(__name__)


class TurnSituation(StrEnum):
    """Deterministic classification of how a turn ended."""

    COMPLETED = "COMPLETED"
    SPECULATED = "SPECULATED"
    NO_GOAL = "NO_GOAL"
    MISSING_ARGUMENT = "MISSING_ARGUMENT"
    INFEASIBLE = "INFEASIBLE"
    FAILED = "FAILED"
    NO_CAPABILITY = "NO_CAPABILITY"
    DISCARDED = "DISCARDED"


# Only this situation may produce a success claim.
_SUCCESS_SITUATIONS = frozenset({TurnSituation.COMPLETED})

# Phrases that assert an action happened. A reply for a non-success situation must not
# contain any of these: that is the safety rule this module exists to hold.
_COMPLETION_CLAIMS = (
    "i've", "i have", "done", "completed", "successfully", "now playing", "is playing",
    "has been", "i opened", "i played", "i sent", "i started", "transfer complete",
    "i've set", "i have set",
)

# Internal identifiers that must never reach a user's ear.
_JARGON = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|OperatorError|\w+Error|no_goal|missing_slot|"
    r"bound predicate|presupposes|cannot achieve|no plan within depth)\b")


@dataclass(frozen=True)
class ReplyContext:
    """The deterministic facts a reply may be built from.

    Deliberately structured rather than a raw reason string: a composer should phrase
    known facts, not parse internal diagnostics. `reason` is carried for the opt-in LLM
    backend but is treated as untrusted text and sanitized before use.
    """

    situation: TurnSituation
    utterance: str = ""
    capability: str | None = None
    goal_type: str | None = None
    missing_slots: tuple[str, ...] = ()
    reason: str | None = None
    executed: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def classify(*, outcome: str | None, capability: str | None, goal: str | None,
             missing_slots: tuple[str, ...] | list[str] = (),
             mode: str = "final", discarded: str | None = None) -> TurnSituation:
    """Map a spine/chat outcome onto a situation the reply layer understands."""
    if discarded and not goal:
        return TurnSituation.DISCARDED
    if outcome == "SATISFIED":
        return TurnSituation.COMPLETED
    if outcome == "SPECULATED":
        return TurnSituation.SPECULATED
    if capability is None and goal is None:
        return TurnSituation.NO_CAPABILITY
    if missing_slots:
        return TurnSituation.MISSING_ARGUMENT
    if goal is None:
        return TurnSituation.NO_GOAL
    if outcome == "FAILED":
        return TurnSituation.FAILED
    if outcome in {"BLOCKED", "INFEASIBLE"}:
        return TurnSituation.INFEASIBLE
    return TurnSituation.NO_GOAL


# A completion phrase is only a claim when it is *asserted*. Negated or hypothetical
# mentions are fine: "nothing is playing" is a truthful failure message, not a claim.
_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:not|no|never|cannot|can't|couldn't|won't|don't|doesn't|didn't|isn't|aren't|"
    r"nothing|none|unable)\b[\w\s,'-]{0,20})$", re.IGNORECASE)
_HYPOTHETICAL_BEFORE = re.compile(r"\b(?:if|whether|when|unless)\b[\w\s,'-]{0,20}$", re.IGNORECASE)


def violates_safety(reply: str, situation: TurnSituation) -> str | None:
    """Return the offending phrase if a reply *asserts* an unearned completion.

    A substring match is not enough: the banned phrases appear legitimately inside
    negated clauses. "I can't do that - nothing is playing at the moment." is truthful,
    but a naive check flags "is playing" and rejects every rephrase of that sentence.
    So a match only counts as a claim when the run-up to it contains no negation or
    hypothetical marker.
    """
    if situation in _SUCCESS_SITUATIONS:
        return None
    lowered = reply.casefold()
    for phrase in _COMPLETION_CLAIMS:
        start = 0
        while True:
            index = lowered.find(phrase, start)
            if index < 0:
                break
            prefix = lowered[max(0, index - 40):index]
            if not (_NEGATION_BEFORE.search(prefix) or _HYPOTHETICAL_BEFORE.search(prefix)):
                return phrase
            start = index + len(phrase)
    return None


def sanitize(reason: str | None) -> str:
    """Reduce an internal diagnostic to something safe to show a model.

    Operator names, error types, and planner vocabulary are stripped rather than
    explained, because a user should never hear them and a small model will happily
    repeat them verbatim.
    """
    if not reason:
        return ""
    text = reason.split("failed steps:", 1)[0]
    text = text.replace("cannot achieve", "cannot do")
    text = text.replace("presupposes", "requires")
    text = re.sub(r"no plan within depth \d+", "no workable plan", text)
    text = _JARGON.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip(" :;,")


# Negation and absence markers. A reply that drops one of these has inverted its
# meaning: "I couldn't work out what you wanted" becoming "I could say that another way"
# is fluent, contains no completion claim, and is simply wrong. Polarity is the signal
# that catches it.
_NEGATIVE = re.compile(
    r"\b(?:not|no|never|cannot|can't|couldn't|won't|don't|doesn't|didn't|isn't|aren't|"
    r"nothing|none|unable|over the limit|too much)\b", re.IGNORECASE)


# Beyond this multiple of the locked length, a "rephrase" has added content rather than
# reworded. A short locked sentence is the dangerous case: it carries too little content
# to check overlap against, so length is the only signal that the model went off-script.
_MAX_LENGTH_RATIO = 3.0


def length_ratio_ok(candidate: str, locked: str) -> bool:
    locked_words = len(locked.split())
    if not locked_words:
        return False
    return len(candidate.split()) / locked_words <= _MAX_LENGTH_RATIO


def meaning_preserved(candidate: str, locked: str, *, min_length_ratio: float = 0.55) -> bool:
    """Whether a rephrase kept the locked sentence's meaning.

    Word-overlap retention was tried first and rejected: a *good* paraphrase deliberately
    changes vocabulary, so it scored worse than the corruption it was meant to catch
    (0.29 vs 0.43 on measured examples). Two signals do separate them:

    * **Polarity** - if the locked sentence negates and the candidate does not (or vice
      versa), the meaning has been inverted. This is what catches the observed failure.
    * **Length** - a rephrase may be longer, but one that collapses to a fraction of the
      original has dropped content.
    """
    if not candidate.strip():
        return False
    locked_negative = bool(_NEGATIVE.search(locked))
    candidate_negative = bool(_NEGATIVE.search(candidate))
    if locked_negative != candidate_negative:
        return False
    locked_words = len(locked.split())
    if locked_words and len(candidate.split()) / locked_words < min_length_ratio:
        return False
    return True


def rephrase_is_acceptable(candidate: str, locked: str, situation: TurnSituation) -> str | None:
    """Why a rephrased reply is unusable, or None when it is acceptable.

    Guards the two failure modes observed in practice: claiming an unearned completion,
    and drifting into third-person narration ("The user played some Jazz") instead of
    speaking to the user.
    """
    if not candidate.strip():
        return "empty"
    offending = violates_safety(candidate, situation)
    if offending:
        return f"completion claim: {offending}"
    lowered = candidate.casefold().strip()
    for narration in ("the user ", "the assistant ", "the system "):
        if lowered.startswith(narration):
            return f"third-person narration: {narration.strip()}"
    if "?" in candidate and "?" not in locked:
        return "added a question the locked reply did not ask"
    if not meaning_preserved(candidate, locked):
        return "dropped content from the locked reply"
    if not length_ratio_ok(candidate, locked):
        return "added content beyond a rephrase"
    return None


# Condition descriptions a user should hear, keyed by the world key involved. The
# planner speaks in terms of keys; a person hears the policy in their own terms.
_CONDITION_PHRASES = (
    ("amount_within_limit", "that amount is over the limit"),
    ("state_has_amount", "I don't have an amount yet"),
    ("state_has_recipient", "I don't know who to send it to"),
    ("search_results", "there's nothing on screen to play yet"),
    ("current_track", "nothing is playing at the moment"),
)


def speakable_reason(reason: str | None) -> str | None:
    """Translate planner vocabulary into a phrase a person would understand.

    Returns None when nothing meaningful can be said, so callers fall back to a generic
    sentence rather than reading out an internal key.
    """
    if not reason:
        return None
    for key, phrase in _CONDITION_PHRASES:
        if key in reason:
            return phrase
    return None


class TemplateReplyComposer:
    """Deterministic phrasing. The default: no model, no dependencies, no fabrication."""

    name = "template"

    _SLOT_LABELS = {
        "query": "what you'd like me to use", "amount": "the amount",
        "recipient": "who to send it to", "currency": "the currency",
        "target": "where to go", "account": "which account",
    }

    async def compose(self, context: ReplyContext) -> str:
        if context.situation is TurnSituation.COMPLETED:
            return self._completed(context)
        if context.situation is TurnSituation.SPECULATED:
            return "Just a moment - let me hear the rest."
        if context.situation is TurnSituation.NO_CAPABILITY:
            return "I'm not able to help with that one."
        if context.situation is TurnSituation.MISSING_ARGUMENT:
            return self._missing(context)
        if context.situation is TurnSituation.INFEASIBLE:
            return self._infeasible(context)
        if context.situation is TurnSituation.FAILED:
            return self._failed(context)
        if context.situation is TurnSituation.DISCARDED:
            return "Let me start that again."
        return "I couldn't work out what you wanted - could you say that another way?"

    def _completed(self, context: ReplyContext) -> str:
        if context.executed:
            return "Done."
        return "That's already the case."

    def _missing(self, context: ReplyContext) -> str:
        labels = [self._SLOT_LABELS.get(name, name.replace("_", " ")) for name in context.missing_slots]
        if len(labels) == 1:
            return f"I still need {labels[0]}."
        if labels:
            return f"I still need {' and '.join(labels[:2])}."
        return "I need a bit more detail for that."

    def _infeasible(self, context: ReplyContext) -> str:
        detail = speakable_reason(context.reason)
        if detail:
            return f"I can't do that - {detail}."
        return "I can't do that right now."

    def _failed(self, context: ReplyContext) -> str:
        return "Something went wrong before I could finish that. Nothing was changed."


class ReplyComposer(Protocol):
    """Turns deterministic facts into one user-facing sentence."""

    name: str

    async def compose(self, context: ReplyContext) -> str | None:
        """Return the sentence to speak, or None to say nothing at all."""
        ...


class SilentReplyComposer:
    """Says nothing. Selected by `AGENTIC_REPLY_COMPOSER=none`.

    Distinct from the template composer: the turn still runs to completion and the
    trace still records the outcome, but no user-facing sentence is produced. Useful
    when an outer system speaks for itself.
    """

    name = "none"
    last_source = "unused"

    async def compose(self, context: ReplyContext) -> str | None:
        return None


class LocalLLMReplyComposer:
    """Rephrases a reply with a small local instruct model. Opt-in (see `reply_config`).

    This backend is deliberately a *rephraser*, not a reasoner. Measured behaviour of a
    small model given the raw facts was that it invented plausible-but-wrong causes: a
    missing argument became "the relevant capability is not available", and exceeding a
    transfer limit became "exceeds the available funds". Those are wrong in ways a user
    cannot detect.

    So the semantic content is computed **first** by `TemplateReplyComposer`, and the
    model is shown that sentence with an explicit instruction to reword it without
    adding, removing, or changing any fact. It cannot invent a cause because it is never
    asked for one. On top of that:

    * the situation is decided deterministically before the model is involved;
    * the output is rejected if it claims an unearned completion, drifts into
      third-person narration, or adds a question the locked sentence did not ask;
    * any rejection or failure falls back to the deterministic sentence.

    A small model is never the last line of defence.
    """

    def __init__(self, model_name: str, *, device: str = "auto", max_new_tokens: int = 64,
                 fallback: ReplyComposer | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.fallback = fallback or TemplateReplyComposer()
        self.name = f"llm:{model_name}"
        self._tokenizer = None
        self._model = None
        # Which backend produced the last reply: "llm", or "fallback:<reason>". Callers
        # report this, so a silent total model failure cannot masquerade as success.
        self.last_source = "unused"
        self.last_problem: str | None = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype="auto")
            self._model = self._model.to(self._resolve_device()).eval()
            emit(logger, "reply_model_loaded", model=self.model_name,
                 device=str(next(self._model.parameters()).device))
        return self._tokenizer, self._model

    async def compose(self, context: ReplyContext) -> str:
        # The deterministic sentence is the source of truth for *what is said*. The model
        # only changes how it is said, so a wrong cause cannot be introduced.
        locked = await self.fallback.compose(context)
        try:
            candidate = self._generate(context, locked)
        except Exception as error:  # a model failure must never break the turn
            self.last_source = f"fallback:model_error:{type(error).__name__}"
            self.last_problem = type(error).__name__
            emit(logger, "reply_model_failed", model=self.model_name,
                 error_type=type(error).__name__, error=str(error)[:200])
            return locked
        problem = rephrase_is_acceptable(candidate, locked, context.situation)
        if problem:
            self.last_source = f"fallback:rejected:{problem}"
            self.last_problem = problem
            emit(logger, "reply_rejected", model=self.model_name, situation=context.situation,
                 problem=problem, candidate=candidate, locked=locked)
            return locked
        self.last_source = "llm"
        self.last_problem = None
        emit(logger, "reply_rephrased", model=self.model_name, situation=context.situation,
             locked=locked, reply=candidate)
        return candidate

    def health(self) -> tuple[bool, str]:
        """Whether the model backend is usable, loading it if needed.

        Surfaced so an operator can distinguish "the LLM is rephrasing" from "the LLM is
        broken and the template is covering for it" - the two look identical in output.
        """
        try:
            self._load()
        except Exception as error:
            return False, f"{type(error).__name__}: {str(error)[:200]}"
        return True, "ok"

    # Prompt shape is measured, not assumed. A long system-style instruction made a 270M
    # model emit an empty string for one of the locked sentences, while a short
    # single-turn instruction produced a faithful rephrase. Gemma also has no system role
    # in its chat template, so a system message is flattened into the user turn with no
    # role separation anyway. One short user turn is what works across both backends.
    instruction = "Rephrase this for a voice assistant (same meaning, one sentence):"

    def _generate(self, context: ReplyContext, locked: str) -> str:
        tokenizer, model = self._load()
        import torch

        user = f"{self.instruction} {locked}"
        messages = [{"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            # A thinking model emits its deliberation into the output unless this is
            # suppressed; the trace is verbose and unusable as a spoken reply.
            text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:  # models without a thinking switch
            text = tokenizer.apply_chat_template(messages, **kwargs)
        encodings = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**encodings, max_new_tokens=self.max_new_tokens,
                                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(generated[0][encodings["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip().strip('"')


def build_reply_composer(*, kind: str, model_name: str, device: str) -> ReplyComposer:
    """Composer factory. `template` is the default and needs no model.

    `none` selects the silent composer: it is not an alias for `template`.
    """
    if kind in {"", "template"}:
        return TemplateReplyComposer()
    if kind in {"none", "off", "disabled"}:
        return SilentReplyComposer()
    if kind in {"llm", "local-llm", "local_llm"}:
        return LocalLLMReplyComposer(model_name, device=device)
    raise ValueError(f"unknown reply composer kind '{kind}'")

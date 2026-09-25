"""Open-ended argument extraction, separate from bounded Jev choices.

Jev only returns bounded answers (Choice, Score, Noul). Arbitrary values such as
"Jazz", "Miles Davis", "report-final.pdf", or a web query cannot be enumerated as
choices, so they are recovered from the transcript by an extractive component.

Contract: an extracted answer must be a literal span of the transcript
(`start`/`end` offsets), and an explicit no-answer outcome is always allowed.
Partial transcripts must not fabricate a required argument.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ping_ponder.observability import emit

logger = logging.getLogger(__name__)

_MINILM_SQUAD2 = "deepset/minilm-uncased-squad2"
_NUEXTRACT_TINY = "numind/NuExtract-1.5-tiny"


class NoAnswer(ValueError):
    """No extractable span met the confidence threshold for this slot."""


@dataclass(frozen=True)
class ExtractedSpan:
    """An answer recovered from the transcript, with its exact character offsets."""

    slot: str
    question: str
    text: str
    start: int
    end: int
    confidence: float
    extractor: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("extracted span offsets must be a non-empty forward range")
        if len(self.text) != self.end - self.start:
            raise ValueError("extracted span text length must match its offsets")

    def verify(self, utterance: str) -> "ExtractedSpan":
        """Confirm the span really indexes this utterance; guards against drift."""
        if utterance[self.start:self.end] != self.text:
            raise NoAnswer(f"span {self.start}:{self.end} does not match the current transcript")
        return self


@runtime_checkable
class ArgumentExtractor(Protocol):
    """Recovers one open-ended argument from an utterance for a given question."""

    name: str

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan: ...


@dataclass(frozen=True)
class SlotRequest:
    """One slot the goal builder wants filled from a single utterance."""

    slot: str
    question: str
    confidence_threshold: float = 0.0
    required: bool = True
    kind: str = "span"
    goal_type: str | None = None


@runtime_checkable
class MultiSlotExtractor(Protocol):
    """Optional capability: fill several slots from one utterance in a single pass.

    Single-slot extraction costs one model call per slot, so a three-slot goal is
    three sequential calls on the hot path. A generative extractor can fill a whole
    JSON template at once. Supporting both keeps that optimization available without
    forcing every backend to implement it, and without the goal builder hard-coding
    which implementation it is talking to.
    """

    name: str

    async def extract_many(self, utterance: str, requests: tuple[SlotRequest, ...]
                           ) -> Mapping[str, ExtractedSpan]: ...


class ExtractiveQAExtractor:
    """Extractive QA over a transcript via a HuggingFace question-answering model.

    The default model is `deepset/minilm-uncased-squad2`. The heavy dependencies
    (`torch`, `transformers`) are imported lazily; construction without them raises
    `ExtractorUnavailable` so a caller can fall back instead of failing at import.

    SQuAD-trained readers expect a narrative passage, not a bare imperative. A
    transcript such as `"Play Miles Davis"` is out of distribution on its own and
    measures as no-answer, while `"The user said: Play Miles Davis"` yields
    `"Miles Davis"` at high confidence. `narrative_frame` therefore wraps the
    transcript for the model only, and the returned offsets are mapped back to the
    raw transcript before the answer is accepted, so callers never see the frame.
    """

    narrative_frame = "The user said: "

    def __init__(self, model_name: str = _MINILM_SQUAD2, *, device: int = -1,
                 narrative_frame: str | None = None) -> None:
        self.name = model_name
        self.model_name = model_name
        self.device = device
        if narrative_frame is not None:
            self.narrative_frame = narrative_frame
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as error:  # pragma: no cover - depends on optional extra
                raise ExtractorUnavailable(
                    "install the 'extraction' extra (torch + transformers) for ExtractiveQAExtractor"
                ) from error
            self._pipeline = pipeline("question-answering", model=self.model_name, device=self.device)
        return self._pipeline

    def available(self) -> bool:
        try:
            self._load()
        except ExtractorUnavailable:
            return False
        return True

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan:
        return self._extract_sync(utterance, question, slot=slot, confidence_threshold=confidence_threshold)

    def _extract_sync(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float) -> ExtractedSpan:
        if not utterance.strip():
            raise NoAnswer(f"slot '{slot}': no transcript to extract from")
        pipeline = self._load()
        frame = self.narrative_frame or ""
        context = f"{frame}{utterance}"
        output = pipeline(question=question, context=context, handle_impossible_answer=True)
        score = float(output.get("score", 0.0))
        if score < confidence_threshold:
            emit(logger, "span_no_answer", slot=slot, extractor=self.name, confidence=score,
                 threshold=confidence_threshold)
            raise NoAnswer(f"slot '{slot}': no span met confidence {confidence_threshold}")
        # Map the model's context offsets back onto the raw transcript. An answer that
        # falls inside the frame (or outside the transcript) is rejected, never
        # rewritten into something the user did not say.
        start = int(output.get("start", 0)) - len(frame)
        end = int(output.get("end", 0)) - len(frame)
        if start < 0 or end > len(utterance) or end <= start:
            emit(logger, "span_no_answer", slot=slot, extractor=self.name, confidence=score,
                 reason="answer_not_in_transcript")
            raise NoAnswer(f"slot '{slot}': extracted span is not inside the transcript")
        raw = utterance[start:end]
        text = raw.strip()
        if not text:
            raise NoAnswer(f"slot '{slot}': extracted span is blank")
        start += raw.index(text)
        end = start + len(text)
        span = ExtractedSpan(slot=slot, question=question, text=text, start=start, end=end,
                             confidence=score, extractor=self.name)
        emit(logger, "span_extracted", slot=slot, extractor=self.name, text=text,
             start=start, end=end, confidence=score)
        return span


class ExtractorUnavailable(RuntimeError):
    """The requested extraction backend is not installed or cannot be loaded."""


class HeuristicSpanExtractor:
    """Dependency-free fallback: strips conversational lead-ins and returns the remainder.

    It is deliberately conservative. It never invents a value; when the utterance
    has no residue after removing the framing, it reports no answer. Lead-ins are
    removed iteratively (longest meaningful match first) rather than by one
    backtracking regex, so an optional group can never swallow the whole value.
    """

    name = "heuristic"

    _LEAD_INS = (
        r"(?:please|kindly)\b\s*",
        r"(?:can|could|would)\s+you\b\s*",
        r"(?:make|set|change|put)\s+the\s+\w+\s+(?:say|to|read)\b\s*",
        r"(?:search|google)\s+(?:(?:the\s+web|google)\s+|\w+\s+)?(?:for|up)\b\s*",
        r"look\s+up\b\s*",
        r"(?:go\s+to|navigate(?:\s+to)?|visit)\b\s*",
        r"(?:play|put\s+on)\b\s*(?:me\b\s*)?(?:some\b\s*)?",
        r"i(?:'d|\s+would)\s+like\s+to\s+(?:hear|listen\s+to)\b\s*",
        r"i\s+want\s+to\s+(?:hear|listen\s+to)\b\s*",
        r"open\s+spotify\b\s*(?:and\b\s*)?",
        r"(?:transfer|send|move)\s+(?:\$?\d+(?:[.,]\d+)?\s*[km]?\s*(?:usd|sgd|eur|gbp|jpy|myr|aud|hkd|cny)?)\s+to\b\s*",
        r"(?:transfer|send)(?:\s+money)?\s+to\b\s*",
        r"pay\b\s*",
        r"spotify\b\s*",
        r"(?:and|then|now)\b\s*",
    )
    _NUMERIC_QUESTIONS = re.compile(
        r"\bhow\s+(?:much|many)\b|\bamount\b|\bnumber\b|\bhow\s+much\s+money\b",
        re.IGNORECASE,
    )
    _NO_CONTENT = frozenset({
        "please", "kindly", "some", "some music", "music", "it", "that", "this",
        "spotify", "and", "then", "now", "for me", "me", "a", "the", "money",
    })
    _TRAILING = (
        r"\s+(?:please|for\s+me|now|thanks)\b",
        # A trailing *numeric* clause belongs to another slot ("for 500"), not this one.
        # A word clause is usually part of the value itself ("music for running"), so it
        # is deliberately not stripped.
        r"\s+(?:for|in|amount|worth)\s+\$?\d[\d.,]*\s*[km]?$",
        r"[,.;!?]+",
    )

    def __init__(self) -> None:
        self._lead_ins = tuple(re.compile(r"^" + pattern, re.IGNORECASE) for pattern in self._LEAD_INS)

    def _strip_framing(self, utterance: str) -> tuple[str, int]:
        value = utterance.strip()
        offset = utterance.index(value)
        changed = True
        while changed and value:
            changed = False
            for pattern in self._lead_ins:
                match = pattern.match(value)
                if match and match.end() > 0:
                    offset += match.end()
                    value = value[match.end():].lstrip()
                    offset = utterance.index(value, offset) if value else len(utterance)
                    changed = True
                    break
        changed = True
        while changed and value:
            changed = False
            for pattern in self._TRAILING:
                stripped = re.sub(pattern + r"$", "", value, flags=re.IGNORECASE).strip()
                if stripped != value:
                    value, changed = stripped, True
        return value, offset

    def _numeric_span(self, utterance: str, question: str, *, slot: str) -> ExtractedSpan | None:
        """When the question asks for a quantity, return the number, not the residue."""
        found = re.search(r"(?<![\w.])\$?(\d+(?:[.,]\d+)?)\s*(k|m)?(?![\w.])",
                          utterance, re.IGNORECASE)
        if found is None:
            return None
        start, end = found.start(), found.end()
        text = utterance[start:end].strip()
        if not text:
            return None
        return ExtractedSpan(slot=slot, question=question, text=text, start=start, end=end,
                             confidence=0.7, extractor=self.name)

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan:
        if not utterance.strip():
            raise NoAnswer(f"slot '{slot}': empty transcript")
        if self._NUMERIC_QUESTIONS.search(question):
            # A quantity question must yield a number or nothing at all; falling
            # through to the residue would answer "Sarah" to "how much money?".
            numeric = self._numeric_span(utterance, question, slot=slot)
            if numeric is None:
                raise NoAnswer(f"slot '{slot}': no numeric quantity in the transcript")
            return numeric
        value, start = self._strip_framing(utterance)
        if not value or value.casefold() in self._NO_CONTENT:
            raise NoAnswer(f"slot '{slot}': no value remains after removing conversational framing")
        end = start + len(value)
        if utterance[start:end] != value:
            raise NoAnswer(f"slot '{slot}': framing strip did not produce a real transcript span")
        # Confidence is a property of the extraction, not of the caller's wish; a
        # heuristic strip is only certain it found *a* value, not the right one, so
        # it reports the configured floor and honours an explicit threshold above it.
        confidence = min(1.0, max(0.5, confidence_threshold))
        if confidence < confidence_threshold:
            raise NoAnswer(f"slot '{slot}': heuristic confidence {confidence} is below {confidence_threshold}")
        emit(logger, "span_extracted", slot=slot, extractor=self.name, text=value,
             start=start, end=end, confidence=confidence)
        return ExtractedSpan(slot=slot, question=question, text=value, start=start, end=end,
                             confidence=confidence, extractor=self.name)


def coerce_number(text: str) -> float | None:
    """Type an already-extracted span as a number, or report that it is not one.

    This is deliberately confined to the span the extractor returned; it never scans
    the transcript for digits, so it cannot invent an amount the user did not say.
    Handles digits, thousands separators, and simple number words.
    """
    cleaned = text.strip().lower().replace(",", "").replace("$", "").strip()
    if not cleaned:
        return None
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20, "thirty": 30,
             "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
             "ninety": 90, "hundred": 100, "thousand": 1000}
    if cleaned in words:
        return float(words[cleaned])
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|m)?", cleaned)
    if not match:
        # Also accept a number embedded in a short span such as "about 500".
        embedded = re.search(r"(\d+(?:\.\d+)?)", cleaned)
        if not embedded:
            return None
        cleaned = embedded.group(1)
        match = re.fullmatch(r"(\d+(?:\.\d+)?)", cleaned)
        if not match:
            return None
    value = float(match.group(1))
    suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else None
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    return value


def locate_span(utterance: str, value: str, *, offset: int = 0, search_from: int = 0) -> tuple[int, int] | None:
    """Find `value` as a literal span of `utterance`, or report that it is absent.

    Extractive answers carry offsets. A generative extractor returns text only, so
    its output has to be located in the transcript before it can be trusted. A value
    that cannot be located verbatim is not a transcript span and is rejected rather
    than repaired: normalizing or fuzzy-matching would let the extractor put words in
    the user's mouth.
    """
    cleaned = value.strip()
    if not cleaned:
        return None
    index = utterance.find(cleaned, search_from)
    if index < 0:
        return None
    return index, index + len(cleaned)


class LunaStructuredSlotExtractor:
    """Fill only the slot names supplied by the selected GoalSchema.

    Responses are schema-constrained JSON. Every nonempty value must still be a
    literal transcript span, so GoalBuilder remains responsible for final validation
    and Luna cannot add words the user did not say.
    """

    response_model_name = "LunaSlotValues"
    retry_failed_batch_as_single_slot = True
    system_prompt = (
        "Extract values for only the declared slots. Return each value as an exact, "
        "contiguous substring of the utterance, preserving qualifiers and important "
        "modifying phrases. Do not paraphrase, infer a site, add constraints, or "
        "change the selected goal type. Return null when the utterance does not "
        "supply a value. The structured response schema is authoritative."
    )

    def __init__(self, provider, *, model: str) -> None:
        if not model:
            raise ValueError("a structured slot extraction model is required")
        self.name = f"luna-slots:{model}"
        self.model = model
        self.provider = provider

    async def extract_many(self, utterance: str, requests: tuple[SlotRequest, ...]
                           ) -> Mapping[str, ExtractedSpan]:
        if not requests or not utterance.strip():
            return {}
        from pydantic import ConfigDict, create_model

        response_model = create_model(
            self.response_model_name,
            __config__=ConfigDict(extra="forbid"),
            **{request.slot: (str | None, ...) for request in requests},
        )
        # Required nullable fields are accepted by strict structured-output APIs and
        # distinguish an explicit no-answer from an omitted/unknown key.
        response_model.__name__ = self.response_model_name
        prompt = {
            "goal_type": requests[0].goal_type,
            "utterance": utterance,
            "declared_slots": [
                {"name": request.slot, "question": request.question,
                 "required": request.required, "kind": request.kind}
                for request in requests
            ],
        }
        response = await self.provider.infer(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_model=response_model,
        )
        values = response.value.model_dump()
        spans: dict[str, ExtractedSpan] = {}
        by_name = {request.slot: request for request in requests}
        for name, value in values.items():
            request = by_name[name]
            if not isinstance(value, str) or not value.strip():
                continue
            located = locate_span(utterance, value)
            if located is None:
                emit(logger, "span_no_answer", slot=name, extractor=self.name,
                     value=value, reason="answer_not_a_transcript_span")
                continue
            start, end = located
            text = utterance[start:end]
            span = ExtractedSpan(slot=name, question=request.question, text=text,
                                 start=start, end=end, confidence=0.90, extractor=self.name)
            if span.confidence >= request.confidence_threshold:
                spans[name] = span
        return spans

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan:
        values = await self.extract_many(utterance, (SlotRequest(slot, question, confidence_threshold),))
        if slot not in values:
            raise NoAnswer(f"slot '{slot}': {self.name} returned no usable transcript span")
        return values[slot]


class LlamaStructuredSlotExtractor(LunaStructuredSlotExtractor):
    """OpenRouter Llama extractor constrained to the selected GoalSchema slots.

    It shares Luna's strict JSON schema, transcript-span check, and declared-slot
    whitelist. GoalBuilder remains the final validator and owner of the goal type.
    """

    response_model_name = "LlamaSlotValues"
    system_prompt = (
        "Extract values for only the declared slots. Copy each value verbatim from the "
        "utterance as one consecutive substring: never summarize, paraphrase, or omit "
        "words between relevant terms. For target slots, preserve the whole requested "
        "item and every qualifier, including intervening words. For example, from "
        "'Find the section of the CSS guide explaining grid areas', copy "
        "'the section of the CSS guide explaining grid areas', not "
        "'the section explaining grid areas'. For a site slot, return only the explicit "
        "website name, without nearby page-description words; return null if no site is "
        "stated. Never add or infer information or change the selected goal type. If a "
        "value cannot be copied exactly, return null. The structured response schema "
        "is authoritative."
    )

    def __init__(self, provider, *, model: str) -> None:
        super().__init__(provider, model=model)
        self.name = f"llama-slots:{model}"


class NuExtractExtractor:
    """Structured extraction with `numind/NuExtract-1.5-tiny`.

    NuExtract is a small *generative* extraction model, not an extractive QA reader.
    It is prompted with a JSON template and returns a filled JSON object, which has
    two consequences this adapter must handle:

    * **It has no answer probability.** A SQuAD reader reports a confidence score; a
      generative model does not. `confidence` here is therefore *derived from
      verification strength* (the value was located verbatim and actually narrowed the
      utterance), not a calibrated probability. It is documented as such rather than
      passed off as a model score.
    * **It can return a non-span.** Measured behaviour includes paraphrasing and, when
      it cannot find the argument, echoing the whole utterance back (`"pause it"` ->
      `"pause it"`). Every answer is therefore located in the transcript with
      `locate_span`, and a full-utterance echo is rejected as "did not narrow".

    Its advantage over the QA reader is multi-field extraction in a single pass and
    better handling of compound values (`"play the album Kind of Blue"` -> `"Kind of
    Blue"`, where the QA reader returns `"Blue"`). Its cost is latency: roughly
    0.7-1.2s on CPU against roughly 0.01-0.04s for the QA reader, so it suits
    lower-frequency or offline use rather than the hot partial-speech path.
    """

    # Values that are not arguments: an answer equal to the whole utterance means the
    # model did not identify anything, so it must not become a goal argument.
    def __init__(self, model_name: str = _NUEXTRACT_TINY, *, max_new_tokens: int = 200,
                 reject_full_echo: bool = True, device: str = "auto") -> None:
        self.name = model_name
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.reject_full_echo = reject_full_echo
        # "auto" uses CUDA when the build can see a GPU and otherwise CPU, so the
        # default keeps working on a CPU-only machine and in tests.
        self.device = device
        self._tokenizer = None
        self._model = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        if self._model is None or self._tokenizer is None:
            started = time.monotonic()
            emit(logger, "nuextract_load_start", model=self.model_name, requested_device=self.device)
            try:
                try:
                    import torch  # noqa: F401  (imported for the availability check)
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                except ImportError as error:  # pragma: no cover - optional extra
                    raise ExtractorUnavailable(
                        "install the 'extraction' extra (torch + transformers) for NuExtractExtractor"
                    ) from error
                # NuExtract ships a custom module, so trust_remote_code is required. That
                # executes code from the model repository: pin the revision in any
                # deployment that handles untrusted input.
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto", trust_remote_code=True)
                self._model = model.to(self._resolve_device()).eval()
            except Exception as error:
                emit(logger, "nuextract_load_failed", model=self.model_name,
                     requested_device=self.device, error_type=type(error).__name__,
                     error=str(error), latency_seconds=time.monotonic() - started)
                raise
            emit(logger, "nuextract_device", model=self.model_name,
                 device=str(next(self._model.parameters()).device))
            emit(logger, "nuextract_load_complete", model=self.model_name,
                 latency_seconds=time.monotonic() - started)
        return self._tokenizer, self._model

    def available(self) -> bool:
        try:
            self._load()
        except ExtractorUnavailable:
            return False
        return True

    async def extract_many(self, utterance: str, requests: tuple[SlotRequest, ...]
                           ) -> Mapping[str, ExtractedSpan]:
        """Fill every requested slot in one generation.

        One template with all slot names is a single forward pass, which is the
        efficiency NuExtract offers over per-slot QA readers. Slots it cannot fill are
        simply absent from the result, so a caller sees "unanswered" rather than a
        fabricated value.
        """
        if not requests:
            return {}
        if not utterance.strip():
            return {}
        tokenizer, model = self._load()
        # NuExtract reads the JSON *key* as the field description, not the slot name.
        # Prompting with bare keys ("query") measurably made it echo the whole utterance
        # back ("play some jazz" -> "play some jazz"), which the anti-echo guard then
        # rejected as "did not narrow" - so the extractor appeared to find nothing at
        # all. The slot's own extraction question is already the right description, and
        # keys are mapped back to slot names below.
        fields = {request.question: "" for request in requests}
        by_question = {request.question: request.slot for request in requests}
        values = self._generate_fields(tokenizer, model, utterance, fields)
        spans: dict[str, ExtractedSpan] = {}
        for request in requests:
            value = values.get(request.question)
            if value is None:
                # Tolerate a model that echoes the key verbatim instead of the question.
                for key, candidate in values.items():
                    if by_question.get(key) == request.slot:
                        value = candidate
                        break
            span = self._verify(utterance, value, request.slot, request.question)
            if span is not None and span.confidence >= request.confidence_threshold:
                spans[request.slot] = span
        return spans

    def _verify(self, utterance: str, value: str | None, slot: str,
                question: str) -> ExtractedSpan | None:
        """Locate a generated value in the transcript; never repair a non-span."""
        if not value:
            return None
        located = locate_span(utterance, value)
        if located is None:
            emit(logger, "span_no_answer", slot=slot, extractor=self.name, value=value,
                 reason="answer_not_a_transcript_span")
            return None
        start, end = located
        text = utterance[start:end]
        if self.reject_full_echo and text == utterance.strip():
            emit(logger, "span_no_answer", slot=slot, extractor=self.name, value=value,
                 reason="did_not_narrow")
            return None
        return ExtractedSpan(slot=slot, question=question, text=text, start=start, end=end,
                             confidence=0.90, extractor=self.name)

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan:
        if not utterance.strip():
            raise NoAnswer(f"slot '{slot}': no transcript to extract from")
        tokenizer, model = self._load()
        value = self._generate(tokenizer, model, utterance, question)
        if not value:
            emit(logger, "span_no_answer", slot=slot, extractor=self.name, reason="empty_template_field")
            raise NoAnswer(f"slot '{slot}': NuExtract returned no value")
        span = self._verify(utterance, value, slot, question)
        if span is None:
            raise NoAnswer(f"slot '{slot}': NuExtract value {value!r} is not a usable transcript span")
        if span.confidence < confidence_threshold:
            raise NoAnswer(f"slot '{slot}': confidence {span.confidence} below {confidence_threshold}")
        emit(logger, "span_extracted", slot=span.slot, extractor=self.name, text=span.text,
             start=span.start, end=span.end, confidence=span.confidence)
        return span

    def _generate(self, tokenizer, model, utterance: str, field: str) -> str | None:
        """One field, keyed by its extraction question (see `extract_many`)."""
        return self._generate_fields(tokenizer, model, utterance, {field: ""}).get(field)

    def _generate_fields(self, tokenizer, model, utterance: str,
                         fields: Mapping[str, str]) -> dict[str, str]:
        """Fill a JSON template in one generation and return its non-empty string values."""
        import json

        import torch

        started = time.monotonic()
        emit(logger, "nuextract_generation_start", model=self.model_name,
             device=str(model.device), fields=list(fields),
             utterance_characters=len(utterance))
        template = json.dumps(dict(fields), indent=4)
        prompt = f"<|input|>\n### Template:\n{template}\n### Text:\n{utterance}\n\n<|output|>"
        # The model may live on CUDA while the tokenizer always returns CPU tensors, so
        # the batch is moved to the model's own device. Skipping this made `device="auto"`
        # silently return nothing on a GPU machine: the inputs stayed on CPU and
        # generation ran against a model on CUDA.
        try:
            encodings = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(**encodings, max_new_tokens=self.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as error:
            emit(logger, "nuextract_generation_failed", model=self.model_name,
                 device=str(model.device), error_type=type(error).__name__,
                 error=str(error), latency_seconds=time.monotonic() - started)
            raise
        emit(logger, "nuextract_generation_complete", model=self.model_name,
             device=str(model.device), latency_seconds=time.monotonic() - started)
        if "<|output|>" not in decoded:
            return {}
        body = decoded.split("<|output|>")[1].strip()
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            emit(logger, "span_extraction_error", extractor=self.name, reason="unparseable_json")
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(key): value.strip() for key, value in parsed.items()
                if isinstance(value, str) and value.strip()}


class ChainExtractor:
    """Tries extractors in order; the first span clearing its floor wins.

    Each extractor has its own confidence floor. A reader can answer with a *short,
    low-confidence* fragment ("PersonaPlex" for "PersonaPlex turn detection") that is
    technically above the caller's threshold; the floor lets a fuller fallback answer
    win instead. Note this is a secondary guard: the primary extraction quality lever
    is the slot question and the narrative frame, not the fallback.

    `floors` aligns positionally with `extractors`; a missing entry means "use the
    caller's threshold". The effective floor for an extractor is the higher of its
    chain floor and the caller's threshold, so a caller can still tighten extraction.
    """

    def __init__(self, extractors: tuple[ArgumentExtractor, ...],
                 *, floors: tuple[float, ...] = (0.5,)) -> None:
        if not extractors:
            raise ValueError("ChainExtractor requires at least one extractor")
        self.extractors = extractors
        self.floors = tuple(floors) + (0.0,) * max(0, len(extractors) - len(floors))
        self.name = "chain:" + ",".join(item.name for item in extractors)

    async def extract(self, utterance: str, question: str, *, slot: str,
                      confidence_threshold: float = 0.0) -> ExtractedSpan:
        errors: list[str] = []
        for index, extractor in enumerate(self.extractors):
            floor = max(self.floors[index] if index < len(self.floors) else 0.0, confidence_threshold)
            try:
                span = await extractor.extract(utterance, question, slot=slot,
                                               confidence_threshold=floor)
            except (NoAnswer, ExtractorUnavailable) as error:
                errors.append(f"{extractor.name}: {error}")
                continue
            if span.confidence < floor:
                errors.append(f"{extractor.name}: confidence {span.confidence} below chain floor {floor}")
                continue
            return span
        emit(logger, "span_extraction_failed", slot=slot, extractor=self.name, attempts=errors)
        raise NoAnswer(f"slot '{slot}': no extractor produced an answer ({'; '.join(errors)})")

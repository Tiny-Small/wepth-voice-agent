"""Goal builder: turn a Local Jev goal type plus extracted spans into a SemanticGoal.

The builder owns slot filling. It reads the capability's `GoalSchema`, asks the
configured `ArgumentExtractor` for each declared slot, and refuses to fabricate a
required argument. An extractive answer must be a literal transcript span; a
missing required slot yields an incomplete goal (never a guessed value).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ping_ponder.observability import emit

from .goals import GoalSchema, SemanticGoal, SlotKind, SlotSpec
from .span import (ArgumentExtractor, ExtractedSpan, MultiSlotExtractor, NoAnswer, SlotRequest,
                   coerce_number)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionRecord:
    """Audit trail for one slot attempt, including explicit no-answer outcomes."""

    slot: str
    question: str
    filled: bool
    text: str | None = None
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    extractor: str | None = None
    reason: str | None = None

    @property
    def unanswered(self) -> bool:
        return not self.filled


@dataclass(frozen=True)
class GoalBuild:
    """A built goal plus the slot audit and any still-missing required slots."""

    goal: SemanticGoal
    extractions: tuple[ExtractionRecord, ...] = ()
    missing_slots: tuple[str, ...] = ()
    rejected: Mapping[str, str] = field(default_factory=dict)
    extraction_latency_s: float = 0.0
    # GoalBuilder's own work, excluding time spent awaiting the extractor.
    goal_builder_latency_s: float = 0.0
    goal_build_wall_latency_s: float = 0.0

    @property
    def complete(self) -> bool:
        return not self.missing_slots

    def extraction(self, slot: str) -> ExtractionRecord | None:
        for record in self.extractions:
            if record.slot == slot:
                return record
        return None

    def promoted(self) -> "GoalBuild":
        """Promote partial no-answer diagnostics without repeating extraction."""
        extractions = tuple(
            replace(record, reason="no_answer")
            if record.reason == "partial_no_answer" else record
            for record in self.extractions
        )
        rejected = {
            slot: "no_answer" if reason == "partial_no_answer" else reason
            for slot, reason in self.rejected.items()
        }
        return replace(self, extractions=extractions, rejected=rejected)


class GoalBuilder:
    """Fills open-ended goal arguments from the transcript with span extraction."""

    def __init__(self, extractor: ArgumentExtractor, *, confidence_threshold: float = 0.30,
                 allow_incomplete: bool = True) -> None:
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.allow_incomplete = allow_incomplete

    async def build(self, *, capability: str, schema: GoalSchema, utterance: str,
                    initial_arguments: Mapping[str, Any] | None = None,
                    final: bool = True) -> GoalBuild:
        build_started = time.monotonic()
        if schema.slots:
            emit(logger, "goal_extraction_profile_selected", capability=capability,
                 goal_type=schema.goal_type,
                 slots=[{"name": slot.name, "required": slot.required} for slot in schema.slots])
        arguments: dict[str, Any] = dict(initial_arguments or {})
        records: list[ExtractionRecord] = []
        rejected: dict[str, str] = {}

        pending: list[SlotSpec] = []
        for slot in schema.slots:
            if arguments.get(slot.name) not in (None, ""):
                records.append(ExtractionRecord(slot=slot.name, question=slot.question, filled=True,
                                                text=str(arguments[slot.name]), extractor="provided"))
            else:
                pending.append(slot)

        # A multi-slot backend fills every pending slot in one pass; otherwise each slot
        # costs its own call. The goal builder does not care which it is talking to.
        extraction_started = time.monotonic()
        spans, batched, batch_failed = await self._extract_pending(
            pending, utterance, goal_type=schema.goal_type)
        extraction_latency = time.monotonic() - extraction_started
        for slot in pending:
            resolve_started = time.monotonic()
            record, value = await self._resolve(slot, utterance, spans.get(slot.name),
                                                batched=batched, final=final,
                                                goal_type=schema.goal_type,
                                                batch_failed=batch_failed)
            if not batched:
                extraction_latency += time.monotonic() - resolve_started
            records.append(record)
            if value is not None:
                arguments[slot.name] = value
            elif record.reason is not None:
                rejected[slot.name] = record.reason

        missing = schema.missing_slots(arguments)
        if missing and not self.allow_incomplete:
            raise IncompleteGoal(f"goal '{schema.goal_type}' is missing required slots: {sorted(missing)}")
        # A partial transcript must never manufacture a required argument.
        if missing and not final:
            emit(logger, "goal_builder_partial_incomplete", capability=capability,
                 goal_type=schema.goal_type, missing=list(missing))
        goal = SemanticGoal(capability=capability, goal_type=schema.goal_type, arguments=arguments)
        emit(logger, "semantic_goal_built", capability=capability, goal_type=schema.goal_type,
             arguments={key: str(value) for key, value in arguments.items()}, final=final,
             missing=list(missing))
        builder_wall_latency = time.monotonic() - build_started
        builder_latency = max(0.0, builder_wall_latency - extraction_latency)
        emit(logger, "goal_builder_end", capability=capability, goal_type=schema.goal_type,
             extraction_latency_seconds=extraction_latency,
             latency_seconds=builder_latency, wall_latency_seconds=builder_wall_latency)
        return GoalBuild(goal=goal, extractions=tuple(records), missing_slots=missing,
                         rejected=rejected, extraction_latency_s=extraction_latency,
                         goal_builder_latency_s=builder_latency,
                         goal_build_wall_latency_s=builder_wall_latency)


    async def _extract_pending(self, pending: list[SlotSpec], utterance: str, *, goal_type: str
                               ) -> tuple[dict[str, ExtractedSpan], bool, bool]:
        """One pass over all pending slots when the backend supports it.

        The booleans report whether the batch ran and whether it failed. When it ran
        successfully, a missing slot is a genuine "no answer" and must not be retried:
        one batched pass already considered every requested slot.
        """
        if not pending or not isinstance(self.extractor, MultiSlotExtractor):
            return {}, False, False
        requests = tuple(SlotRequest(slot=slot.name, question=slot.question,
                                     confidence_threshold=max(self.confidence_threshold,
                                                              slot.confidence_threshold),
                                     required=slot.required, kind=slot.kind.value,
                                     goal_type=goal_type)
                         for slot in pending)
        try:
            spans = dict(await self.extractor.extract_many(utterance, requests))
        except Exception as error:  # a batch failure falls back to per-slot extraction
            emit(logger, "multi_slot_extraction_failed", extractor=self.extractor.name,
                 error_type=type(error).__name__, error=str(error), recovery="per_slot")
            return {}, False, True
        emit(logger, "multi_slot_extraction", extractor=self.extractor.name,
             requested=[request.slot for request in requests], filled=sorted(spans))
        return spans, True, False

    async def _resolve(self, slot: SlotSpec, utterance: str, span: ExtractedSpan | None,
                       *, batched: bool, final: bool, goal_type: str,
                       batch_failed: bool
                       ) -> tuple[ExtractionRecord, str | None]:
        """Turn an already-extracted span (or its absence) into a slot record."""
        if span is None:
            if batched:
                # The batch pass covered this slot and returned nothing for it.
                return (ExtractionRecord(slot=slot.name, question=slot.question, filled=False,
                                         reason="no_answer" if final else "partial_no_answer"), None)
            record, value = await self._fill(slot, utterance, final=final,
                                             goal_type=goal_type, batch_failed=batch_failed)
            return record, value
        return await self._record(slot, utterance, span, final=final)

    async def _fill(self, slot: SlotSpec, utterance: str, *, final: bool,
                    goal_type: str, batch_failed: bool = False
                    ) -> tuple[ExtractionRecord, str | None]:
        threshold = max(self.confidence_threshold, slot.confidence_threshold)
        try:
            if (batch_failed and getattr(self.extractor, "retry_failed_batch_as_single_slot", False)
                    and isinstance(self.extractor, MultiSlotExtractor)):
                # A failed batch may be retried one slot at a time, but each retry
                # still receives the selected schema and that slot's requirements.
                requests = (SlotRequest(slot=slot.name, question=slot.question,
                                        confidence_threshold=threshold, required=slot.required,
                                        kind=slot.kind.value, goal_type=goal_type),)
                spans = await self.extractor.extract_many(utterance, requests)
                span = spans.get(slot.name)
                if span is None:
                    raise NoAnswer(f"slot '{slot.name}': extractor returned no usable transcript span")
            else:
                span = await self.extractor.extract(
                    utterance, slot.question, slot=slot.name, confidence_threshold=threshold)
            span.verify(utterance)
        except NoAnswer:
            # Partial transcripts are allowed to leave a required slot unanswered.
            return (ExtractionRecord(slot=slot.name, question=slot.question, filled=False,
                                     reason="no_answer" if final else "partial_no_answer"), None)
        except Exception as error:  # extractor backend failure must not fabricate a value
            emit(logger, "span_extraction_error", slot=slot.name, extractor=self.extractor.name,
                 error_type=type(error).__name__, error=str(error))
            return (ExtractionRecord(slot=slot.name, question=slot.question, filled=False,
                                     reason=f"error:{type(error).__name__}"), None)
        return await self._record(slot, utterance, span, final=final)

    async def _record(self, slot: SlotSpec, utterance: str, span: ExtractedSpan,
                      *, final: bool) -> tuple[ExtractionRecord, str | None]:
        """Validate a span against its slot and type it."""
        if not slot.accepts(span.text):
            emit(logger, "span_rejected_by_slot", slot=slot.name, text=span.text)
            return (ExtractionRecord(slot=slot.name, question=slot.question, filled=False,
                                     text=span.text, reason="rejected_by_slot",
                                     confidence=span.confidence, extractor=span.extractor), None)
        value: Any = span.text
        if slot.kind is SlotKind.NUMBER:
            # Coerce the extracted span; an untypeable span is a rejected slot, never
            # a value scraped from elsewhere in the transcript.
            typed = coerce_number(span.text)
            if typed is None:
                return (ExtractionRecord(slot=slot.name, question=slot.question, filled=False,
                                         text=span.text, reason="not_a_number",
                                         confidence=span.confidence, extractor=span.extractor), None)
            value = typed
        return (ExtractionRecord(slot=slot.name, question=slot.question, filled=True, text=span.text,
                                 start=span.start, end=span.end, confidence=span.confidence,
                                 extractor=span.extractor), value)


class IncompleteGoal(ValueError):
    """A required slot could not be filled and incomplete goals are not allowed."""

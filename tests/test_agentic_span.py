"""Argument extraction: real transcript spans, no-answer, thresholds, no fabrication."""

import pytest

from ping_ponder.agentic.goal_builder import GoalBuilder, IncompleteGoal
from ping_ponder.agentic.goals import GoalSchema, SlotSpec
from ping_ponder.agentic.span import (ChainExtractor, ExtractedSpan, ExtractiveQAExtractor,
                                      HeuristicSpanExtractor, NoAnswer)
from ping_ponder.agentic.world import Condition

PLAY = GoalSchema("PLAY", slots=(SlotSpec("query", "What should be played?"),),
                  satisfied_when=(Condition("media.playing"),))


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance,expected", [
    ("Play some Jazz", "Jazz"),
    ("Please play some Jazz for me", "Jazz"),
    ("Make the title say hello", "hello"),
    ("Search for PersonaPlex turn detection", "PersonaPlex turn detection"),
    ("Search Google for Miles Davis", "Miles Davis"),
    ("Open Spotify and play Jazz", "Jazz"),
    ("Go to Youtube", "Youtube"),
])
async def test_heuristic_extractor_returns_literal_transcript_spans(utterance, expected):
    span = await HeuristicSpanExtractor().extract(utterance, "What should be played?", slot="query")
    assert span.text == expected
    assert utterance[span.start:span.end] == expected
    assert span.verify(utterance) is span


@pytest.mark.asyncio
@pytest.mark.parametrize("utterance", ["play some", "Please", "put on some music", "open spotify", ""])
async def test_heuristic_extractor_reports_no_answer_instead_of_inventing(utterance):
    with pytest.raises(NoAnswer):
        await HeuristicSpanExtractor().extract(utterance, "What should be played?", slot="query")


def test_extracted_span_rejects_inconsistent_offsets():
    with pytest.raises(ValueError):
        ExtractedSpan(slot="q", question="q", text="jazz", start=5, end=4, confidence=1.0, extractor="t")
    with pytest.raises(ValueError):
        ExtractedSpan(slot="q", question="q", text="jazz", start=0, end=9, confidence=1.0, extractor="t")
    span = ExtractedSpan(slot="q", question="q", text="jazz", start=0, end=4, confidence=1.0, extractor="t")
    with pytest.raises(NoAnswer):
        span.verify("nope")


@pytest.mark.asyncio
async def test_confidence_threshold_is_honoured():
    with pytest.raises(NoAnswer):
        await HeuristicSpanExtractor().extract("Play some Jazz", "q", slot="query",
                                               confidence_threshold=1.01)


@pytest.mark.asyncio
async def test_goal_builder_fills_slots_and_reports_missing_required_ones():
    builder = GoalBuilder(HeuristicSpanExtractor())
    built = await builder.build(capability="Spotify", schema=PLAY, utterance="Play some Jazz")
    assert built.goal.arguments["query"] == "Jazz"
    assert built.complete and built.extraction("query").start == 10

    partial = await builder.build(capability="Spotify", schema=PLAY, utterance="play some", final=False)
    assert not partial.complete and partial.missing_slots == ("query",)
    assert partial.rejected["query"] == "partial_no_answer"
    # A partial transcript must never fabricate a required argument.
    assert "query" not in partial.goal.arguments


@pytest.mark.asyncio
async def test_goal_builder_can_require_completeness():
    builder = GoalBuilder(HeuristicSpanExtractor(), allow_incomplete=False)
    with pytest.raises(IncompleteGoal):
        await builder.build(capability="Spotify", schema=PLAY, utterance="play some")


@pytest.mark.asyncio
async def test_goal_builder_uses_provided_arguments_without_extraction():
    builder = GoalBuilder(HeuristicSpanExtractor())
    built = await builder.build(capability="Spotify", schema=PLAY, utterance="whatever",
                                initial_arguments={"query": "Jazz"})
    assert built.goal.arguments["query"] == "Jazz"
    assert built.extraction("query").extractor == "provided"


@pytest.mark.asyncio
async def test_extractive_qa_backend_is_swappable_and_not_hard_coded():
    class StubQA:
        name = "stub-qa"

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            assert question == "What should be played?"
            start = utterance.index("Jazz")
            return ExtractedSpan(slot=slot, question=question, text="Jazz", start=start, end=start + 4,
                                 confidence=0.99, extractor=self.name)

    builder = GoalBuilder(StubQA())
    built = await builder.build(capability="Spotify", schema=PLAY, utterance="Play Jazz")
    assert built.goal.arguments["query"] == "Jazz"
    assert built.extraction("query").extractor == "stub-qa"


@pytest.mark.asyncio
async def test_chain_extractor_falls_through_to_the_next_backend():
    class Failing:
        name = "failing"

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            raise NoAnswer("nope")

    chain = ChainExtractor((Failing(), HeuristicSpanExtractor()))
    span = await chain.extract("Play some Jazz", "What should be played?", slot="query")
    assert span.text == "Jazz" and span.extractor == "heuristic"


@pytest.mark.asyncio
async def test_chain_floor_lets_a_better_fallback_answer_win():
    """A low-confidence model fragment must not preempt a fuller fallback answer."""

    class ShakyQA:
        name = "shaky-qa"

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            start = utterance.index("PersonaPlex")
            return ExtractedSpan(slot=slot, question=question, text="PersonaPlex",
                                 start=start, end=start + len("PersonaPlex"),
                                 confidence=0.39, extractor=self.name)

    chain = ChainExtractor((ShakyQA(), HeuristicSpanExtractor()), floors=(0.5,))
    span = await chain.extract("Search for PersonaPlex turn detection", "What should be searched for?",
                               slot="query")
    assert span.text == "PersonaPlex turn detection" and span.extractor == "heuristic"

    # A confident model answer still wins over the fallback.
    class ConfidentQA(ShakyQA):
        name = "confident-qa"

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            span = await ShakyQA.extract(self, utterance, question, slot=slot,
                                         confidence_threshold=confidence_threshold)
            return ExtractedSpan(slot=slot, question=question, text=span.text, start=span.start,
                                 end=span.end, confidence=0.95, extractor=self.name)

    chain = ChainExtractor((ConfidentQA(), HeuristicSpanExtractor()), floors=(0.5,))
    span = await chain.extract("Search for PersonaPlex turn detection", "What should be searched for?",
                               slot="query")
    assert span.text == "PersonaPlex" and span.extractor == "confident-qa"


def test_extractive_qa_defaults_to_minilm_squad2():
    extractor = ExtractiveQAExtractor()
    assert extractor.model_name == "deepset/minilm-uncased-squad2"


@pytest.mark.asyncio
async def test_extractive_qa_rejects_an_answer_inside_the_narrative_frame(monkeypatch):
    """Offsets are mapped back to the raw transcript; a frame answer is not a span."""
    import builtins

    from ping_ponder.agentic.span import ExtractorUnavailable

    loaded = {}

    class FakePipeline:
        def __call__(self, *, question, context, handle_impossible_answer):
            loaded["context"] = context
            # Answer the frame instead of the utterance.
            start = context.index("user")
            return {"answer": "user", "score": 0.99, "start": start, "end": start + 4}

    extractor = ExtractiveQAExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: FakePipeline())
    with pytest.raises(NoAnswer):
        await extractor.extract("Play some Jazz", "What music does the user want to play?", slot="q")
    # The frame is used for the model only, never for the caller.
    assert loaded["context"].endswith("Play some Jazz")


@pytest.mark.asyncio
async def test_extractive_qa_maps_framed_offsets_back_to_the_transcript(monkeypatch):
    class FakePipeline:
        def __call__(self, *, question, context, handle_impossible_answer):
            start = context.index("Jazz")
            return {"answer": "Jazz", "score": 0.97, "start": start, "end": start + 4}

    extractor = ExtractiveQAExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: FakePipeline())
    span = await extractor.extract("Play some Jazz", "What music does the user want to play?", slot="q")
    assert span.text == "Jazz" and (span.start, span.end) == (10, 14)
    assert span.verify("Play some Jazz") is span


@pytest.mark.asyncio
async def test_extractive_qa_reports_unavailable_without_optional_dependencies(monkeypatch):
    """A missing backend must surface as `ExtractorUnavailable`, never a fabricated answer.

    The transformers import is forced to fail so this never downloads a model.
    """
    import builtins

    from ping_ponder.agentic.span import ExtractorUnavailable

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("no transformers in this environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    extractor = ExtractiveQAExtractor()
    assert extractor.available() is False
    with pytest.raises(ExtractorUnavailable):
        await extractor.extract("Play some Jazz", "What should be played?", slot="query")


def test_locate_span_requires_a_verbatim_transcript_span():
    from ping_ponder.agentic.span import locate_span

    assert locate_span("Play some Jazz", "Jazz") == (10, 14)
    assert locate_span("Play some Jazz", "  Jazz  ") == (10, 14)
    # Case differences and paraphrases are not spans. Rejecting beats repairing:
    # normalizing would let an extractor put words in the user's mouth.
    assert locate_span("Play some Jazz", "jazz") is None
    assert locate_span("Play some Jazz", "Play the Beatles") is None
    assert locate_span("Play some Jazz", "") is None


def test_nuextract_is_a_swappable_backend_not_hard_coded():
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()
    assert extractor.model_name == "numind/NuExtract-1.5-tiny"
    # It must satisfy the same ArgumentExtractor contract as every other backend.
    from ping_ponder.agentic.span import ArgumentExtractor

    assert isinstance(extractor, ArgumentExtractor)


@pytest.mark.asyncio
async def test_nuextract_templates_use_the_slot_question_as_the_json_key(monkeypatch):
    """NuExtract reads the JSON key as the field description.

    Bare keys made it echo the whole utterance back, which the anti-echo guard then
    rejected - so the extractor silently found nothing. The key must be the slot's
    extraction question, and the reply must be mapped back to the slot name.
    """
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()
    seen: dict[str, str] = {}

    def fake_generate_fields(tokenizer, model, utterance, fields):
        seen.update(fields)
        return {next(iter(fields)): "jazz"}

    monkeypatch.setattr(extractor, "_load", lambda: (object(), object()))
    monkeypatch.setattr(extractor, "_generate_fields", fake_generate_fields)
    result = await extractor.extract("play some jazz", "What music does the user want to play?",
                                     slot="query")
    assert list(seen) == ["What music does the user want to play?"]
    assert result.slot == "query" and result.text == "jazz"


@pytest.mark.asyncio
async def test_nuextract_multi_slot_maps_question_keys_back_to_slot_names(monkeypatch):
    """A batch keyed by question must still return spans keyed by slot name."""
    from ping_ponder.agentic.span import NuExtractExtractor, SlotRequest

    extractor = NuExtractExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: (object(), object()))
    monkeypatch.setattr(extractor, "_generate_fields",
                        lambda tok, model, utterance, fields: {
                            "Who should receive the money?": "Sarah",
                            "How much money should be transferred?": "500"})
    spans = await extractor.extract_many("Transfer money to Sarah for 500", (
        SlotRequest(slot="recipient", question="Who should receive the money?"),
        SlotRequest(slot="amount", question="How much money should be transferred?"),
    ))
    assert set(spans) == {"recipient", "amount"}
    assert spans["recipient"].text == "Sarah" and spans["amount"].text == "500"


@pytest.mark.asyncio
async def test_nuextract_rejects_a_full_utterance_echo(monkeypatch):
    """A generative extractor echoes the utterance when it finds nothing; that is no answer."""
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: (object(), object()))
    monkeypatch.setattr(extractor, "_generate", lambda *a, **k: "pause it")
    with pytest.raises(NoAnswer):
        await extractor.extract("pause it", "What music?", slot="music")

    monkeypatch.setattr(extractor, "_generate", lambda *a, **k: "Jazz")
    span = await extractor.extract("Play some Jazz", "What music?", slot="music")
    assert span.text == "Jazz" and span.verify("Play some Jazz") is span


@pytest.mark.asyncio
async def test_nuextract_accepts_a_verbatim_compound_value(monkeypatch):
    """NuExtract's win over the QA reader: it keeps a compound value intact."""
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: (object(), object()))
    monkeypatch.setattr(extractor, "_generate", lambda *a, **k: "Kind of Blue")
    span = await extractor.extract("play the album Kind of Blue", "What music?", slot="music")
    assert span.text == "Kind of Blue"


@pytest.mark.asyncio
async def test_nuextract_rejects_a_value_that_is_not_in_the_transcript(monkeypatch):
    """A generative model can return a plausible value the user never said."""
    from ping_ponder.agentic.span import NuExtractExtractor

    extractor = NuExtractExtractor()
    monkeypatch.setattr(extractor, "_load", lambda: (object(), object()))
    # The user said "Jaz", the model "corrected" it to "Jazz".
    monkeypatch.setattr(extractor, "_generate", lambda *a, **k: "Jazz")
    with pytest.raises(NoAnswer):
        await extractor.extract("Play some Jaz", "What music?", slot="music")


@pytest.mark.asyncio
async def test_multi_slot_backend_fills_a_goal_in_one_pass():
    """A three-slot goal costs one model call, not three."""
    from ping_ponder.agentic.span import ExtractedSpan

    class BatchExtractor:
        name = "batch"

        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            self.single_calls += 1
            raise NoAnswer("single-slot path not used")

        async def extract_many(self, utterance, requests):
            self.batch_calls += 1
            found = {}
            for request in requests:
                value = {"recipient": "Sarah", "amount": "500"}.get(request.slot)
                if value is None:
                    continue
                start = utterance.index(value)
                found[request.slot] = ExtractedSpan(
                    slot=request.slot, question=request.question, text=value, start=start,
                    end=start + len(value), confidence=0.95, extractor=self.name)
            return found

    from ping_ponder.agentic.capabilities.transfer import build_transfer_descriptor

    schema = build_transfer_descriptor(None).schema("TRANSFER")
    extractor = BatchExtractor()
    built = await GoalBuilder(extractor).build(capability="Transfer", schema=schema,
                                               utterance="Transfer money to Sarah for 500")
    assert extractor.batch_calls == 1 and extractor.single_calls == 0
    assert built.goal.arguments["recipient"] == "Sarah"
    assert built.goal.arguments["amount"] == 500.0


@pytest.mark.asyncio
async def test_single_slot_backend_still_works_when_batching_is_unavailable():
    """The multi-slot path is optional; a plain extractor must keep working."""
    builder = GoalBuilder(HeuristicSpanExtractor())
    built = await builder.build(capability="Spotify", schema=PLAY, utterance="Play some Jazz")
    assert built.goal.arguments["query"] == "Jazz"


@pytest.mark.asyncio
async def test_batch_failure_falls_back_to_per_slot_extraction():
    from ping_ponder.agentic.span import ExtractedSpan

    class BrokenBatch:
        name = "broken-batch"

        async def extract_many(self, utterance, requests):
            raise RuntimeError("batch backend down")

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            start = utterance.index("Jazz")
            return ExtractedSpan(slot=slot, question=question, text="Jazz", start=start,
                                 end=start + 4, confidence=0.9, extractor=self.name)

    built = await GoalBuilder(BrokenBatch()).build(capability="Spotify", schema=PLAY,
                                                   utterance="Play some Jazz")
    assert built.goal.arguments["query"] == "Jazz"

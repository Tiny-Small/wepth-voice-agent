"""Transfer capability on the agentic spine: planning, safety gates, no auto-confirm."""

import pytest

from ping_ponder.agentic.capabilities.transfer import (AUTHORIZED_KEY, CONFIRMED_KEY, PREPARED_KEY,
                                                       TRANSACTION_KEY, build_transfer_descriptor,
                                                       transfer_subject, transfer_world_schema)
from ping_ponder.agentic.confirmation import ConfirmationLedger
from ping_ponder.agentic.goal_builder import GoalBuilder
from ping_ponder.agentic.jev import GlobalRoute
from ping_ponder.agentic.span import HeuristicSpanExtractor


class FakeGlobal:
    """Minimal router for tests that drive the spine directly."""

    async def route(self, utterance, *, active_local=None, context=None):
        return GlobalRoute(capability="Transfer", confidence=0.9)
from ping_ponder.agentic.executor import ExecutionOutcome, Executor
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.planner import DeterministicPlanner, PlanningFailed
from ping_ponder.agentic.world import WorldState

DESCRIPTOR = build_transfer_descriptor(None)
TRANSFER = DESCRIPTOR.schema("TRANSFER")


def world() -> WorldState:
    return WorldState.from_schema(transfer_world_schema())


def goal(recipient: str = "Sarah", amount: float = 500.0) -> SemanticGoal:
    return SemanticGoal("Transfer", "TRANSFER", {"recipient": recipient, "amount": amount})


def executor() -> Executor:
    return Executor(DeterministicPlanner())


def authorize(world_state, ledger, subject=None):
    """Grant authorization the way the host does: bind, then mark authorized."""
    subject = subject or transfer_subject(world_state.get(TRANSACTION_KEY))
    ledger.activate("Transfer", subject)
    assert ledger.confirms("Transfer", subject)
    return world_state.updated({AUTHORIZED_KEY: True,
                                "session.confirmation_ledger": ledger})


def test_planner_derives_the_transfer_sequence_from_the_goal():
    plan = DeterministicPlanner().plan(goal(), TRANSFER, world(), DESCRIPTOR.operators,
                                       limits=DESCRIPTOR.planning_limits())
    assert [step.operator.name for step in plan.steps] == [
        "StartTransfer", "ResolveRecipient", "SelectSourceAccount", "CheckLimit",
        "PrepareTransfer"]


@pytest.mark.asyncio
async def test_execution_prepares_but_never_confirms_without_user_authorization():
    """The planner must not be able to authorize a transfer by itself."""
    report = await executor().execute(goal(), TRANSFER, world(), DESCRIPTOR)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert "ConfirmTransfer" not in report.executed
    assert report.world_after.get(CONFIRMED_KEY) is False
    assert report.world_after.get(TRANSACTION_KEY).status.value == "AWAITING_AUTHORIZATION"


@pytest.mark.asyncio
async def test_confirmation_is_reachable_only_with_the_host_authorization_signal():
    prepared = await executor().execute(goal(), TRANSFER, world(), DESCRIPTOR)
    ledger = ConfirmationLedger()
    authorized = authorize(prepared.world_after, ledger)
    confirm = DESCRIPTOR.schema("CONFIRM")
    report = await executor().execute(SemanticGoal("Transfer", "CONFIRM"), confirm, authorized,
                                      DESCRIPTOR)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.executed == ("ConfirmTransfer",)
    assert report.world_after.get(CONFIRMED_KEY) is True


@pytest.mark.asyncio
async def test_confirmation_is_refused_without_a_bound_presentation():
    """The authorization flag alone is not enough; a live binding must also exist."""
    prepared = await executor().execute(goal(), TRANSFER, world(), DESCRIPTOR)
    unbound = prepared.world_after.updated(
        {AUTHORIZED_KEY: True, "session.confirmation_ledger": ConfirmationLedger()})
    report = await executor().execute(SemanticGoal("Transfer", "CONFIRM"),
                                      DESCRIPTOR.schema("CONFIRM"), unbound, DESCRIPTOR)
    assert report.outcome is ExecutionOutcome.BLOCKED
    assert "not bound to this transfer" in report.reason
    assert report.world_after.get(CONFIRMED_KEY) is False


@pytest.mark.asyncio
async def test_confirmation_is_not_replanned_once_granted():
    """An idempotent command goal must report satisfaction instead of looping."""
    prepared = await executor().execute(goal(), TRANSFER, world(), DESCRIPTOR)
    ledger = ConfirmationLedger()
    authorized = authorize(prepared.world_after, ledger)
    confirmed = (await executor().execute(SemanticGoal("Transfer", "CONFIRM"),
                                          DESCRIPTOR.schema("CONFIRM"), authorized, DESCRIPTOR)).world_after
    confirmed = confirmed.updated({"session.confirmation_ledger": ledger})
    again = await executor().execute(SemanticGoal("Transfer", "CONFIRM"),
                                     DESCRIPTOR.schema("CONFIRM"), confirmed, DESCRIPTOR)
    assert again.outcome is ExecutionOutcome.SATISFIED
    assert again.executed == ()


@pytest.mark.asyncio
async def test_over_limit_amount_is_infeasible_not_merely_blocked():
    """A policy limit is a semantic outcome, not an internal search failure.

    INFEASIBLE (rather than BLOCKED) is what lets the reply layer say "that amount is
    over the limit" instead of reporting a search bound.
    """
    report = await executor().execute(goal(amount=5000.0), TRANSFER, world(), DESCRIPTOR)
    assert report.outcome is ExecutionOutcome.INFEASIBLE
    assert "amount_within_limit" in report.reason
    assert report.world_after.get(PREPARED_KEY) is False
    assert report.world_after.get(CONFIRMED_KEY) is False


def test_limit_is_declarative_so_the_planner_can_see_it():
    """The limit must be a precondition, not only an execution-time check.

    While it lived only in the executor's validation, the planner projected a limit
    check that execution would reject and thrashed until it exhausted its budget,
    hiding the real reason from the user.
    """
    operator = DESCRIPTOR.operator("CheckLimit")
    assert any(condition.key == "transfer.amount_within_limit"
               for condition in operator.preconditions)


@pytest.mark.asyncio
async def test_satisfaction_requires_the_requested_recipient_not_merely_a_transfer():
    """A prepared transfer for someone else must not satisfy this goal."""
    first = await executor().execute(goal("Sarah", 500.0), TRANSFER, world(), DESCRIPTOR)
    assert first.outcome is ExecutionOutcome.SATISFIED
    other = await executor().execute(goal("Bob", 200.0), TRANSFER, first.world_after, DESCRIPTOR)
    assert other.outcome is ExecutionOutcome.SATISFIED
    state = other.world_after.get(TRANSACTION_KEY)
    assert state.recipient_text == "Bob" and float(state.amount) == 200.0


@pytest.mark.asyncio
async def test_a_new_transfer_does_not_inherit_prior_confirmation_or_authorization():
    """Confirmation must not leak from one transfer to the next."""
    prepared = await executor().execute(goal("Sarah", 500.0), TRANSFER, world(), DESCRIPTOR)
    ledger = ConfirmationLedger()
    authorized = authorize(prepared.world_after, ledger)
    confirmed = (await executor().execute(SemanticGoal("Transfer", "CONFIRM"),
                                          DESCRIPTOR.schema("CONFIRM"), authorized, DESCRIPTOR)).world_after
    assert confirmed.get(CONFIRMED_KEY) is True

    fresh = await executor().execute(goal("Bob", 200.0), TRANSFER, confirmed, DESCRIPTOR)
    assert fresh.world_after.get(CONFIRMED_KEY) is False
    assert fresh.world_after.get(AUTHORIZED_KEY) is False
    assert "ConfirmTransfer" not in fresh.executed


def test_absent_recipient_makes_transfer_unplannable():
    with pytest.raises(PlanningFailed):
        DeterministicPlanner().plan(SemanticGoal("Transfer", "TRANSFER", {"amount": 500.0}),
                                    TRANSFER, world(), DESCRIPTOR.operators,
                                    limits=DESCRIPTOR.planning_limits())


def test_goal_schemas_declare_open_ended_slots_with_typed_amount():
    assert [slot.name for slot in TRANSFER.slots] == ["recipient", "amount", "currency"]
    assert TRANSFER.slot("amount").kind.value == "number"
    assert TRANSFER.slot("currency").required is False
    # A currency slot must reject a non-currency answer from a weak extractor.
    assert TRANSFER.slot("currency").accepts("SGD") is True
    assert TRANSFER.slot("currency").accepts("Sarah") is False


# --- confirmation binding: the attacks a stale "yes" must not win -------------------

@pytest.mark.asyncio
async def test_confirmation_cannot_authorize_a_materially_edited_transfer():
    """A "yes" for 500 must not authorize 900 after the amount changes.

    Exercised through the spine, because dropping a stale binding is the session's job
    (via `Descriptor.confirmation_subject`), not something an isolated ledger can know.
    """
    from ping_ponder.agentic.spine import VoiceActionSpine
    from ping_ponder.agentic.wiring import build_default_registry

    service = VoiceActionSpine(registry=build_default_registry(), global_jev=FakeGlobal(),
                               goal_builder=GoalBuilder(HeuristicSpanExtractor()),
                               world=world())
    await service.resolve_final("Transfer money to Sarah for 500")
    presented = transfer_subject(service.world.get(TRANSACTION_KEY))
    service.present_for_confirmation("Transfer", presented)
    assert service.confirmations.current() is not None

    await service.resolve_final("Transfer money to Sarah for 900")
    # The material edit dropped the stale presentation, so the old yes cannot be used.
    assert service.confirmations.current() is None
    assert service.confirm_action("Transfer", presented) is False


@pytest.mark.asyncio
async def test_confirmation_cannot_survive_a_capability_change():
    ledger = ConfirmationLedger()
    subject = "transfer to Sarah 500 SGD from acct-sgd"
    ledger.activate("Transfer", subject)
    assert ledger.confirms("Transfer", subject) is True
    ledger.change_topic()
    assert ledger.confirms("Transfer", subject) is False
    assert ledger.current() is None


def test_confirmation_requires_a_presentation_before_it_binds():
    ledger = ConfirmationLedger()
    assert ledger.confirms("Transfer", "anything") is False


def test_confirmation_refuses_a_different_subject_or_capability():
    ledger = ConfirmationLedger()
    ledger.activate("Transfer", "transfer A")
    assert ledger.confirms("Transfer", "transfer A") is True
    assert ledger.confirms("Transfer", "transfer B") is False
    assert ledger.confirms("Spotify", "transfer A") is False
    assert ledger.confirms("Transfer", None) is False


def test_refresh_if_changed_drops_a_pending_confirmation():
    ledger = ConfirmationLedger()
    ledger.activate("Transfer", "transfer A")
    ledger.refresh_if_changed("Transfer", "transfer A")
    assert ledger.current() is not None
    ledger.refresh_if_changed("Transfer", "transfer B")
    assert ledger.current() is None


def test_transfer_subject_is_derived_only_from_material_fields():
    from ping_ponder.domain.transaction import TransactionState

    base = TransactionState(task_id="tx-1", recipient_text="Sarah", recipient_id="sarah",
                            amount=500, currency="SGD", source_account="acct-sgd")
    subject = transfer_subject(base)
    assert subject == "transfer to Sarah 500 SGD from acct-sgd"
    # Any material edit changes the subject.
    for change in ({"amount": 900}, {"recipient_text": "Bob"}, {"currency": "USD"},
                   {"source_account": "acct-usd"}):
        assert transfer_subject(base.model_copy(update=change)) != subject
    # An incomplete transfer has no confirmable subject.
    assert transfer_subject(TransactionState(task_id="tx-2")) is None

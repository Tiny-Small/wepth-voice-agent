"""Transfer capability on the agentic spine.

This is the decoupling target for the older routing/decision path. It reuses the
transaction *domain* (versioned state, controller, mock bank) and the
irreversible-action safety rules, but expresses Local Jev output as semantic goals
and derives the tool sequence with the planner instead of a hand-written
`_schedule_feasible` candidate list.

Semantic goals are `TRANSFER`, `CHECK_BALANCE`, `CANCEL`, and `CONFIRM`. A transfer's
slots (`recipient`, `amount`, `currency`) are open-ended and filled by span
extraction; `amount` is a NUMBER slot typed from its extracted span.

Two properties make this safe to plan with:

* **Projections are pure and total.** `project(world, args)` returns the world
  changes a step produces on success and never logs, mutates, or raises. Declared
  effects are recomputed from the same function, so the planner cannot project a
  state the executor will not produce.
* **Validation happens at execution.** Limit and completeness checks live in
  `validate`, so a failed check is observed by the executor and replanned rather
  than corrupting a planning projection.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable, Mapping

from ping_ponder.domain.transaction import TransactionState, TransactionStatus
from ping_ponder.observability import emit

from ..confirmation import CONFIRMATION_LEDGER_KEY, ConfirmationLedger
from ..goals import GoalSchema, SlotKind, SlotSpec
from ..operators import ActionOperator, OperatorError, OperatorMetadata
from ..registry import CapabilityDescriptor
from ..world import Compare, Condition, Effect, WorldState

logger = logging.getLogger(__name__)

# Capability-private world keys. Planner/executor inputs only: no Jev ever sees these.
TRANSACTION_KEY = "transfer.transaction"
CONTROLLER_KEY = "transfer.controller"
CONFIRMED_KEY = "transfer.confirmed"
RESOLVED_KEY = "transfer.recipient_resolved"
ACCOUNT_KEY = "transfer.account_selected"
LIMIT_KEY = "transfer.limit_checked"
PREPARED_KEY = "transfer.prepared"
BALANCE_KEY = "transfer.balance_reported"
RECEIPT_KEY = "transfer.receipt"
NEXT_TASK_KEY = "transfer.next_task_number"
# Set by the host only for an explicit confirmation utterance. The planner can never
# produce this key, which is what stops it from authorizing a transfer on its own.
# Authorization is additionally bound to the exact prepared transfer via the session
# ledger, so a confirmation cannot be replayed onto a different transfer.
AUTHORIZED_KEY = "transfer.user_authorized"

DEFAULT_CURRENCY = "SGD"
KNOWN_CURRENCIES = frozenset({"SGD", "USD", "EUR", "GBP", "JPY", "MYR", "AUD", "HKD", "CNY"})


def _looks_like_currency(text: str) -> bool:
    return text.strip().upper() in KNOWN_CURRENCIES
MAX_TRANSFER = Decimal("1000")

_INVALIDATED_BY_EDIT = (RESOLVED_KEY, ACCOUNT_KEY, LIMIT_KEY, PREPARED_KEY, CONFIRMED_KEY, RECEIPT_KEY)

Projection = Callable[[WorldState, Mapping[str, Any]], Mapping[str, Any]]
Validation = Callable[[WorldState, Mapping[str, Any]], None]


def transfer_world_schema() -> Mapping[str, Any]:
    return {TRANSACTION_KEY: None, CONTROLLER_KEY: None, CONFIRMED_KEY: False,
            AUTHORIZED_KEY: False, RESOLVED_KEY: False, ACCOUNT_KEY: False, LIMIT_KEY: False,
            PREPARED_KEY: False, BALANCE_KEY: False, RECEIPT_KEY: None, NEXT_TASK_KEY: 1,
            CONFIRMATION_LEDGER_KEY: None}


def _state(world: WorldState) -> TransactionState | None:
    return world.get(TRANSACTION_KEY)


def _bump(state: TransactionState, **changes: Any) -> TransactionState:
    """Every material edit bumps the version, which is what invalidates authorization."""
    return state.model_copy(update={**changes, "version": state.version + 1})


# --- pure projections -----------------------------------------------------------------

def _start(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    number = int(world.get(NEXT_TASK_KEY) or 1)
    state = TransactionState(task_id=f"tx-{number:03d}", recipient_text=arguments.get("recipient"))
    changes: dict[str, Any] = {}
    if arguments.get("amount") is not None:
        changes["amount"] = Decimal(str(arguments["amount"]))
    if arguments.get("currency"):
        changes["currency"] = arguments["currency"]
    if changes:
        state = _bump(state, **changes)
    return {TRANSACTION_KEY: state, NEXT_TASK_KEY: number + 1, CONFIRMED_KEY: False,
            AUTHORIZED_KEY: False, RESOLVED_KEY: False, ACCOUNT_KEY: False, LIMIT_KEY: False,
            PREPARED_KEY: False, RECEIPT_KEY: None}


def _resolve_recipient(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    state = _state(world)
    return {TRANSACTION_KEY: _bump(state, recipient_id=state.recipient_text.strip().lower().replace(" ", "-")),
            RESOLVED_KEY: True}


def _select_account(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    state = _state(world)
    return {TRANSACTION_KEY: _bump(state, source_account=f"acct-{state.currency.lower()}"),
            ACCOUNT_KEY: True}


def _check_limit(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    # A policy check changes no transaction material, so the version must not move.
    return {LIMIT_KEY: True}


def _prepare(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    state = _state(world)
    return {TRANSACTION_KEY: _bump(state, status=TransactionStatus.AWAITING_AUTHORIZATION),
            PREPARED_KEY: True}


def _check_balance(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    return {BALANCE_KEY: True}


def _cancel(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    state = _state(world)
    return {TRANSACTION_KEY: _bump(state, status=TransactionStatus.CANCELLED), PREPARED_KEY: False,
            CONFIRMED_KEY: False, AUTHORIZED_KEY: False, LIMIT_KEY: False, RECEIPT_KEY: None}


def _confirm(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """The irreversible-action gate.

    Requires a host-set authorization signal *and* a session ledger binding that
    confirmation to this exact prepared transfer in the current topic epoch. The
    planner can produce neither, so a transfer request can never confirm itself, and a
    confirmation cannot be replayed onto a materially different transfer.
    """
    return {CONFIRMED_KEY: True}


# --- validation (execution only) ------------------------------------------------------

def _require_state(world: WorldState, arguments: Mapping[str, Any]) -> None:
    if _state(world) is None:
        raise OperatorError("no active transaction")


def _require_recipient(world: WorldState, arguments: Mapping[str, Any]) -> None:
    _require_state(world, arguments)
    if not _state(world).recipient_text:
        raise OperatorError("a recipient must be known before it can be resolved")


def _require_limit_ok(world: WorldState, arguments: Mapping[str, Any]) -> None:
    _require_state(world, arguments)
    state = _state(world)
    if state.amount is None:
        raise OperatorError("limit check requires an amount")
    if state.amount > MAX_TRANSFER:
        raise OperatorError(f"amount {state.amount} exceeds the transfer limit {MAX_TRANSFER}")


def _require_complete(world: WorldState, arguments: Mapping[str, Any]) -> None:
    _require_state(world, arguments)
    state = _state(world)
    if not state.recipient_id:
        raise OperatorError("prepared transfer requires a resolved recipient id")
    if not state.is_complete():
        raise OperatorError("prepared transfer requires resolved execution fields")


def transfer_subject(state: TransactionState | None) -> str | None:
    """The stable identity of a prepared transfer, used for confirmation binding.

    Derived only from material fields, so any edit in between (amount, recipient,
    account, currency) changes the subject and invalidates a prior confirmation.
    """
    if state is None or not state.is_complete():
        return None
    return (f"transfer to {state.recipient_text or state.recipient_id} "
            f"{state.amount} {state.currency} from {state.source_account}")


def _ledger(world: WorldState) -> ConfirmationLedger | None:
    return world.get(CONFIRMATION_LEDGER_KEY)


def _already_confirmed(world: WorldState) -> bool:
    return bool(world.get(CONFIRMED_KEY))


def _already_cancelled(world: WorldState) -> bool:
    state = _state(world)
    return bool(state and state.status is TransactionStatus.CANCELLED)


def _require_prepared(world: WorldState, arguments: Mapping[str, Any]) -> None:
    _require_state(world, arguments)
    if not world.get(PREPARED_KEY):
        raise OperatorError("confirmation requires a prepared transfer")
    if not world.get(AUTHORIZED_KEY):
        raise OperatorError("confirmation requires explicit user authorization")
    ledger = _ledger(world)
    if ledger is None:
        raise OperatorError("confirmation requires a session confirmation ledger")
    state = _state(world)
    subject = transfer_subject(state)
    if not ledger.confirms("Transfer", subject):
        raise OperatorError(
            "confirmation is not bound to this transfer "
            f"(presented={ledger.current().describe() if ledger.current() else None}, "
            f"current={subject})")


def _require_pending(world: WorldState, arguments: Mapping[str, Any]) -> None:
    _require_state(world, arguments)


def _executor(projection: Projection, validation: Validation | None = None) -> Callable:
    """Validate, then apply the same pure projection the planner used."""

    def run(world: WorldState, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if validation is not None:
            validation(world, arguments)
        emit(logger, "transfer_transition", projection=projection.__name__)
        return projection(world, arguments)

    return run


def _declared(projection: Projection, keys: tuple[str, ...]) -> tuple[Effect, ...]:
    """Declare exactly the keys a projection writes, using that same pure projection."""
    return tuple(Effect(key, derive=lambda world, args, _p=projection, _k=key: _p(world, args)[_k])
                 for key in keys)


# --- declarative conditions -----------------------------------------------------------

def _has_recipient_text(world: WorldState) -> bool:
    state = _state(world)
    return bool(state and state.recipient_text)


def _has_amount(world: WorldState) -> bool:
    state = _state(world)
    return bool(state and state.amount is not None)


def _goal_subject(world: WorldState, arguments: Mapping[str, Any]) -> str | None:
    """What this goal asked for, in the same shape as `transfer_subject`."""
    state = _state(world)
    if state is None:
        return None
    recipient = arguments.get("recipient") or state.recipient_text
    amount = arguments.get("amount")
    currency = arguments.get("currency") or state.currency
    if not recipient or amount is None:
        return None
    amount_text = Decimal(str(amount))
    account = state.source_account
    if account is None:
        return None
    return f"transfer to {recipient} {amount_text} {currency} from {account}"


def _matches_goal(world: WorldState, arguments: Mapping[str, Any]) -> bool:
    """Whether the prepared transfer is the one this goal asked for."""
    state = _state(world)
    if state is None or not world.get(PREPARED_KEY):
        return False
    if arguments.get("recipient") and state.recipient_text != arguments["recipient"]:
        return False
    if arguments.get("amount") is not None and state.amount != Decimal(str(arguments["amount"])):
        return False
    return True


COND_STATE = Condition(TRANSACTION_KEY, predicate=lambda world: _state(world) is not None)
COND_MATCHES_GOAL = Condition("transfer.matches_goal", bound_predicate=_matches_goal)
COND_NO_MATCHING_TRANSFER = Condition("transfer.needs_start", bound_predicate=lambda world, args: not _matches_goal(world, args))
COND_RECIPIENT = Condition("transfer.state_has_recipient", predicate=_has_recipient_text)
COND_AMOUNT = Condition("transfer.state_has_amount", predicate=_has_amount)
COND_PREPARED = Condition(PREPARED_KEY, predicate=lambda world: bool(world.get(PREPARED_KEY)))
def _confirmed_for_presented(world: WorldState, arguments: Mapping[str, Any]) -> bool:
    """Whether the transfer the user was shown is the confirmed one.

    A CONFIRM goal carries no arguments, so this compares against the ledger's
    presented subject. That keeps a confirmed transfer from making an unrelated later
    transfer look satisfied.
    """
    if not world.get(CONFIRMED_KEY):
        return False
    primer = _ledger(world)
    presented = primer.current().subject if primer is not None and primer.current() else None
    return presented is not None and transfer_subject(_state(world)) == presented


COND_CONFIRMED_FOR_THIS = Condition("transfer.confirmed_bound",
                                    bound_predicate=_confirmed_for_presented)
COND_ACCOUNT = Condition(ACCOUNT_KEY, predicate=lambda world: bool(world.get(ACCOUNT_KEY)))
COND_RECIPIENT_RESOLVED = Condition(RESOLVED_KEY, predicate=lambda world: bool(world.get(RESOLVED_KEY)))
COND_LIMIT = Condition(LIMIT_KEY, predicate=lambda world: bool(world.get(LIMIT_KEY)))


def _amount_within_limit(world: WorldState) -> bool:
    """The transfer-limit policy, expressed so the planner can reason about it.

    The executor still enforces this; stating it as a precondition as well is what lets
    an over-limit goal fail fast with the real reason instead of exhausting the search.
    """
    state = _state(world)
    return bool(state) and state.amount is not None and state.amount <= MAX_TRANSFER


COND_WITHIN_LIMIT = Condition("transfer.amount_within_limit", predicate=_amount_within_limit)
COND_AUTHORIZED = Condition(AUTHORIZED_KEY, predicate=lambda world: bool(world.get(AUTHORIZED_KEY)))


def transfer_operators() -> tuple[ActionOperator, ...]:
    return (
        ActionOperator(
            name="StartTransfer",
            # Slots come from the goal; the task identifier is allocated from world
            # state, so it never becomes a slot the extractor must fill. The
            # precondition permits replacing a transaction that does not match this
            # goal (a new transfer request), but never one already prepared for it.
            parameters=("recipient",),
            optional_parameters=("amount", "currency"),
            preconditions=(COND_NO_MATCHING_TRANSFER,),
            effects=_declared(_start, (TRANSACTION_KEY, NEXT_TASK_KEY, CONFIRMED_KEY, AUTHORIZED_KEY,
                                       RESOLVED_KEY, ACCOUNT_KEY, LIMIT_KEY, PREPARED_KEY, RECEIPT_KEY)),
            cost=1.0,
            executor=_executor(_start),
            metadata=OperatorMetadata(speculative_safe=False, requires_final=True),
        ),
        ActionOperator(
            name="ResolveRecipient",
            preconditions=(COND_STATE, COND_RECIPIENT),
            effects=_declared(_resolve_recipient, (TRANSACTION_KEY, RESOLVED_KEY)),
            cost=1.0,
            executor=_executor(_resolve_recipient, _require_recipient),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="SelectSourceAccount",
            preconditions=(COND_STATE, COND_RECIPIENT_RESOLVED),
            effects=_declared(_select_account, (TRANSACTION_KEY, ACCOUNT_KEY)),
            cost=1.0,
            executor=_executor(_select_account, _require_recipient),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="CheckLimit",
            preconditions=(COND_STATE, COND_AMOUNT, COND_WITHIN_LIMIT),
            effects=_declared(_check_limit, (LIMIT_KEY,)),
            cost=1.0,
            executor=_executor(_check_limit, _require_limit_ok),
            metadata=OperatorMetadata(speculative_safe=False, requires_final=True),
        ),
        ActionOperator(
            name="PrepareTransfer",
            preconditions=(COND_STATE, COND_ACCOUNT, COND_LIMIT),
            effects=_declared(_prepare, (TRANSACTION_KEY, PREPARED_KEY)),
            cost=1.0,
            executor=_executor(_prepare, _require_complete),
            metadata=OperatorMetadata(speculative_safe=False, requires_final=True),
        ),
        ActionOperator(
            name="CheckBalance",
            effects=_declared(_check_balance, (BALANCE_KEY,)),
            cost=1.0,
            executor=_executor(_check_balance),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="CancelTransfer",
            preconditions=(COND_STATE,),
            effects=_declared(_cancel, (TRANSACTION_KEY, PREPARED_KEY, CONFIRMED_KEY, AUTHORIZED_KEY,
                                        LIMIT_KEY, RECEIPT_KEY)),
            cost=1.0,
            executor=_executor(_cancel, _require_pending),
            satisfies_goals=("CANCEL",),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="ConfirmTransfer",
            preconditions=(COND_PREPARED, COND_AUTHORIZED),
            effects=_declared(_confirm, (CONFIRMED_KEY,)),
            cost=1.0,
            executor=_executor(_confirm, _require_prepared),
            satisfies_goals=("CONFIRM",),
            metadata=OperatorMetadata(speculative_safe=False, requires_final=True),
        ),
    )


def transfer_goal_schemas() -> Mapping[str, GoalSchema]:
    return {
        # Satisfaction is "prepared and explicitly confirmed", never "executed". The
        # controller owns irreversible execution outside the planner.
        "TRANSFER": GoalSchema(
            "TRANSFER",
            slots=(
                SlotSpec("recipient", "Who does the user want to send money to?"),
                SlotSpec("amount", "How much money does the user want to send?", kind=SlotKind.NUMBER),
                SlotSpec("currency", "Which three-letter currency code does the user want to send, if one is stated?",
                         required=False, confidence_threshold=0.6, validator=_looks_like_currency),
            ),
            # Satisfaction means the requested transfer is prepared: same recipient and
            # amount, not merely "a transfer exists". Preparation, not execution -
            # confirmation and bank execution are separate explicit steps the planner
            # is not allowed to take.
            satisfied_when=(COND_MATCHES_GOAL,),
        ),
        "CHECK_BALANCE": GoalSchema(
            "CHECK_BALANCE",
            slots=(SlotSpec("account", "Which account does the user want to check?", required=False),),
            satisfied_when=(Condition(BALANCE_KEY, Compare.TRUTHY),),
        ),
        "CANCEL": GoalSchema("CANCEL", command=True,
                             already_done_when=(Condition(TRANSACTION_KEY, predicate=_already_cancelled),)),
        "CONFIRM": GoalSchema("CONFIRM", command=True, already_done_when=(COND_CONFIRMED_FOR_THIS,)),
    }


def transfer_goal_types() -> tuple[str, ...]:
    """Goal types are declared once, in `transfer_goal_schemas`; this is a convenience view."""
    return tuple(transfer_goal_schemas())


class TransferCapability:
    """Bundles transfer Local Jev goals, schemas, and operators."""

    name = "Transfer"
    description = ("Sending money: transfer an amount to a recipient, check an account "
                   "balance, cancel a pending transfer, or confirm a prepared transfer.")

    world_schema = staticmethod(transfer_world_schema)

    def __init__(self, local_jev: Any) -> None:
        self.local_jev = local_jev

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(name=self.name, description=self.description,
                                    local_jev=self.local_jev,
                                    goal_schemas=transfer_goal_schemas(),
                                    operators=transfer_operators(),
                                    world_schema=transfer_world_schema(),
                                    # The spine drops a pending confirmation when this
                                    # changes, so a stale yes cannot authorize a new transfer.
                                    confirmation_subject=lambda world: transfer_subject(_state(world)))


def build_transfer_descriptor(local_jev: Any) -> CapabilityDescriptor:
    return TransferCapability(local_jev).descriptor()

"""Session-scoped confirmation binding for irreversible actions.

An explicit confirmation is only meaningful for the *exact* action the user was
asked about, in the *same* conversational context. This module ports that rule from
the older transaction path (`ConversationState.activate_confirmation` /
`confirms`, which bound a confirmation to `(task_id, task_version, topic_epoch)`)
into a capability-agnostic form.

Two bindings must hold for a confirmation to authorize execution:

* **Subject** - the action the user was asked about must be the action about to run.
  The ledger stores the subject that was presented; a capability recomputes the
  subject from current state, so a material edit in between (a changed amount or
  recipient) changes the subject and the stale confirmation is refused.
* **Topic epoch** - switching capability or changing topic invalidates the
  confirmation, so "confirm" after an unrelated detour cannot authorize anything.

The ledger is deliberately small and mutable, and lives in world state by reference
like the transaction controller. That is what lets an epoch change survive the
executor replacing world state after every step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ping_ponder.observability import emit

logger = logging.getLogger(__name__)

# World key holding the session ledger, so operators and the executor can consult it.
CONFIRMATION_LEDGER_KEY = "session.confirmation_ledger"


@dataclass(frozen=True)
class ConfirmationContext:
    """What the user was asked to confirm, and the context they were asked in."""

    capability: str
    subject: str
    topic_epoch: int

    def describe(self) -> str:
        return f"{self.capability}[{self.subject}]@epoch{self.topic_epoch}"


class ConfirmationLedger:
    """Tracks the one action currently presented to the user for confirmation."""

    def __init__(self) -> None:
        self._context: ConfirmationContext | None = None
        self._topic_epoch = 0

    @property
    def topic_epoch(self) -> int:
        return self._topic_epoch

    def current(self) -> ConfirmationContext | None:
        return self._context

    def change_topic(self) -> None:
        """Enter a new topic/capability. Any prior confirmation becomes unusable."""
        self._topic_epoch += 1
        if self._context is not None:
            emit(logger, "confirmation_invalidated", reason="topic_change",
                 context=self._context.describe(), topic_epoch=self._topic_epoch)
        self._context = None

    def activate(self, capability: str, subject: str) -> ConfirmationContext:
        """Record the action currently presented to the user.

        Called by the host after the assistant asks the confirmation question, i.e.
        when it is legitimate for a following "yes" to authorize this exact action.
        """
        if not capability or not subject:
            raise ValueError("a confirmation requires a capability and a subject")
        context = ConfirmationContext(capability=capability, subject=subject,
                                      topic_epoch=self._topic_epoch)
        self._context = context
        emit(logger, "confirmation_activated", context=context.describe())
        return context

    def confirms(self, capability: str, subject: str | None) -> bool:
        """Whether an explicit confirmation may authorize this exact action now.

        Refused when nothing was presented, when the presented action differs, or when
        the topic epoch has moved on.
        """
        if subject is None:
            return False
        context = self._context
        if context is None:
            return False
        if context.topic_epoch != self._topic_epoch:
            emit(logger, "confirmation_refused", reason="topic_epoch_changed",
                 presented=context.describe(), topic_epoch=self._topic_epoch)
            return False
        if context.capability != capability or context.subject != subject:
            emit(logger, "confirmation_refused", reason="subject_changed",
                 presented=context.describe(), requested=f"{capability}[{subject}]")
            return False
        return True

    def refresh_if_changed(self, capability: str, subject: str | None) -> None:
        """Drop a pending confirmation whose presented action no longer exists.

        A materially edited action is no longer what the user was asked about, so its
        confirmation must not stay live. This is the counterpart to `change_topic` for
        edits within the same topic.
        """
        context = self._context
        if context is None:
            return
        if context.capability != capability or context.subject != subject:
            emit(logger, "confirmation_invalidated", reason="subject_changed",
                 presented=context.describe(), current=f"{capability}[{subject}]")
            self._context = None

    def clear(self) -> None:
        self._context = None

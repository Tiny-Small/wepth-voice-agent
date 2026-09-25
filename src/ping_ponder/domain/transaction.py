from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class TransactionIntent(StrEnum):
    TRANSFER = "TRANSFER"


class TransactionStatus(StrEnum):
    COLLECTING = "COLLECTING"
    PREPARING = "PREPARING"
    AWAITING_AUTHORIZATION = "AWAITING_AUTHORIZATION"
    AUTHORIZED = "AUTHORIZED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


MATERIAL_FIELDS = frozenset({"intent", "recipient_text", "recipient_id", "amount", "currency", "source_account"})


class TransactionState(BaseModel):
    task_id: str
    version: int = Field(default=1, ge=1)
    intent: TransactionIntent = TransactionIntent.TRANSFER
    recipient_text: str | None = None
    recipient_id: str | None = None
    amount: Decimal | None = None
    currency: str = "SGD"
    source_account: str | None = None
    status: TransactionStatus = TransactionStatus.COLLECTING
    authorized_version: int | None = None

    @field_validator("amount")
    @classmethod
    def positive_amount(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and (not value.is_finite() or value <= 0):
            raise ValueError("amount must be finite and positive")
        return value

    def is_complete(self) -> bool:
        return bool(self.recipient_id and self.source_account and self.amount is not None and self.currency)

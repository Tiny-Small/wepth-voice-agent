"""Bounded queue that drops the oldest item under backpressure."""

import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be positive")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
        self.dropped = 0

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    def put_nowait(self, item: T) -> None:
        if self._queue.full():
            self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()

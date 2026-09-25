"""The microphone buffer's shared queue keeps its existing drop-oldest policy."""

import importlib.util

import pytest


@pytest.mark.asyncio
async def test_shared_queue_drops_oldest_frame_and_tracks_count():
    assert importlib.util.find_spec("ping_ponder.queueing") is not None
    from ping_ponder.queueing import DropOldestQueue

    queue = DropOldestQueue[int](2)
    queue.put_nowait(1)
    queue.put_nowait(2)
    queue.put_nowait(3)

    assert queue.depth == 2
    assert queue.dropped == 1
    assert await queue.get() == 2
    assert await queue.get() == 3

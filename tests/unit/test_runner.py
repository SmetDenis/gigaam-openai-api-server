"""Тесты Runner: сериализация (1 за раз), backpressure (QueueFullError), shutdown."""

import asyncio
import threading
import time

import pytest

from gigaam_api.runner import QueueFullError, Runner

pytestmark = pytest.mark.asyncio


async def test_run_executes_and_returns_result() -> None:
    runner = Runner(max_queue=4)
    try:
        result = await runner.run(lambda x: x + 1, 41)
        assert result == 42
    finally:
        runner.shutdown()


async def test_run_serializes_one_at_a_time() -> None:
    runner = Runner(max_queue=8)
    state: dict[str, int] = {"concurrent": 0, "max": 0}

    def work(_: int) -> int:
        # Безопасно: single-worker гарантирует serial execution — реального параллелизма нет.
        state["concurrent"] += 1
        state["max"] = max(state["max"], state["concurrent"])
        time.sleep(0.02)
        state["concurrent"] -= 1
        return 0

    try:
        await asyncio.gather(*[runner.run(work, i) for i in range(4)])
        assert state["max"] == 1  # одновременно не более одного
    finally:
        runner.shutdown()


async def test_queue_full_raises_when_over_limit() -> None:
    runner = Runner(max_queue=1)
    started = threading.Event()
    release = threading.Event()

    def blocker() -> str:
        started.set()
        release.wait(timeout=5)
        return "done"

    task = asyncio.create_task(runner.run(blocker))
    try:
        await asyncio.sleep(0)  # дать задаче стартовать и занять единственный слот
        await asyncio.to_thread(started.wait, 5)  # дождаться, пока единственный слот занят
        with pytest.raises(QueueFullError):
            await runner.run(lambda: "second")
    finally:
        release.set()
        assert await task == "done"
        runner.shutdown()

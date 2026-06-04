"""Runner tests: serialization (one at a time), backpressure (QueueFullError), shutdown."""

import asyncio
import threading
import time

import pytest

from gigaam_api.runner import QueueFullError, Runner


@pytest.mark.asyncio
async def test_run_executes_and_returns_result() -> None:
    runner = Runner(max_queue=4)
    try:
        result = await runner.run(lambda x: x + 1, 41)
        assert result == 42
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_run_serializes_one_at_a_time() -> None:
    runner = Runner(max_queue=8)
    state: dict[str, int] = {"concurrent": 0, "max": 0}

    def work(_: int) -> int:
        # Safe: a single worker guarantees serial execution — there is no real parallelism.
        state["concurrent"] += 1
        state["max"] = max(state["max"], state["concurrent"])
        time.sleep(0.02)
        state["concurrent"] -= 1
        return 0

    try:
        await asyncio.gather(*[runner.run(work, i) for i in range(4)])
        assert state["max"] == 1  # never more than one at a time
    finally:
        runner.shutdown()


def test_try_acquire_enforces_max_queue_and_release_frees() -> None:
    runner = Runner(max_queue=2)
    try:
        runner.try_acquire()
        runner.try_acquire()
        with pytest.raises(QueueFullError):
            runner.try_acquire()  # over the limit — backpressure
        runner.release()
        runner.try_acquire()  # slot freed → can acquire again
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_submit_runs_in_worker_and_returns_future() -> None:
    runner = Runner(max_queue=4)
    try:
        fut = runner.submit(lambda x: x * 2, 21)
        assert await asyncio.wrap_future(fut) == 42
    finally:
        runner.shutdown()


@pytest.mark.asyncio
async def test_submit_and_run_share_single_worker() -> None:
    runner = Runner(max_queue=8)
    order: list[str] = []

    def slow(tag: str) -> str:
        order.append(f"start:{tag}")
        time.sleep(0.02)
        order.append(f"end:{tag}")
        return tag

    try:
        fut = runner.submit(slow, "a")  # occupies the single worker
        await runner.run(slow, "b")  # must wait for "a" to finish (1 worker)
        await asyncio.wrap_future(fut)
        assert order == ["start:a", "end:a", "start:b", "end:b"]  # strictly serial
    finally:
        runner.shutdown()


@pytest.mark.asyncio
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
        await asyncio.sleep(0)  # let the task start and occupy the single slot
        await asyncio.to_thread(started.wait, 5)  # wait until the single slot is occupied
        with pytest.raises(QueueFullError):
            await runner.run(lambda: "second")
    finally:
        release.set()
        assert await task == "done"
        runner.shutdown()

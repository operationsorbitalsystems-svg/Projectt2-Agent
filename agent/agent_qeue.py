"""
agent_queue.py
──────────────
Task queue for the COA invoice classification agent.

Mirrors the LLMQueue pattern but each task runs a full agentic loop
(multi-turn Bedrock conversation + tree navigation) instead of a single LLM call.

What varies per task:
  - expenses_tree  : the COA Expenses subtree (can differ per tenant / company)
  - line_item      : a single invoice line item string to classify
  - vendor_name    : optional vendor context

One line item  →  one task  →  one agent run  →  one selected ledger name.
Multiple line items from the same invoice share a batch_id.

Usage:
    queue = get_agent_queue()

    # Enqueue a batch of line items
    task_ids = []
    for item in invoice_line_items:
        tid = await queue.enqueue_request(
            batch_id="INV-2025-001",
            line_item=item["description"],
            expenses_tree=EXPENSES_TREE,
            vendor_name=item.get("vendor"),
            metadata={"line_item_id": item["id"]},
        )
        task_ids.append(tid)

    # Wait for results
    for tid in task_ids:
        response = await queue.wait_for_response(tid)
        print(response.selected_leaf, response.success)
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel

from config import redis_client
from utils.logger import setup_logger, batch_id_var
from .agent import run_agent          # the existing agentic loop

from redis import Redis

logger = setup_logger()


# ── Data models ───────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    task_id: str
    batch_id: str
    line_item: str                       # invoice line item description
    expenses_tree: Dict[str, Any]           # COA Expenses subtree (JSON-serialisable)
    vendor_name: Optional[str] = None
    enqueued_at: str
    metadata: Dict[str, Any] = {}
    file_name: str


class AgentResponse(BaseModel):
    task_id: str
    batch_id: str
    line_item: str
    selected_leaf: Optional[str]            # None on failure
    success: bool
    error: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = {}


# ── Queue ─────────────────────────────────────────────────────────────────────

class AgentQueue:
    """
    Redis-backed task queue for the COA classification agent.

    Key layout (mirrors LLMQueue):
        agent_queue:pending:{batch_id}   — list of serialised AgentRequest
        agent_queue:response:{task_id}  — serialised AgentResponse (TTL 1 h)
        agent_queue:processing:{task_id}— in-flight guard (TTL 1 h)
        agent_queue:active_batches      — set of batch_ids with pending work
        agent_queue:round_robin_index   — int for fair scheduling across batches
    """

    PENDING_PREFIX       = "agent_queue:pending:"
    RESPONSE_PREFIX      = "agent_queue:response:"
    PROCESSING_PREFIX    = "agent_queue:processing:"
    ACTIVE_BATCHES_KEY   = "agent_queue:active_batches"
    ROUND_ROBIN_KEY      = "agent_queue:round_robin_index"

    def __init__(self, redis_client: Redis):
        if redis_client is None:
            raise ValueError("Redis client is required")
        self.redis = redis_client
        logger.info("AgentQueue initialised")

    # ── Enqueue ───────────────────────────────────────────────────────────────

    async def enqueue_request(
        self,
        task_id: str,
        batch_id: str,
        line_item: str,
        file_name: str,
        expenses_tree: Dict[str, Any],
        vendor_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,

    ) -> str:
        """
        Add one line item to the queue.  Returns the task_id.
        """
        request = AgentRequest(
            task_id=task_id,
            batch_id=batch_id,
            line_item=line_item,
            expenses_tree=expenses_tree,
            vendor_name=vendor_name,
            enqueued_at=datetime.utcnow().isoformat(),
            file_name=file_name,
            metadata=metadata or {},
        )

        queue_key = f"{self.PENDING_PREFIX}{batch_id}"
        await self.redis.rpush(queue_key, request.model_dump_json())
        await self.redis.sadd(self.ACTIVE_BATCHES_KEY, batch_id)

        logger.info(f"📝 Enqueued agent task {request.task_id} for batch {batch_id}")
        return request.task_id


    # ── Scheduling ────────────────────────────────────────────────────────────

    async def _get_next_task(self) -> Optional[AgentRequest]:
        """Round-robin dequeue across active batches."""
        active_batches = await self.redis.smembers(self.ACTIVE_BATCHES_KEY)
        if not active_batches:
            return None

        batches = sorted(list(active_batches))
        index_str = await self.redis.get(self.ROUND_ROBIN_KEY)
        index = int(index_str) if index_str else 0

        for offset in range(len(batches)):
            selected_index = (index + offset) % len(batches)
            selected_batch = batches[selected_index]

            queue_key = f"{self.PENDING_PREFIX}{selected_batch}"
            request_json = await self.redis.lpop(queue_key)

            if request_json:
                next_index = (selected_index + 1) % len(batches)
                await self.redis.set(self.ROUND_ROBIN_KEY, next_index)

                request = AgentRequest.model_validate_json(request_json)

                # Clean up empty batches
                if await self.redis.llen(queue_key) == 0:
                    await self.redis.srem(self.ACTIVE_BATCHES_KEY, selected_batch)

                logger.info(
                    f"⚡ Dequeued agent task {request.task_id} "
                    f"from batch {selected_batch}"
                )
                return request

        return None

    # ── Execution ─────────────────────────────────────────────────────────────

    async def _run_task(self, request: AgentRequest) -> AgentResponse:
        """
        Run the full agentic loop for one line item.
        run_agent() is synchronous (boto3 + blocking I/O), so we
        offload it to a thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        try:
            selected_leaf = await loop.run_in_executor(
                None,               # default ThreadPoolExecutor
                run_agent,          # the existing agent entry point
                request.line_item,
                request.expenses_tree,  # passed through to run_agent
                request.batch_id,
                request.task_id,
                request.vendor_name,
            )
            return AgentResponse(
                task_id=request.task_id,
                batch_id=request.batch_id,
                line_item=request.line_item,
                selected_leaf=selected_leaf,
                success=True,
                completed_at=datetime.utcnow().isoformat(),
                metadata=request.metadata,
            )
        except Exception as e:
            logger.error(
                f"❌ Agent error for task {request.task_id}: {e}"
            )
            return AgentResponse(
                task_id=request.task_id,
                batch_id=request.batch_id,
                line_item=request.line_item,
                selected_leaf=None,
                success=False,
                error=str(e),
                completed_at=datetime.utcnow().isoformat(),
                metadata=request.metadata,
            )

    # ── Worker loop ───────────────────────────────────────────────────────────

    async def worker_loop(self):
        """
        Single worker coroutine.  Spawn multiple of these for parallelism:

            await asyncio.gather(
                queue.worker_loop(),
                queue.worker_loop(),
                queue.worker_loop(),
            )

        Or use start_workers() below.
        """
        logger.info("🚀 AgentQueue worker started")

        while True:
            _ctx_token = None
            try:
                request = await self._get_next_task()

                if request is None:
                    await asyncio.sleep(1)
                    continue

                _ctx_token = batch_id_var.set(request.batch_id)

                # Mark as in-flight
                processing_key = f"{self.PROCESSING_PREFIX}{request.task_id}"
                await self.redis.setex(
                    processing_key, 3600, request.model_dump_json()
                )

                logger.info(
                    f"🤖 Running agent for task {request.task_id}: "
                    f"{request.line_item[:80]!r}"
                )

                response = await self._run_task(request)

                # Store result
                response_key = f"{self.RESPONSE_PREFIX}{request.task_id}"
                await self.redis.setex(response_key, 3600, response.model_dump_json())

                # Clear in-flight marker
                await self.redis.delete(processing_key)

                logger.info(
                    f"✅ Task {request.task_id} done — "
                    f"leaf={response.selected_leaf!r} success={response.success}"
                )

            except Exception as e:
                logger.error(f"❌ AgentQueue worker error: {e}")
                await asyncio.sleep(5)

            finally:
                if _ctx_token is not None:
                    batch_id_var.reset(_ctx_token)

    def start_workers(self, n: int = 5):
        """
        Spawn `n` worker coroutines as background Tasks — returns immediately.
        NOT async. Call it plain: agent_queue.start_workers(3)
        """
        logger.info(f"🚀 Starting {n} AgentQueue workers")
        for i in range(n):
            asyncio.create_task(self.worker_loop(), name=f"agent-worker-{i}")

    # ── Response retrieval ────────────────────────────────────────────────────

    async def wait_for_response(
        self,
        task_id: str,
        timeout: int = 300,         # agent runs take longer than single LLM calls
    ) -> AgentResponse:
        """Poll until the response is available or timeout is reached."""
        response_key = f"{self.RESPONSE_PREFIX}{task_id}"
        start = asyncio.get_event_loop().time()

        while True:
            response_json = await self.redis.get(response_key)
            if response_json:
                logger.info(f"📬 Retrieved agent response for task {task_id}")
                return AgentResponse.model_validate_json(response_json)

            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Agent task {task_id} timed out after {timeout}s"
                )

            await asyncio.sleep(1)   # poll every second (agent turns take time)

    # ── Introspection ─────────────────────────────────────────────────────────

    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Quick status check for a batch.
        Returns counts of pending / processing / completed tasks.
        """
        pending = await self.redis.llen(f"{self.PENDING_PREFIX}{batch_id}")

        processing_keys = await self.redis.keys(f"{self.PROCESSING_PREFIX}*")
        processing = 0
        for key in processing_keys:
            raw = await self.redis.get(key)
            if raw:
                try:
                    req = AgentRequest.model_validate_json(raw)
                    if req.batch_id == batch_id:
                        processing += 1
                except Exception:
                    pass

        return {
            "batch_id": batch_id,
            "pending": pending,
            "processing": processing,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent_queue: Optional[AgentQueue] = None


def get_agent_queue() -> AgentQueue:
    global _agent_queue
    if _agent_queue is None:
        _agent_queue = AgentQueue(redis_client)
    return _agent_queue
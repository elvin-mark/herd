import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from herd.api.state import manager
from herd.services.agent import AgentEventListener, AgentSession

router = APIRouter()


class AgentRunRequest(BaseModel):
    objective: str
    model: Optional[str] = None
    max_turns: int = 10
    yolo: bool = True
    use_memory: bool = True


class WebAgentEventListener(AgentEventListener):
    """Pushes agent lifecycle events to an asyncio Queue for real-time SSE streaming."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def _emit(self, event_type: str, payload: Dict[str, Any]):
        msg = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def on_turn_start(self, turn: int, max_turns: int):
        self._emit("turn_start", {"turn": turn, "max_turns": max_turns})

    def on_plan_update(self, plan_summary: str):
        self._emit("plan_update", {"plan": plan_summary})

    def on_thought(self, thought: str):
        self._emit("thought", {"thought": thought})

    def on_action(self, action: str, action_input: Any):
        self._emit("action", {"action": action, "action_input": action_input})

    def on_observation(self, observation: str):
        self._emit("observation", {"observation": observation})

    def on_compaction(self, msg_count: int, orig_chars: int, comp_chars: int):
        self._emit(
            "compaction",
            {
                "msg_count": msg_count,
                "orig_chars": orig_chars,
                "comp_chars": comp_chars,
            },
        )

    def on_finish(self, answer: str):
        self._emit("finish", {"answer": answer})

    def on_error(self, error_msg: str):
        self._emit("error", {"error": error_msg})


@router.post("/v1/agent/run")
async def run_agent_stream(req: AgentRunRequest):
    """Executes an agent task objective and streams real-time step events via Server-Sent Events (SSE)."""
    # 1. Resolve model
    target_model = req.model
    if not target_model or target_model == "auto":
        running = list(manager.running_models.keys())
        if running:
            target_model = manager.running_models[running[0]]["model_name"]
        else:
            from herd.core.config import DEFAULT_LLM

            target_model = DEFAULT_LLM

    # Ensure model is started
    port = await manager.get_or_start_server(target_model)
    gateway_url = f"http://127.0.0.1:{port}"

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    listener = WebAgentEventListener(queue, loop)

    session = AgentSession(
        model_name=target_model,
        gateway_url=gateway_url,
        yolo=req.yolo,
        use_memory=req.use_memory,
        listener=listener,
    )

    async def event_generator():
        # Run agent task in background worker thread
        task = asyncio.create_task(
            asyncio.to_thread(session.run_task, req.objective, req.max_turns)
        )

        try:
            while not task.done() or not queue.empty():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
            yield "event: done\ndata: {}\n\n"
        except asyncio.CancelledError:
            task.cancel()
            yield "event: cancelled\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

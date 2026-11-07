"""Minimal tool-using agent loop over a text-only LLM port."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "AgentAction",
    "AgentError",
    "AgentResult",
    "Budget",
    "FakeLLM",
    "LLMPort",
    "Message",
    "RecordedLLM",
    "RecordingExhaustedError",
    "StepRecord",
    "Tool",
    "__version__",
    "render_system_prompt",
    "run_agent",
]

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ALLOWED_TYPES = {"str", "int", "float", "bool"}


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@runtime_checkable
class LLMPort(Protocol):
    """Protocol for text LLM completion."""

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        """Complete the given messages, returning raw text reply."""
        ...


class RecordingExhaustedError(RuntimeError):
    """Raised when RecordedLLM has exhausted all pre-recorded replies."""


class RecordedLLM:
    """Test double returning pre-recorded replies in sequence."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        self.calls.append(list(messages))
        if self._index >= len(self._replies):
            raise RecordingExhaustedError("Recorded LLM replies exhausted")
        reply = self._replies[self._index]
        self._index += 1
        return reply


class FakeLLM:
    """Test double delegating replies to a callable."""

    def __init__(self, fn: Callable[[Sequence[Message]], str]) -> None:
        self._fn = fn
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, max_tokens: int) -> str:
        self.calls.append(list(messages))
        return self._fn(messages)


@dataclass(frozen=True)
class Tool:
    """Tool definition available to the agent."""

    name: str
    description: str
    fn: Callable[..., Any]
    params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not TOOL_NAME_RE.match(self.name):
            raise AgentError(
                "invalid_tools",
                f"Invalid tool name '{self.name}'. Tool names must match ^[a-z][a-z0-9_]{{0,63}}$",
            )
        for p_name, p_type in self.params.items():
            if p_type not in ALLOWED_TYPES:
                msg = (
                    f"Invalid parameter type '{p_type}' for parameter '{p_name}'. "
                    f"Allowed: {ALLOWED_TYPES}"
                )
                raise AgentError("invalid_tools", msg)


@dataclass(frozen=True)
class Budget:
    """Budget constraints for execution of the agent loop."""

    max_steps: int = 10
    max_seconds: float = 60.0
    max_context_chars: int = 100_000
    max_consecutive_errors: int = 3
    max_invalid_replies: int = 2
    max_result_chars: int = 10_000
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if (
            self.max_steps < 1
            or self.max_seconds < 1
            or self.max_context_chars < 1
            or self.max_consecutive_errors < 1
            or self.max_invalid_replies < 1
            or self.max_result_chars < 1
            or self.max_tokens < 1
        ):
            raise AgentError("invalid_budget", "Budget parameters must be >= 1")


@dataclass(frozen=True)
class StepRecord:
    """Execution details of a single agent step."""

    step: int
    kind: Literal["tool", "final", "invalid"]
    tool: str | None
    outcome: str
    duration_ms: int


@dataclass(frozen=True)
class AgentResult:
    """Final result returned by run_agent."""

    status: Literal[
        "final",
        "max_steps",
        "deadline",
        "loop_detected",
        "tool_errors",
        "invalid_replies",
        "context_exhausted",
    ]
    final: str | None
    steps: tuple[StepRecord, ...]


class AgentError(Exception):
    """Base exception for misconfiguration or invalid usage."""

    def __init__(self, code: Literal["invalid_tools", "invalid_budget"], message: str) -> None:
        super().__init__(message[:200])
        self.code = code


class AgentAction:
    """Helper holding parsed reply action."""

    def __init__(
        self,
        is_final: bool,
        final_text: str | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        self.is_final = is_final
        self.final_text = final_text
        self.tool_name = tool_name
        self.tool_args = tool_args


def render_system_prompt(tools: Sequence[Tool]) -> str:
    """Render system prompt listing available tools and response protocol."""
    prompt = (
        "You are an autonomous assistant that solves tasks by selecting tools "
        "or answering directly.\n"
        "To act, reply ONLY with a single JSON object (optionally enclosed in a code fence) "
        "in one of these two shapes:\n"
        '1. {"tool": "<name>", "args": {<arguments>}}\n'
        '2. {"final": "<answer>"}\n\n'
        "Available tools:\n"
    )
    if not tools:
        prompt += "(None)\n"
        return prompt

    for tool in tools:
        params_str = ", ".join(f"{p}: {t}" for p, t in tool.params.items())
        prompt += f"- {tool.name}({params_str}): {tool.description}\n"
    return prompt


def _strip_fence(reply: str) -> str:
    text = reply.strip()
    if text.startswith("```"):
        first_line_end = text.find("\n")
        if first_line_end != -1:
            last_fence = text.rfind("```")
            if last_fence > first_line_end:
                return text[first_line_end + 1 : last_fence].strip()
    return text


def _parse_reply(reply: str) -> AgentAction | None:
    cleaned = _strip_fence(reply)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    has_final = "final" in parsed
    has_tool = "tool" in parsed

    if has_final and has_tool:
        return None

    if has_final:
        final_val = parsed["final"]
        if isinstance(final_val, str) and len(parsed) == 1:
            return AgentAction(is_final=True, final_text=final_val)
        return None

    if has_tool:
        tool_val = parsed["tool"]
        if not isinstance(tool_val, str) or "args" not in parsed or len(parsed) != 2:
            return None
        args_val = parsed["args"]
        if not isinstance(args_val, dict):
            return None
        return AgentAction(is_final=False, tool_name=tool_val, tool_args=args_val)

    return None


def _validate_args(args: dict[str, Any], params: Mapping[str, str]) -> bool:
    if set(args.keys()) != set(params.keys()):
        return False
    for name, expected_type in params.items():
        val = args[name]
        if expected_type == "bool":
            if not isinstance(val, bool):
                return False
        elif expected_type == "int":
            # in Python, isinstance(True, int) is True, but bool is not int for our spec
            if isinstance(val, bool) or not isinstance(val, int):
                return False
        elif expected_type == "float":
            # Note: ints could strictly not be float, or float strictly float. isinstance(x, float)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                return False
        elif expected_type == "str":
            if not isinstance(val, str):
                return False
        else:
            return False
    return True


def _escape_and_truncate_result(raw_result: Any, max_result_chars: int) -> str:
    text = str(raw_result)
    # Escaped by replacing '</tool_result' with '&lt;/tool_result' and the following '>' with '&gt;'
    escaped = re.sub(r"</tool_result([^>]*)>", r"&lt;/tool_result\1&gt;", text)
    if "</tool_result" in escaped:
        escaped = escaped.replace("</tool_result", "&lt;/tool_result")

    if len(escaped) > max_result_chars:
        return escaped[:max_result_chars] + "\n[truncated]"
    return escaped


DEFAULT_BUDGET = Budget()


def run_agent(
    llm: LLMPort,
    task: str,
    tools: Sequence[Tool],
    *,
    budget: Budget = DEFAULT_BUDGET,
    clock: Callable[[], float] = time.monotonic,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Execute an agent loop over the specified LLMPort until terminal condition."""
    seen_names: set[str] = set()
    tools_by_name: dict[str, Tool] = {}
    for t in tools:
        if t.name in seen_names:
            raise AgentError("invalid_tools", f"Duplicate tool name '{t.name}'")
        seen_names.add(t.name)
        tools_by_name[t.name] = t

    system_prompt = render_system_prompt(tools)
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    step_records: list[StepRecord] = []
    consecutive_errors = 0
    consecutive_invalid = 0
    recent_actions: list[str] = []

    start_time = clock()
    prev_llm_time: float | None = None

    while True:
        now = clock()
        if prev_llm_time is not None and step_records:
            # Update previous step's duration
            last_record = step_records[-1]
            dur = int((now - prev_llm_time) * 1000)
            step_records[-1] = StepRecord(
                step=last_record.step,
                kind=last_record.kind,
                tool=last_record.tool,
                outcome=last_record.outcome,
                duration_ms=dur,
            )
            if on_event is not None:
                on_event(
                    {
                        "step": step_records[-1].step,
                        "kind": step_records[-1].kind,
                        "tool": step_records[-1].tool,
                        "outcome": step_records[-1].outcome,
                        "duration_ms": step_records[-1].duration_ms,
                    }
                )

        # 1. Check deadline
        if now - start_time >= budget.max_seconds:
            return AgentResult(status="deadline", final=None, steps=tuple(step_records))

        # 2. Check steps used
        if len(step_records) == budget.max_steps:
            return AgentResult(status="max_steps", final=None, steps=tuple(step_records))

        # 3. Check context
        total_context_chars = sum(len(m["content"]) for m in messages)
        if total_context_chars > budget.max_context_chars:
            return AgentResult(status="context_exhausted", final=None, steps=tuple(step_records))

        prev_llm_time = now
        step_num = len(step_records) + 1
        reply = llm.complete(messages, max_tokens=budget.max_tokens)

        action = _parse_reply(reply)
        if action is None:
            consecutive_invalid += 1
            recent_actions.clear()
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        'Reply with one JSON object: {"tool": "<name>", "args": {...}} '
                        'or {"final": "<answer>"}'
                    ),
                }
            )
            step_records.append(
                StepRecord(
                    step=step_num,
                    kind="invalid",
                    tool=None,
                    outcome="invalid_reply",
                    duration_ms=0,
                )
            )
            if consecutive_invalid >= budget.max_invalid_replies:
                if on_event is not None:
                    on_event(
                        {
                            "step": step_records[-1].step,
                            "kind": step_records[-1].kind,
                            "tool": step_records[-1].tool,
                            "outcome": step_records[-1].outcome,
                            "duration_ms": 0,
                        }
                    )
                return AgentResult(status="invalid_replies", final=None, steps=tuple(step_records))
            continue

        if action.is_final:
            step_records.append(
                StepRecord(
                    step=step_num,
                    kind="final",
                    tool=None,
                    outcome="final_answer",
                    duration_ms=0,
                )
            )
            if on_event is not None:
                on_event(
                    {
                        "step": step_records[-1].step,
                        "kind": step_records[-1].kind,
                        "tool": step_records[-1].tool,
                        "outcome": step_records[-1].outcome,
                        "duration_ms": 0,
                    }
                )
            return AgentResult(status="final", final=action.final_text, steps=tuple(step_records))

        # Tool call
        tool_name = action.tool_name or ""
        tool_args = action.tool_args or {}
        messages.append({"role": "assistant", "content": reply})

        # Check loop detection
        action_sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
        recent_actions.append(action_sig)
        if (
            len(recent_actions) >= 3
            and recent_actions[-1] == recent_actions[-2] == recent_actions[-3]
        ):
            step_records.append(
                StepRecord(
                    step=step_num,
                    kind="tool",
                    tool=tool_name,
                    outcome="loop_detected",
                    duration_ms=0,
                )
            )
            if on_event is not None:
                on_event(
                    {
                        "step": step_records[-1].step,
                        "kind": step_records[-1].kind,
                        "tool": step_records[-1].tool,
                        "outcome": step_records[-1].outcome,
                        "duration_ms": 0,
                    }
                )
            return AgentResult(status="loop_detected", final=None, steps=tuple(step_records))

        tool_obj = tools_by_name.get(tool_name)
        if tool_obj is None:
            consecutive_errors += 1
            tool_result_content = (
                f'<tool_result name="{tool_name}" error="unknown_tool">unknown tool</tool_result>'
            )
            messages.append({"role": "user", "content": tool_result_content})
            step_records.append(
                StepRecord(
                    step=step_num,
                    kind="tool",
                    tool=tool_name,
                    outcome="unknown_tool",
                    duration_ms=0,
                )
            )
        elif not _validate_args(tool_args, tool_obj.params):
            consecutive_errors += 1
            tool_result_content = (
                f'<tool_result name="{tool_name}" error="invalid_args">invalid args</tool_result>'
            )
            messages.append({"role": "user", "content": tool_result_content})
            step_records.append(
                StepRecord(
                    step=step_num,
                    kind="tool",
                    tool=tool_name,
                    outcome="invalid_args",
                    duration_ms=0,
                )
            )
        else:
            # Execute tool
            try:
                res = tool_obj.fn(**tool_args)
                consecutive_errors = 0
                consecutive_invalid = 0
                formatted_res = _escape_and_truncate_result(res, budget.max_result_chars)
                tool_result_content = (
                    f'<tool_result name="{tool_name}">{formatted_res}</tool_result>'
                )
                messages.append({"role": "user", "content": tool_result_content})
                step_records.append(
                    StepRecord(
                        step=step_num,
                        kind="tool",
                        tool=tool_name,
                        outcome="success",
                        duration_ms=0,
                    )
                )
            except Exception as exc:
                consecutive_errors += 1
                err_msg = str(exc)[:200]
                err_text = f"{type(exc).__name__}: {err_msg}"
                tool_result_content = (
                    f'<tool_result name="{tool_name}" error="tool_failed">{err_text}</tool_result>'
                )
                messages.append({"role": "user", "content": tool_result_content})
                step_records.append(
                    StepRecord(
                        step=step_num,
                        kind="tool",
                        tool=tool_name,
                        outcome="tool_failed",
                        duration_ms=0,
                    )
                )

        if consecutive_errors >= budget.max_consecutive_errors:
            if on_event is not None:
                on_event(
                    {
                        "step": step_records[-1].step,
                        "kind": step_records[-1].kind,
                        "tool": step_records[-1].tool,
                        "outcome": step_records[-1].outcome,
                        "duration_ms": 0,
                    }
                )
            return AgentResult(status="tool_errors", final=None, steps=tuple(step_records))

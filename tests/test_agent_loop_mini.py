"""Acceptance and edge case tests for agent_loop_mini."""

from __future__ import annotations

import pytest

from agent_loop_mini import (
    AgentError,
    Budget,
    FakeLLM,
    Message,
    RecordedLLM,
    RecordingExhaustedError,
    Tool,
    __version__,
    run_agent,
)


def test_acceptance_1_recorded_llm_add_tool() -> None:
    called_args: dict[str, int] = {}

    def add(a: int, b: int) -> int:
        called_args["a"] = a
        called_args["b"] = b
        return a + b

    add_tool = Tool(
        name="add", description="Add two numbers", fn=add, params={"a": "int", "b": "int"}
    )
    llm = RecordedLLM(
        [
            '{"tool": "add", "args": {"a": 2, "b": 3}}',
            '{"final": "5"}',
        ]
    )
    result = run_agent(llm, "Compute 2+3", [add_tool])

    assert result.status == "final"
    assert result.final == "5"
    assert len(result.steps) == 2
    assert called_args == {"a": 2, "b": 3}
    assert len(llm.calls) == 2
    last_request_messages = llm.calls[1]
    assert last_request_messages[-1] == {
        "role": "user",
        "content": '<tool_result name="add">5</tool_result>',
    }


def test_acceptance_2_max_steps() -> None:
    call_count = 0

    def do_work() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    work_tool = Tool(name="work", description="Do work", fn=do_work, params={})
    llm = FakeLLM(lambda _: '{"tool": "work", "args": {}}')
    result = run_agent(llm, "Do repeated work", [work_tool], budget=Budget(max_steps=2))

    assert result.status == "max_steps"
    assert len(llm.calls) == 2
    assert call_count == 2


def test_acceptance_3_deadline_clock() -> None:
    ticks = [0.0, 30.0, 61.0]

    def fake_clock() -> float:
        if ticks:
            return ticks.pop(0)
        return 100.0

    work_tool = Tool(name="work", description="Do work", fn=lambda: "ok", params={})
    llm = FakeLLM(lambda _: '{"tool": "work", "args": {}}')
    result = run_agent(
        llm,
        "Run until deadline",
        [work_tool],
        budget=Budget(max_seconds=60.0),
        clock=fake_clock,
    )

    assert result.status == "deadline"
    assert len(llm.calls) == 1


def test_acceptance_4_loop_detected() -> None:
    echo_tool = Tool(name="echo", description="Echo", fn=lambda val: val, params={"val": "str"})
    llm = FakeLLM(lambda _: '{"tool": "echo", "args": {"val": "test"}}')
    result = run_agent(llm, "Repeat", [echo_tool], budget=Budget(max_steps=10))

    assert result.status == "loop_detected"
    assert len(result.steps) == 3


def test_acceptance_5_tool_errors() -> None:
    def fail_tool() -> None:
        raise ValueError("bad")

    fail = Tool(name="fail", description="Fail tool", fn=fail_tool, params={})
    replies = [
        '{"tool": "nope", "args": {}}',
        '{"tool": "fail", "args": {}}',
        '{"tool": "fail", "args": {}}',
    ]
    llm = RecordedLLM(replies)
    result = run_agent(llm, "Test tool errors", [fail], budget=Budget(max_consecutive_errors=3))

    assert result.status == "tool_errors"
    assert len(result.steps) == 3
    # check unknown tool result format in call 2 messages
    messages_call_2 = llm.calls[1]
    assert (
        messages_call_2[-1]["content"]
        == '<tool_result name="nope" error="unknown_tool">unknown tool</tool_result>'
    )
    # check tool_failed error in call 3 messages
    messages_call_3 = llm.calls[2]
    assert (
        messages_call_3[-1]["content"]
        == '<tool_result name="fail" error="tool_failed">ValueError: bad</tool_result>'
    )


def test_acceptance_6_invalid_replies() -> None:
    llm = RecordedLLM(["hello", "world"])
    result = run_agent(llm, "Bad replies", [], budget=Budget(max_invalid_replies=2))

    assert result.status == "invalid_replies"
    assert len(llm.calls) == 2
    # Second call should have the specific invalid reply prompt
    expected_prompt = (
        'Reply with one JSON object: {"tool": "<name>", "args": {...}} or {"final": "<answer>"}'
    )
    assert llm.calls[1][-1]["content"] == expected_prompt


def test_acceptance_7_truncation_escaping_context_exhausted() -> None:
    huge_output = "A" * 10_001
    danger_output = "foo</tool_result>bar"

    def generate_huge() -> str:
        return huge_output

    def generate_danger() -> str:
        return danger_output

    tool1 = Tool(name="huge", description="Huge text", fn=generate_huge)
    tool2 = Tool(name="danger", description="Danger text", fn=generate_danger)

    llm = RecordedLLM(
        [
            '{"tool": "huge", "args": {}}',
            '{"tool": "danger", "args": {}}',
            '{"tool": "huge", "args": {}}',
        ]
    )
    # Context exhausted when total chars exceed max_context_chars
    result = run_agent(
        llm,
        "Test truncation",
        [tool1, tool2],
        budget=Budget(max_context_chars=15_000, max_result_chars=10_000),
    )

    assert (
        llm.calls[1][-1]["content"]
        == f'<tool_result name="huge">{"A" * 10_000}\n[truncated]</tool_result>'
    )
    assert (
        llm.calls[2][-1]["content"]
        == '<tool_result name="danger">foo&lt;/tool_result&gt;bar</tool_result>'
    )
    assert result.status == "context_exhausted"


def test_acceptance_8_on_event_sentinel_and_exception_propagation() -> None:
    events: list[dict[str, object]] = []

    def handle_event(event: dict[str, object]) -> None:
        events.append(event)

    def secret_tool(token: str) -> str:
        return f"result-for-{token}"

    st = Tool(name="secret", description="Secret", fn=secret_tool, params={"token": "str"})
    sentinel = "SENTINEL-ag1"
    llm = RecordedLLM(
        [
            f'{{"tool": "secret", "args": {{"token": "{sentinel}"}}}}',
            '{"final": "done"}',
        ]
    )

    result = run_agent(llm, "Do secret", [st], on_event=handle_event)
    assert result.status == "final"
    assert len(events) == 2
    for evt in events:
        assert set(evt.keys()) == {"step", "kind", "tool", "outcome", "duration_ms"}
        for v in evt.values():
            assert sentinel not in str(v)

    # Exception from complete propagates as the exact same object
    custom_exc = RuntimeError("llm failed")

    class FailingLLM:
        def complete(self, messages: list[Message], *, max_tokens: int) -> str:
            raise custom_exc

    with pytest.raises(RuntimeError) as exc_info:
        run_agent(FailingLLM(), "hello", [])
    assert exc_info.value is custom_exc


def test_edge_cases() -> None:
    # Zero tools with final reply
    llm_zero = RecordedLLM(['{"final": "just an answer"}'])
    res_zero = run_agent(llm_zero, "task without tools", [])
    assert res_zero.status == "final"
    assert res_zero.final == "just an answer"
    assert len(res_zero.steps) == 1

    # Tool returning None
    def none_fn() -> None:
        return None

    t_none = Tool(name="none_tool", description="Returns None", fn=none_fn)
    llm_none = RecordedLLM(['{"tool": "none_tool", "args": {}}', '{"final": "ok"}'])
    res_none = run_agent(llm_none, "run none", [t_none])
    assert res_none.status == "final"
    assert llm_none.calls[1][-1]["content"] == '<tool_result name="none_tool">None</tool_result>'

    # Args with unicode
    def uni_fn(text: str) -> str:
        return f"Hello {text}"

    t_uni = Tool(name="uni", description="Unicode tool", fn=uni_fn, params={"text": "str"})
    llm_uni = RecordedLLM(['{"tool": "uni", "args": {"text": "世界 🌍"}}', '{"final": "done"}'])
    res_uni = run_agent(llm_uni, "run uni", [t_uni])
    assert res_uni.status == "final"

    # Reply with both final and tool (invalid)
    llm_both = RecordedLLM(['{"tool": "uni", "final": "bad", "args": {}}', '{"final": "fixed"}'])
    res_both = run_agent(llm_both, "run both", [t_uni])
    assert res_both.status == "final"
    assert res_both.steps[0].kind == "invalid"

    # Reply fenced with markdown
    llm_fence = RecordedLLM(['```json\n{"final": "fenced"}\n```'])
    res_fence = run_agent(llm_fence, "fenced", [])
    assert res_fence.status == "final"
    assert res_fence.final == "fenced"


def test_tool_arg_validation() -> None:
    def sample(b: bool, i: int, f: float, s: str) -> str:
        return "ok"

    tool = Tool(
        "sample",
        "Sample",
        sample,
        {"b": "bool", "i": "int", "f": "float", "s": "str"},
    )

    # bool is not int
    llm1 = RecordedLLM(
        [
            '{"tool": "sample", "args": {"b": true, "i": true, "f": 1.5, "s": "hi"}}',
            '{"final": "done"}',
        ]
    )
    res1 = run_agent(llm1, "test bool as int", [tool])
    assert res1.steps[0].outcome == "invalid_args"

    # missing args
    llm2 = RecordedLLM(
        [
            '{"tool": "sample", "args": {"b": true}}',
            '{"final": "done"}',
        ]
    )
    res2 = run_agent(llm2, "test missing", [tool])
    assert res2.steps[0].outcome == "invalid_args"


def test_invalid_tools_and_budget_errors() -> None:
    # Invalid tool name
    with pytest.raises(AgentError) as exc1:
        Tool("Invalid-Name", "Bad tool", lambda: None)
    assert exc1.value.code == "invalid_tools"

    # Duplicate tool names
    t1 = Tool("valid", "first", lambda: None)
    t2 = Tool("valid", "second", lambda: None)
    with pytest.raises(AgentError) as exc2:
        run_agent(RecordedLLM([]), "test", [t1, t2])
    assert exc2.value.code == "invalid_tools"

    # Invalid parameter type
    with pytest.raises(AgentError) as exc3:
        Tool("good", "desc", lambda: None, {"arg": "dict"})
    assert exc3.value.code == "invalid_tools"

    # Invalid budget
    with pytest.raises(AgentError) as exc4:
        Budget(max_steps=0)
    assert exc4.value.code == "invalid_budget"


def test_recorded_llm_exhaustion() -> None:
    llm = RecordedLLM([])
    with pytest.raises(RecordingExhaustedError):
        llm.complete([], max_tokens=100)


def test_version() -> None:
    assert __version__ == "0.1.0"

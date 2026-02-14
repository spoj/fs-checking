"""Generic async agent loop with pluggable model calls and tool execution.

Adapted from recon/src/agent_loop.py with one key change:
ToolExecutor is async, and all tool calls within a turn are executed
concurrently via asyncio.gather.

Design:
- ModelCallable: async function that takes messages and returns (assistant_message, usage)
- ToolExecutor: async function that executes a tool call and returns result string
- ValidationFn: optional function called when agent stops, returns (valid, error_msg)
- The loop continues until validation passes or max_iterations reached

Example usage:

    async def call_model(messages: list[dict]) -> tuple[dict, dict]:
        return assistant_message, usage

    async def execute_tool(name: str, args: dict) -> str:
        if name == "vision":
            return await call_flash(args["query"])
        return json.dumps({"error": f"Unknown tool: {name}"})

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=[...],
    )
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class AgentResult:
    """Result of an agent loop run."""

    success: bool
    final_message: str
    iterations: int
    messages: list[dict]
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }
    )
    tool_calls_count: int = 0


# Type aliases
ModelCallable = Callable[[list[dict]], Awaitable[tuple[dict, dict]]]
AsyncToolExecutor = Callable[[str, dict], Awaitable[str]]
ValidationFn = Callable[[], tuple[bool, str]]
OnIterationFn = Callable[[int, dict | None, list[dict] | None], None]


def accumulate_usage(total: dict, usage: dict) -> None:
    """Add usage from a response to the running total."""
    total["prompt_tokens"] += usage.get("prompt_tokens", 0)
    total["completion_tokens"] += usage.get("completion_tokens", 0)
    if usage.get("cost") is not None:
        total["cost"] = total.get("cost", 0.0) + float(usage["cost"])

    # Track reasoning tokens (OpenAI models)
    reasoning = 0
    ctd = usage.get("completion_tokens_details") or {}
    if isinstance(ctd, dict):
        reasoning = ctd.get("reasoning_tokens", 0) or 0
    total["reasoning_tokens"] = total.get("reasoning_tokens", 0) + reasoning

    # Track cached prompt tokens
    cached = 0
    ptd = usage.get("prompt_tokens_details") or {}
    if isinstance(ptd, dict):
        cached = ptd.get("cached_tokens", 0) or 0
    # Also check native_tokens_details (OpenRouter sometimes uses this)
    ntd = usage.get("native_tokens_details") or {}
    if isinstance(ntd, dict):
        cached = max(cached, (ntd.get("cached_tokens", 0) or 0))
    total["cached_tokens"] = total.get("cached_tokens", 0) + cached


async def run_agent_loop(
    call_model: ModelCallable,
    tool_executor: AsyncToolExecutor,
    initial_messages: list[dict],
    validation_fn: ValidationFn | None = None,
    max_iterations: int = 200,
    on_iteration: OnIterationFn | None = None,
) -> AgentResult:
    """Run an agent loop until completion or max iterations.

    All tool calls within a single turn are executed concurrently.

    Args:
        call_model: Async function (messages) -> (assistant_message, usage)
        tool_executor: Async function (tool_name, args) -> result_string.
            Called concurrently for all tool calls in a turn.
        initial_messages: Starting messages (system + user prompts)
        validation_fn: Optional () -> (valid, error_msg), called when agent stops
        max_iterations: Maximum iterations (0 = unlimited)
        on_iteration: Optional callback (iteration, assistant_msg, tool_results)

    Returns:
        AgentResult with success status, final message, usage stats
    """
    messages = list(initial_messages)
    total_usage: dict = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }
    tool_calls_count = 0
    iteration = 0

    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1

        assistant_message, usage = await call_model(messages)
        accumulate_usage(total_usage, usage)
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            # Parse all tool calls
            parsed = []
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                parsed.append((tc["id"], func_name, args))

            # Execute all tool calls concurrently
            async def _exec(call_id: str, name: str, args: dict) -> tuple[str, str]:
                result = await tool_executor(name, args)
                return call_id, result

            results = await asyncio.gather(
                *[_exec(cid, name, args) for cid, name, args in parsed],
                return_exceptions=True,
            )

            tool_results = []
            for r in results:
                if isinstance(r, BaseException):
                    # Shouldn't happen but be safe
                    call_id = parsed[results.index(r)][0]
                    result_str = json.dumps({"error": str(r)})
                else:
                    call_id, result_str = r

                tool_calls_count += 1
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_str,
                }
                tool_results.append(tool_msg)
                messages.append(tool_msg)

            if on_iteration:
                on_iteration(iteration, assistant_message, tool_results)
        else:
            # No tool calls — agent is done
            if on_iteration:
                on_iteration(iteration, assistant_message, None)

            content = assistant_message.get("content", "")

            if validation_fn:
                valid, error_msg = validation_fn()
                if valid:
                    return AgentResult(
                        success=True,
                        final_message=content,
                        iterations=iteration,
                        messages=messages,
                        usage=total_usage,
                        tool_calls_count=tool_calls_count,
                    )
                else:
                    messages.append(
                        {"role": "user", "content": f"Validation failed:\n{error_msg}"}
                    )
            else:
                return AgentResult(
                    success=True,
                    final_message=content,
                    iterations=iteration,
                    messages=messages,
                    usage=total_usage,
                    tool_calls_count=tool_calls_count,
                )

    return AgentResult(
        success=False,
        final_message="Max iterations reached without valid output",
        iterations=iteration,
        messages=messages,
        usage=total_usage,
        tool_calls_count=tool_calls_count,
    )

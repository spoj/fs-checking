"""Generic async agent loop with pluggable model calls and tool execution.

Adapted from recon/src/agent_loop.py with async tool support from ntdocs.

This is the core agent abstraction - use this as the foundation for any
LLM agent project. It provides:

- Async agent loop with tool call handling
- Both sync and async tool executor support
- Optional validation loop (re-prompts on failure)
- Usage tracking across iterations
- Clean separation: you provide callbacks, loop handles the rest

Example usage:

    from fs_checking.api import OpenRouterClient, create_model_callable

    client = OpenRouterClient()
    call_model = create_model_callable(client, "google/gemini-3-flash-preview", tools)

    async def execute_tool(name: str, args: dict) -> str:
        if name == "search":
            return json.dumps(search(args["query"]))
        return json.dumps({"error": f"Unknown tool: {name}"})

    def validate() -> tuple[bool, str]:
        # Check if output meets requirements
        return (True, "") if valid else (False, "Missing required field X")

    result = await run_agent_loop(
        call_model=call_model,
        tool_executor=execute_tool,
        initial_messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        validation_fn=validate,
    )

    if result.success:
        print(result.final_message)
"""

import json
import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any


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
            "cache_read_tokens": 0,
            "reasoning_tokens": 0,
        }
    )
    tool_calls_count: int = 0


# Type aliases for clarity
ModelCallable = Callable[[list[dict]], Awaitable[tuple[dict, dict]]]
SyncToolExecutor = Callable[[str, dict], str]
AsyncToolExecutor = Callable[[str, dict], Awaitable[str]]
ToolExecutor = SyncToolExecutor | AsyncToolExecutor
ValidationFn = Callable[[], tuple[bool, str]]
OnIterationFn = Callable[[int, dict | None, list[dict] | None], None]


def accumulate_usage(total: dict, usage: dict) -> None:
    """Add usage from a response to the running total."""
    total["prompt_tokens"] += usage.get("prompt_tokens", 0)
    total["completion_tokens"] += usage.get("completion_tokens", 0)

    prompt_details = usage.get("prompt_tokens_details", {})
    total["cache_read_tokens"] += prompt_details.get("cached_tokens", 0)

    completion_details = usage.get("completion_tokens_details", {})
    total["reasoning_tokens"] += completion_details.get("reasoning_tokens", 0)


async def _execute_tool_call(
    tc: dict,
    tool_executor: ToolExecutor,
    is_async: bool,
) -> dict:
    """Execute a single tool call, handling both sync and async executors."""
    func_name = tc["function"]["name"]
    try:
        args = json.loads(tc["function"]["arguments"])
    except json.JSONDecodeError:
        args = {}

    if is_async:
        result = await tool_executor(func_name, args)  # type: ignore[misc]
    else:
        result = tool_executor(func_name, args)  # type: ignore[misc]

    return {
        "role": "tool",
        "tool_call_id": tc["id"],
        "content": result,
    }


async def process_tool_calls(
    tool_calls: list[dict],
    tool_executor: ToolExecutor,
    parallel: bool = True,
) -> list[dict]:
    """Execute tool calls and return tool response messages.

    Args:
        tool_calls: List of tool call dicts from assistant message
        tool_executor: Function (name, args) -> result_string
        parallel: If True and executor is async, run tools in parallel

    Returns:
        List of tool response messages
    """
    is_async = inspect.iscoroutinefunction(tool_executor)

    if is_async and parallel and len(tool_calls) > 1:
        # Run all tool calls in parallel
        results = await asyncio.gather(
            *[_execute_tool_call(tc, tool_executor, is_async=True) for tc in tool_calls]
        )
        return list(results)
    else:
        # Run sequentially
        results = []
        for tc in tool_calls:
            result = await _execute_tool_call(tc, tool_executor, is_async)
            results.append(result)
        return results


async def run_agent_loop(
    call_model: ModelCallable,
    tool_executor: ToolExecutor,
    initial_messages: list[dict],
    validation_fn: ValidationFn | None = None,
    max_iterations: int = 200,
    on_iteration: OnIterationFn | None = None,
    parallel_tools: bool = True,
) -> AgentResult:
    """
    Run an agent loop until completion or max iterations.

    The loop continues until:
    1. Agent stops calling tools (returns final message)
    2. If validation_fn provided: validation passes
    3. max_iterations reached

    If validation fails, the error is fed back as a user message and
    the agent continues. This enables self-correction loops.

    Args:
        call_model: Async function (messages) -> (assistant_message, usage)
        tool_executor: Function (name, args) -> result_string (sync or async)
        initial_messages: Starting messages (system + user prompts)
        validation_fn: Optional () -> (valid, error_msg), called when agent stops
        max_iterations: Maximum iterations (0 = unlimited)
        on_iteration: Optional callback (iteration, assistant_msg, tool_results)
        parallel_tools: Run async tool calls in parallel (default True)

    Returns:
        AgentResult with success status, final message, usage stats
    """
    messages = list(initial_messages)
    total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_read_tokens": 0,
        "reasoning_tokens": 0,
    }
    tool_calls_count = 0
    iteration = 0

    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1

        # Call the model
        assistant_message, usage = await call_model(messages)
        accumulate_usage(total_usage, usage)

        # Add assistant message to history
        messages.append(assistant_message)

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if tool_calls:
            # Execute tools
            tool_results = await process_tool_calls(
                tool_calls, tool_executor, parallel=parallel_tools
            )
            tool_calls_count += len(tool_calls)
            messages.extend(tool_results)

            if on_iteration:
                on_iteration(iteration, assistant_message, tool_results)
        else:
            # No tool calls - agent is done
            if on_iteration:
                on_iteration(iteration, assistant_message, None)

            content = assistant_message.get("content", "")

            # Run validation if provided
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
                    # Validation failed - feed error back and continue
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Validation failed:\n{error_msg}",
                        }
                    )
            else:
                # No validation - just return
                return AgentResult(
                    success=True,
                    final_message=content,
                    iterations=iteration,
                    messages=messages,
                    usage=total_usage,
                    tool_calls_count=tool_calls_count,
                )

    # Max iterations reached
    return AgentResult(
        success=False,
        final_message="Max iterations reached without valid output",
        iterations=iteration,
        messages=messages,
        usage=total_usage,
        tool_calls_count=tool_calls_count,
    )

"""Swarm Strategy - Concurrent agents with shared state for financial statement checking.

Adapted from tech_pack_study/strategies/swarm.py for financial statement validation.

Key features:
- Multiple agents work CONCURRENTLY on shared JSON state
- JSON Patch (RFC 6902) for state modifications
- Hierarchical delegation with spawn(handle, prompt, pages)
- Sub-agent continuation with continue(handle, prompt)
- Scoped handles - no global namespace collision
- reasoning=high for reliable agent coordination

Check categories:
- cross_footing: Row/column totals verify
- internal_consistency: A=L+E, CF ties to BS, etc.
- note_ties: Note values match statement values
- period_comparison: YoY/QoQ variance reasonableness
- rounding: Consistent precision, no rounding errors

Usage:
    python -m scripts.cli document.pdf -o checks.json
"""

import asyncio
import base64
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonpatch

from .api import OpenRouterClient
from .pdf_utils import pdf_to_images

# === Configuration ===

MAX_CONCURRENT_TASKS = int(os.environ.get("SWARM_MAX_CONCURRENT", "20"))
MAX_ITERATIONS = int(os.environ.get("SWARM_MAX_ITERATIONS", "20"))
DEFAULT_MODEL = os.environ.get("FS_CHECK_MODEL", "google/gemini-3-flash-preview")

_semaphore: asyncio.Semaphore | None = None


def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    return _semaphore


# === JSON Patch ===


def resolve_path(obj: dict | list, path: str) -> tuple[Any, bool]:
    """Resolve JSON pointer path to value. Returns (value, exists)."""
    if path == "" or path == "/":
        return obj, True
    parts = path.strip("/").split("/")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return None, False
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    return None, False
                current = current[idx]
            except ValueError:
                return None, False
        else:
            return None, False
    return current, True


def apply_patch(state: dict, operations: list[dict]) -> tuple[bool, str]:
    """Apply JSON Patch operations to state."""
    try:
        patch = jsonpatch.JsonPatch(operations)
        result = patch.apply(state)
        state.clear()
        state.update(result)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# === Shared State ===


def get_structure(obj: Any) -> Any:
    """Get structure summary: keys for dicts, length for arrays, type for primitives."""
    if isinstance(obj, dict):
        return {k: get_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return f"[{len(obj)} items]"
    else:
        return type(obj).__name__


def safe_exec(code: str, local_vars: dict) -> Any:
    """Safely execute Python code with limited builtins."""
    safe_builtins = {
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "sorted": sorted,
        "reversed": reversed,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "range": range,
        "isinstance": isinstance,
        "type": type,
        "None": None,
        "True": True,
        "False": False,
    }
    return eval(code, {"__builtins__": safe_builtins}, local_vars)


def format_preview(value: Any, max_len: int = 15000) -> str:
    """Format value as JSON with truncation if needed."""
    try:
        result = json.dumps(value, indent=2, ensure_ascii=False)
        if len(result) > max_len:
            result = result[:max_len] + "\n... (truncated)"
        return result
    except (TypeError, ValueError):
        result = str(value)
        if len(result) > max_len:
            result = result[:max_len] + "... (truncated)"
        return result


class SharedState:
    """Thread-safe shared JSON state for all agents."""

    def __init__(self):
        self.state: dict = {
            "metadata": {},
            "checks": [],
            "values": {},
        }
        self.lock = asyncio.Lock()
        self.patch_log: list[dict] = []

    async def eval(self, code: str) -> str:
        """Read-only eval on state snapshot."""
        async with self.lock:
            snapshot = copy.deepcopy(self.state)

        result = safe_exec(code, {"state": snapshot})
        return format_preview(result)

    async def patch(self, agent_path: str, operations: list[dict]) -> tuple[bool, str]:
        async with self.lock:
            success, error = apply_patch(self.state, operations)
            self.patch_log.append(
                {
                    "agent": agent_path,
                    "ops": len(operations),
                    "success": success,
                    "error": error if not success else None,
                }
            )
            return success, error


# === Tool Definitions ===

TOOL_SPAWN = {
    "type": "function",
    "function": {
        "name": "spawn",
        "description": "Spawn a sub-agent for a subset of pages. Returns a handle you can use with continue(). Call multiple spawns in ONE turn for parallel execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Unique handle to reference this sub-agent (e.g., 'balance_sheet', 'income_statement')",
                },
                "prompt": {
                    "type": "string",
                    "description": "Instructions for what to check/extract",
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Page labels for sub-agent, e.g. ["Page 1", "Page 2"]',
                },
            },
            "required": ["handle", "prompt", "pages"],
        },
    },
}

TOOL_CONTINUE = {
    "type": "function",
    "function": {
        "name": "continue",
        "description": "Continue a previously spawned sub-agent with a follow-up prompt. The sub-agent retains its context and conversation history.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle of the sub-agent to continue",
                },
                "prompt": {
                    "type": "string",
                    "description": "Follow-up instruction or question",
                },
            },
            "required": ["handle", "prompt"],
        },
    },
}

TOOL_READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read state using Python expression. Variable 'state' is the shared dict.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": 'Python expression, e.g. \'state\', \'state.get("checks")\', \'len(state["checks"])\', \'[c for c in state["checks"] if c.get("status")=="fail"]\'',
                },
            },
            "required": ["expr"],
        },
    },
}

TOOL_PATCH = {
    "type": "function",
    "function": {
        "name": "patch",
        "description": "Apply JSON Patch operations to the shared state. Use to add checks and extracted values.",
        "parameters": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "JSON Patch operations (RFC 6902)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "add",
                                    "remove",
                                    "replace",
                                    "test",
                                    "move",
                                    "copy",
                                ],
                            },
                            "path": {"type": "string"},
                            "value": {},
                            "from": {"type": "string"},
                        },
                        "required": ["op", "path"],
                    },
                },
            },
            "required": ["operations"],
        },
    },
}

TOOL_CALC = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Evaluate arithmetic expression. Alias for read() with math focus.",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "e.g. '1234567 + 890123 - 45678' or 'abs(100 - 150)'",
                },
            },
            "required": ["expr"],
        },
    },
}


# === System Prompts ===

SYSTEM_PROMPT_CORE = """\
You are an agent in a SWARM. Multiple agents work concurrently on shared JSON state.

## Tools

**read(expr)** - Read state via Python expression. Variable `state` is the shared dict.
- `read("state")` - full state
- `read("state.get('checks')")` - subtree
- `read("list(state.keys())")` - structure discovery
- `read("len(state['checks'])")` - counts
- `read("[c for c in state['checks'] if c.get('status')=='fail']")` - filtering

**patch(operations)** - JSON Patch (RFC 6902):
- `{"op": "add", "path": "/key", "value": ...}` - add/create
- `{"op": "add", "path": "/array/-", "value": ...}` - append to array
- `{"op": "replace", "path": "/key", "value": ...}` - update existing
- `{"op": "remove", "path": "/key"}` - delete

**calc(expr)** - Arithmetic: `calc("1234 + 5678 - 90")`. Supports +, -, *, /, abs(), round(), sum().

## Rules

- read() FIRST before writing
- Use existing structure, don't recreate
- Append to arrays with "/-" for safe concurrency
- If patch fails, read() and adapt

Final message (no tool calls) = summary for parent.
"""

SYSTEM_PROMPT_DELEGATION = """
## Delegation

**spawn(handle, prompt, pages)** - Create sub-agent for pages
**continue(handle, prompt)** - Follow-up to existing sub-agent

**4+ pages → MUST delegate.** Spawns run in parallel = faster.

Spawn multiple in ONE turn:
```
spawn("bs", "Check balance sheet", ["Page 1", "Page 2"])
spawn("pl", "Check P&L", ["Page 3"])
```

After spawns, read() to verify. Use continue() for corrections.
"""

SYSTEM_PROMPT_ROOT = """
## Root Agent

You orchestrate sub-agents. Create workspace structures as needed:
```json
{"op": "add", "path": "/workspace", "value": {"ledger": [], "note_map": {}}}
```
Clean up when done: `{"op": "remove", "path": "/workspace"}`
"""

SYSTEM_PROMPT_INTROSPECTION = """
## Completion

Final message must include:

1. **Task Outcome**: Results summary, pass/fail counts, key findings.

2. **Introspection**: Reflect on what you learned:
   - What worked well in delegation/coordination?
   - What was inefficient or could be improved?
   - What changes to system/prompts/tools would help?
   - Patterns noticed that could inform future runs?

Be specific and actionable.
"""


def get_system_prompt(
    can_delegate: bool, is_root: bool = False, introspection: bool = True
) -> str:
    prompt = SYSTEM_PROMPT_CORE
    if can_delegate:
        prompt += SYSTEM_PROMPT_DELEGATION
    if is_root:
        prompt += SYSTEM_PROMPT_ROOT
        if introspection:
            prompt += SYSTEM_PROMPT_INTROSPECTION
    return prompt


def get_tools(can_delegate: bool) -> list[dict]:
    tools = [TOOL_READ, TOOL_PATCH, TOOL_CALC]
    if can_delegate:
        tools = [TOOL_SPAWN, TOOL_CONTINUE] + tools
    return tools


# === Agent Context ===


@dataclass
class AgentContext:
    """Context for a single agent in the swarm."""

    node: list[str]  # Page labels this agent can see
    all_pages: dict[str, bytes]  # All page images
    shared_state: SharedState
    model: str
    client: OpenRouterClient
    depth: int = 0
    call_path: str = "root"
    terminate_depth: int = 2  # Stop delegation at this depth
    introspection: bool = True  # Include introspection in root agent output
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    )
    sub_agents: dict[str, "AgentContext"] = field(default_factory=dict)

    def can_delegate(self) -> bool:
        return self.depth < self.terminate_depth

    def log(self, msg: str):
        print(f"[{self.call_path}] {msg}", file=sys.stderr)


# === Agent Execution ===


async def run_agent(
    ctx: AgentContext, prompt: str, is_continuation: bool = False
) -> dict:
    """Run agent loop until completion."""

    can_delegate = ctx.can_delegate()
    is_root = ctx.depth == 0
    tools = get_tools(can_delegate)
    system_prompt = get_system_prompt(
        can_delegate, is_root=is_root, introspection=ctx.introspection
    )

    if is_continuation:
        ctx.messages.append({"role": "user", "content": prompt})
        ctx.log(f"CONTINUE: {prompt[:50]}...")
    else:
        # Build user content with page images
        user_content = []
        for label in sorted(
            ctx.node, key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0
        ):
            if label in ctx.all_pages:
                user_content.append({"type": "text", "text": f"\n=== {label} ==="})
                img_b64 = base64.b64encode(ctx.all_pages[label]).decode()
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )

        user_content.append({"type": "text", "text": f"\n## Task\n\n{prompt}"})

        ctx.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        ctx.log(f"START pages={ctx.node}")

    for iteration in range(MAX_ITERATIONS):
        async with get_semaphore():
            response = await ctx.client.chat(
                model=ctx.model,
                messages=ctx.messages,
                tools=tools,
                reasoning_effort="high",
            )

        # Accumulate usage
        usage = response.get("usage", {})
        ctx.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        ctx.usage["completion_tokens"] += usage.get("completion_tokens", 0)

        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            # Agent done
            content = message.get("content", "")
            ctx.log(f"DONE: {content[:100]}...")
            ctx.messages.append(message)
            # If root agent, print full final message including introspection
            if ctx.depth == 0:
                print(
                    f"\n{'=' * 60}\nROOT AGENT FINAL OUTPUT:\n{'=' * 60}\n{content}\n{'=' * 60}\n",
                    file=sys.stderr,
                )
            return ctx.usage

        ctx.messages.append(message)
        tool_results = await process_tool_calls(ctx, tool_calls)
        ctx.messages.extend(tool_results)

    ctx.log("MAX ITERATIONS reached")
    return ctx.usage


async def process_tool_calls(ctx: AgentContext, tool_calls: list[dict]) -> list[dict]:
    """Process tool calls, running spawns/continues in parallel."""

    spawn_calls = []
    continue_calls = []
    other_calls = []

    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}

        if name == "spawn" and ctx.can_delegate():
            spawn_calls.append((tc, args))
        elif name == "continue" and ctx.can_delegate():
            continue_calls.append((tc, args))
        else:
            other_calls.append((tc, args))

    results = []

    # Handle read/patch calls
    for tc, args in other_calls:
        name = tc["function"]["name"]

        if name == "read":
            expr = args.get("expr", "state")
            try:
                content = await ctx.shared_state.eval(expr)
                ctx.log(f"read({expr[:30]}) -> {len(content)} chars")
            except Exception as e:
                content = f"Error: {e}"
                ctx.log(f"read() -> ERROR: {e}")
            results.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": content}
            )

        elif name == "patch":
            ops = args.get("operations", [])
            success, error = await ctx.shared_state.patch(ctx.call_path, ops)
            if success:
                content = f"OK ({len(ops)} operations applied)"
                ctx.log(f"patch() -> OK ({len(ops)} ops)")
            else:
                content = f"FAILED: {error}"
                ctx.log(f"patch() -> FAILED: {error[:50]}")
            results.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": content}
            )

        elif name == "calc":
            # calc is an alias for read - just evaluates the expression
            expr = args.get("expr", "0")
            try:
                content = await ctx.shared_state.eval(expr)
                ctx.log(f"calc({expr[:30]}) -> {content[:50]}")
            except Exception as e:
                content = f"Error: {e}"
                ctx.log(f"calc() -> ERROR: {e}")
            results.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": content}
            )

        elif name in ("spawn", "continue"):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: {name}() not available at this depth. Use read/patch directly.",
                }
            )

        else:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Unknown tool: {name}",
                }
            )

    # Run spawn calls in parallel
    if spawn_calls:
        ctx.log(f"Spawning {len(spawn_calls)} sub-agents...")

        async def run_spawn(tc, args):
            handle = args.get("handle", "sub")
            prompt = args.get("prompt", "")
            pages = args.get("pages", [])

            valid_pages = [p for p in pages if p in ctx.node]
            if not valid_pages:
                return (
                    tc["id"],
                    handle,
                    f"Error: No valid pages in {pages}. Available: {ctx.node}",
                    {},
                    None,
                )

            child_ctx = AgentContext(
                node=valid_pages,
                all_pages=ctx.all_pages,
                shared_state=ctx.shared_state,
                model=ctx.model,
                client=ctx.client,
                depth=ctx.depth + 1,
                call_path=f"{ctx.call_path}/{handle}",
                terminate_depth=ctx.terminate_depth,
                introspection=ctx.introspection,
            )

            try:
                usage = await run_agent(child_ctx, prompt)
                final_msg = ""
                for msg in reversed(child_ctx.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_msg = msg["content"]
                        break
                return tc["id"], handle, f"[{handle}] {final_msg}", usage, child_ctx
            except Exception as e:
                return tc["id"], handle, f"Error: {e}", {}, None

        spawn_results = await asyncio.gather(
            *[run_spawn(tc, args) for tc, args in spawn_calls]
        )

        for tc_id, handle, content, sub_usage, child_ctx in spawn_results:
            results.append({"role": "tool", "tool_call_id": tc_id, "content": content})
            ctx.usage["prompt_tokens"] += sub_usage.get("prompt_tokens", 0)
            ctx.usage["completion_tokens"] += sub_usage.get("completion_tokens", 0)
            if child_ctx:
                ctx.sub_agents[handle] = child_ctx

    # Run continue calls in parallel
    if continue_calls:
        ctx.log(f"Continuing {len(continue_calls)} sub-agents...")

        async def run_continue(tc, args):
            handle = args.get("handle", "")
            prompt = args.get("prompt", "")

            if handle not in ctx.sub_agents:
                return (
                    tc["id"],
                    f"Error: No sub-agent with handle '{handle}'. Available: {list(ctx.sub_agents.keys())}",
                    {},
                )

            child_ctx = ctx.sub_agents[handle]
            try:
                usage = await run_agent(child_ctx, prompt, is_continuation=True)
                final_msg = ""
                for msg in reversed(child_ctx.messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        final_msg = msg["content"]
                        break
                return tc["id"], f"[{handle}] {final_msg}", usage
            except Exception as e:
                return tc["id"], f"Error: {e}", {}

        continue_results = await asyncio.gather(
            *[run_continue(tc, args) for tc, args in continue_calls]
        )

        for tc_id, content, sub_usage in continue_results:
            results.append({"role": "tool", "tool_call_id": tc_id, "content": content})
            ctx.usage["prompt_tokens"] += sub_usage.get("prompt_tokens", 0)
            ctx.usage["completion_tokens"] += sub_usage.get("completion_tokens", 0)

    return results


# === Main Entry Point ===


async def run_swarm(
    pdf_path: Path,
    output_path: Path | None = None,
    model: str | None = None,
    max_depth: int = 2,
    prompt: str | None = None,
    introspection: bool = True,
) -> dict:
    """Run the swarm on a PDF financial statement.

    Args:
        pdf_path: Path to PDF file
        output_path: Output JSON path (default: pdf_path.with_suffix('.checks.json'))
        model: Model to use (default: FS_CHECK_MODEL env or gemini-3-flash-preview)
        max_depth: Maximum delegation depth
        prompt: Custom prompt (default: standard FS checking prompt)
        introspection: Include introspection in root agent output (default: True)

    Returns:
        The final shared state dict
    """
    model = model or DEFAULT_MODEL
    output_path = output_path or pdf_path.with_suffix(".checks.json")

    print(f"Loading {pdf_path}...", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)

    page_images = pdf_to_images(pdf_path.read_bytes(), dpi=150)
    all_pages = {f"Page {i + 1}": img for i, img in enumerate(page_images)}
    labels = list(all_pages.keys())
    print(f"Pages: {len(labels)}", file=sys.stderr)

    shared_state = SharedState()
    client = OpenRouterClient(reasoning_effort="high")

    ctx = AgentContext(
        node=labels,
        all_pages=all_pages,
        shared_state=shared_state,
        model=model,
        client=client,
        terminate_depth=max_depth,
        introspection=introspection,
    )

    default_prompt = """## IFRS Financial Statement Verification

Verify internal consistency and mathematical accuracy.

## Output

Add checks to `/checks` array:
```json
{"id": "...", "category": "cross_footing|internal_consistency|note_ties", 
 "status": "pass|fail|warn", "expected": N, "actual": N, "difference": N, 
 "reason": "...", "page": N}
```

Store extracted values in `/values` (e.g., `/values/balance_sheet/total_assets`).

## Check Types

- **cross_footing**: Totals add up (revenue - expenses = profit, subtotals correct)
- **internal_consistency**: Assets = Liabilities + Equity, CF ties to BS cash
- **note_ties**: Note totals match statement line items

Delegate by statement/section. Record ALL checks including passes."""

    start = time.time()
    usage = await run_agent(ctx, prompt or default_prompt)
    elapsed = time.time() - start

    # Write output
    output_path.write_text(json.dumps(shared_state.state, indent=2, ensure_ascii=False))

    # Summary
    checks = shared_state.state.get("checks", [])
    # Handle case where checks might be stored as dict or other structure
    if not isinstance(checks, list):
        checks = []
    pass_count = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") == "pass"
    )
    fail_count = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") == "fail"
    )
    warn_count = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") == "warn"
    )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Completed in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"Tokens: {usage['prompt_tokens']:,} in, {usage['completion_tokens']:,} out",
        file=sys.stderr,
    )
    print(
        f"Checks: {len(checks)} total ({pass_count} pass, {fail_count} fail, {warn_count} warn)",
        file=sys.stderr,
    )
    print(f"Output: {output_path}", file=sys.stderr)

    return shared_state.state

"""Swarm Strategy - Concurrent agents with shared state for financial statement checking.

Adapted from tech_pack_study/strategies/swarm.py for financial statement validation.

Key features:
- Multiple agents work CONCURRENTLY on shared JSON state
- Single eval() tool for reads AND writes (under lock)
- Hierarchical delegation with spawn(handle, prompt, pages)
- Sub-agent continuation with continue(handle, prompt)
- Scoped handles - no global namespace collision
- reasoning=high for reliable agent coordination

Check categories:
- cross_footing: Row/column totals verify
- internal_consistency: A=L+E, CF ties to BS, etc.
- note_ties: Note values match statement values

Usage:
    python -m scripts.cli document.pdf -o checks.json
"""

import asyncio
import base64
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# === Safe Exec ===


def safe_exec(code: str, variables: dict, persistent_locals: dict | None = None) -> Any:
    """Execute Python code with restricted builtins.

    Behavior:
    - Single-line: evaluated as expression, returns value
    - Multi-line: executed as statements, returns None (caller shows "OK")
    - Rebind detection: caller checks if state was rebound
    - state.clear() blocked: raises error
    - print(): no-op (use expressions to see values)
    - Local vars: persist if persistent_locals provided

    Args:
        code: Python code to execute
        variables: Variables to inject (e.g., {"state": {...}})
        persistent_locals: If provided, used as local namespace (persists across calls)
    """
    safe_builtins = {
        # Constants
        "None": None,
        "True": True,
        "False": False,
        # Types
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "bytes": bytes,
        # Iteration
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "next": next,
        "iter": iter,
        # Aggregation
        "sum": sum,
        "min": min,
        "max": max,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
        "pow": pow,
        # Introspection
        "isinstance": isinstance,
        "type": type,
        "hasattr": hasattr,
        "getattr": getattr,
        "callable": callable,
        # Formatting
        "repr": repr,
        "format": format,
        # Math module
        "math": math,
        # Collections
        "Counter": Counter,
        # Print as no-op
        "print": lambda *args, **kwargs: None,
        # Exceptions (for try/except)
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
    }

    # Helper functions
    def group_by(iterable, key_fn):
        """Group items by key function or dict key."""
        result = {}
        for item in iterable:
            k = key_fn(item) if callable(key_fn) else item.get(key_fn)
            result.setdefault(k, []).append(item)
        return result

    def pluck(iterable, key):
        """Extract a key from each item."""
        return [
            item.get(key) if isinstance(item, dict) else getattr(item, key, None)
            for item in iterable
        ]

    helpers = {
        "group_by": group_by,
        "pluck": pluck,
    }

    # Use persistent_locals as actual namespace, or create fresh one
    ns = persistent_locals if persistent_locals is not None else {}

    # Inject builtins and helpers
    ns["__builtins__"] = safe_builtins
    ns.update(helpers)

    # Inject variables (state, etc.) - always update to get fresh references
    for k, v in variables.items():
        ns[k] = v

    original_state = variables.get("state")

    code = code.strip()
    is_single_line = "\n" not in code

    if is_single_line:
        # Single line: try as expression first
        try:
            result = eval(code, ns, ns)
        except SyntaxError:
            # Statement like assignment
            exec(code, ns, ns)
            result = None
    else:
        # Multi-line: exec using same dict for globals and locals
        # This gives "normal" Python scoping - functions can see all variables
        exec(code, ns, ns)
        result = None

    # Check for state.clear() - simple string detection
    if "state.clear()" in code or "state.clear ()" in code:
        raise RuntimeError(
            "state.clear() is not allowed. Use del state['key'] or state.pop('key') instead."
        )

    return result


def format_preview(value: Any, max_len: int = 10000) -> str:
    """Format value with smart truncation.

    - None → "OK"
    - Small values → full JSON
    - Large values → truncated with count hint
    """
    if value is None:
        return "OK"

    try:
        s = json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(value)

    if len(s) <= max_len:
        return s

    # Smart truncation: show what was truncated
    truncated = s[:max_len]

    # Try to figure out what type and count
    if isinstance(value, list):
        hint = f"list with {len(value)} items"
    elif isinstance(value, dict):
        hint = f"dict with {len(value)} keys"
    else:
        hint = f"{len(s)} chars total"

    return f"{truncated}\n\n... TRUNCATED ({hint}). Data saved to state. Use eval() to inspect specific parts."


# === Shared State ===


class SharedState:
    """Thread-safe shared JSON state for all agents.

    Eval semantics:
    - Each eval() call runs under lock - atomic read-modify-write
    - Only mutations to `state` persist (it's the same dict object across all evals)
    - Local variables do NOT persist between eval calls
    - Rebinding `state` (e.g., `state = {...}`) is detected and returns an error

    Examples:
        eval("state")                          # read full state
        eval("state.get('checks', [])")        # read with default
        eval("state['metadata'] = {...}")      # write
        eval("state.setdefault('checks', []).append({...})")  # atomic append
        eval("state = {'a': 1}")               # ERROR: rebinding detected
    """

    def __init__(self):
        self.state: dict = {
            "metadata": {},
            "checks": [],
            "values": {},
        }
        self.lock = asyncio.Lock()
        self.eval_log: list[dict] = []

    async def eval(
        self, code: str, agent_path: str = "", eval_locals: dict | None = None
    ) -> str:
        """Execute Python on state (reads and writes) under lock.

        Args:
            code: Python code to execute
            agent_path: Agent identifier for logging
            eval_locals: Agent's persistent local namespace (variables persist across calls)
        """
        async with self.lock:
            # Use agent's persistent locals, or empty dict if none
            local_ns = eval_locals if eval_locals is not None else {}

            # Always inject fresh state reference (shared, not agent-local)
            local_ns["state"] = self.state

            try:
                result = safe_exec(
                    code,
                    {"state": self.state},
                    persistent_locals=local_ns,
                )
            except Exception as e:
                return f"Error: {type(e).__name__}: {e}"

            self.eval_log.append({"agent": agent_path, "code": code[:100]})

            # Detect rebinding: if agent did `state = {...}`, local_ns["state"]
            # is a NEW dict, not self.state. This loses their work silently.
            if local_ns.get("state") is not self.state:
                # Restore correct state reference
                local_ns["state"] = self.state
                return (
                    "Error: Rebinding 'state' loses your changes. Mutate instead:\n"
                    "  state['key'] = value\n"
                    "  state.update({...})\n"
                    "  state.setdefault('key', []).append(...)"
                )

        return format_preview(result)


# === Tool Definitions ===

TOOL_SPAWN = {
    "type": "function",
    "function": {
        "name": "spawn",
        "description": "Spawn a sub-agent for a subset of pages. Returns a handle for continue(). Call multiple spawns in ONE turn for parallel execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Unique handle (e.g., 'balance_sheet', 'note_10')",
                },
                "prompt": {
                    "type": "string",
                    "description": "Instructions for what to check/extract",
                },
                "pages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Page labels, e.g. ["Page 1", "Page 2"]',
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
        "description": "Continue a previously spawned sub-agent with follow-up prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "Handle of sub-agent"},
                "prompt": {"type": "string", "description": "Follow-up instruction"},
            },
            "required": ["handle", "prompt"],
        },
    },
}

TOOL_EVAL = {
    "type": "function",
    "function": {
        "name": "eval",
        "description": "Execute Python. `state` is shared with all agents. Other variables are local to you and persist across calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code. Examples: x = state.get('data'), state['result'] = x * 2, state.setdefault('checks', []).append({...})",
                },
            },
            "required": ["code"],
        },
    },
}


# === System Prompts ===

SYSTEM_PROMPT_CORE = """\
You are an agent in a SWARM. Multiple agents work concurrently on shared JSON state.

## Tool

**eval(code)** - Execute Python on shared `state` dict. Use for reads AND writes.

| Code type | Result |
|-----------|--------|
| Single-line expression | Returns value |
| Multi-line statements | Returns "OK" |

Read examples:
- `eval("state")` - full state
- `eval("list(state.keys())")` - structure discovery  
- `eval("state.get('checks', [])")` - subtree with default
- `eval("len(state.get('checks', []))")` - count
- `eval("[c for c in state['checks'] if c['status']=='fail']")` - filter
- `eval("1234 + 5678")` - arithmetic

Write examples:
- `eval("state['metadata'] = {'company': 'ABC'}")` → "OK"
- `eval("state.setdefault('checks', []).append({...})")` → "OK"

## Variables

- **`state`** - SHARED with all agents. Mutate it, don't rebind.
- **Other variables** - LOCAL to you, persist across your eval() calls. Other agents cannot see them.

Example:
```
eval("my_data = extract_values()")   # my_data saved locally
eval("my_data['total']")             # works - my_data persists
eval("state['result'] = my_data")    # share via state
```

## Rules

- eval() to read state FIRST before writing
- Use setdefault() + append() for safe concurrent writes
- If something fails, re-read state and adapt

Final message (no tool calls) = summary for parent.
"""

SYSTEM_PROMPT_DELEGATION = """
## Delegation

**spawn(handle, prompt, pages)** - Create sub-agent for pages
**continue(handle, prompt)** - Follow-up to existing sub-agent

**4+ pages → MUST delegate.** Spawns run in parallel = faster.

Spawn multiple in ONE turn:
```
spawn("bs", "Check balance sheet...", ["Page 1", "Page 2"])
spawn("pl", "Check P&L...", ["Page 3"])
```

**IMPORTANT:** Include schema/format requirements in spawn prompts. Sub-agents don't see your instructions.

After spawns, eval() to verify. Use continue() for corrections.
"""

SYSTEM_PROMPT_ROOT = """
## Root Agent

You orchestrate sub-agents. Create workspace structures as needed:
```python
eval("state['workspace'] = {'ledger': [], 'note_map': {}}")
```

**Before finishing:**
1. eval() to verify sub-agent output matches required schema
2. Fix/normalize any inconsistent structures
3. Clean up: `eval("del state['workspace']")`
"""

SYSTEM_PROMPT_INTROSPECTION = """
## Completion

Final message must include:

1. **Task Outcome**: Results summary, pass/fail counts, key findings.

2. **Introspection**: Reflect on what you learned:
   - What worked well in delegation/coordination?
   - What was inefficient or could be improved?
   - What changes to system/prompts/tools would help?

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
    tools = [TOOL_EVAL]
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
    eval_locals: dict = field(default_factory=dict)  # Persistent local variables

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


def _now() -> str:
    """Get current wall clock time as HH:MM:SS."""
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def _prefix_time(content: str) -> str:
    """Prefix content with wall clock time."""
    return f"[{_now()}] {content}"


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

    # Handle eval calls
    for tc, args in other_calls:
        name = tc["function"]["name"]

        if name == "eval":
            code = args.get("code", "state")
            content = await ctx.shared_state.eval(code, ctx.call_path, ctx.eval_locals)
            # Truncate log for long code
            code_preview = code[:40] + "..." if len(code) > 40 else code
            content_preview = content[:50] + "..." if len(content) > 50 else content
            ctx.log(f"eval({code_preview}) -> {content_preview}")
            results.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": content}
            )

        elif name in ("spawn", "continue"):
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": f"Error: {name}() not available at this depth. Use eval() directly.",
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

    # Prefix all results with wall clock time
    for r in results:
        r["content"] = _prefix_time(r["content"])

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

Verify internal consistency and mathematical accuracy. Be THOROUGH - check every subtotal, every note tie, every rollforward.

## Check Schema (STRICT)

Every check MUST have exactly these fields:
```python
state.setdefault('checks', []).append({
    "id": "bs_total_assets_2023",   # unique snake_case with year
    "category": "cross_footing",     # cross_footing | internal_consistency | note_ties
    "status": "pass",                # pass | fail | warn
    "expected": 1234,                # calculated value
    "actual": 1234,                  # stated value in document
    "difference": 0,                 # expected - actual
    "description": "Total assets = sum of all asset lines",
    "page": 3
})
```

**When delegating, COPY this schema into spawn prompts.** Sub-agents don't see your instructions.

## Required Checks by Statement

### Balance Sheet
- All subtotals cross-foot (non-current assets, current assets, etc.)
- Total Assets = Total Liabilities + Equity (both years)
- Net current assets/liabilities calculation
- Prior year comparatives match

### P&L / Income Statement  
- Gross profit = Revenue - Cost of sales
- Operating profit = Gross profit - Operating expenses
- All subtotals and totals cross-foot
- Note 4 (Operating profit) ties to P&L line items

### OCI (Other Comprehensive Income)
- Subtotals for items that will/won't be reclassified
- Total comprehensive income = Net profit + OCI

### Cash Flow Statement
- Operating/Investing/Financing subtotals cross-foot
- Net change in cash = Op + Inv + Fin
- Closing cash = Opening + Net change
- Closing cash ties to Balance Sheet cash

### Notes Verification
- **Note rollforwards**: Opening + Additions - Disposals = Closing (PPE, provisions, etc.)
- **Note ties**: Note totals match corresponding line items on face of statements
- **Internal math**: All subtotals within notes cross-foot
- **Look for $100k or round number discrepancies** - often indicate missing entries

### Cross-Statement Consistency
- Net profit in P&L = Net profit in OCI = Net profit in Cash Flow reconciliation
- Closing cash in CF = Cash in BS
- Retained earnings movement ties to net profit

## Delegation Strategy

Spawn by major section:
- P&L + OCI + Note 4 (operating expenses)
- Balance Sheet + PPE notes + Receivables notes
- Cash Flow + Note 31 (cash flow reconciliation)

Each sub-agent should check EVERY number, not just totals. Record ALL checks including passes."""

    start = time.time()
    usage = await run_agent(ctx, prompt or default_prompt)
    elapsed = time.time() - start

    # Write output
    output_path.write_text(json.dumps(shared_state.state, indent=2, ensure_ascii=False))

    # Summary
    checks = shared_state.state.get("checks", [])
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
    print(f"Evals: {len(shared_state.eval_log)}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)

    return shared_state.state

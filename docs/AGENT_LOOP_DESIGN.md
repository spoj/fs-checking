# Agent Loop Design

Minimal reference for building LLM agent loops. Copy `api.py` and `agent_core.py` to new projects.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   cli.py     │────▶│  your_app.py │────▶│ agent_core.py│
│   (entry)    │     │  (domain)    │     │ (generic)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                          ┌───────▼───────┐
                                          │    api.py     │
                                          │ (OpenRouter)  │
                                          └───────────────┘
```

## Core Files

### `api.py` - HTTP Client

```python
class OpenRouterClient:
    def __init__(
        self,
        reasoning_effort: str = "high",
        timeout: float = 600.0,
        max_retries: int = 8,
    ): ...

    async def chat(self, model, messages, tools=None) -> {"message": ..., "usage": ...}
```

Features:
- Exponential backoff with jitter (3s → 30s)
- Don't retry 401/403/404
- Anthropic prompt caching support

### `agent_core.py` - Loop Logic

```python
async def run_agent_loop(
    call_model: ModelCallable,      # (messages) -> (msg, usage)
    tool_executor: ToolExecutor,    # (name, args) -> str  (sync or async)
    initial_messages: list[dict],
    validation_fn: ValidationFn | None = None,  # () -> (valid, error)
    max_iterations: int = 200,
    parallel_tools: bool = True,
) -> AgentResult
```

Features:
- Auto-detects sync/async tool executor
- Parallel tool execution for async executors
- Validation loop: feeds errors back as user messages
- Usage accumulation across iterations

## Usage Pattern

```python
from your_app.api import OpenRouterClient, create_model_callable
from your_app.agent_core import run_agent_loop

# 1. Create client and model callable
client = OpenRouterClient()
call_model = create_model_callable(client, "google/gemini-3-flash-preview", tools)

# 2. Define tool executor
async def execute_tool(name: str, args: dict) -> str:
    if name == "search":
        return json.dumps(await search(args["query"]))
    return json.dumps({"error": f"Unknown: {name}"})

# 3. Optional validation
def validate() -> tuple[bool, str]:
    if output_is_valid():
        return True, ""
    return False, "Missing required field X"

# 4. Run
result = await run_agent_loop(
    call_model=call_model,
    tool_executor=execute_tool,
    initial_messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ],
    validation_fn=validate,
)

if result.success:
    print(result.final_message)
    print(f"Tokens: {result.usage}")
```

## Tool Definition

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]
```

## AgentResult

```python
@dataclass
class AgentResult:
    success: bool           # Did it complete successfully?
    final_message: str      # Last assistant message
    iterations: int         # How many loop iterations
    messages: list[dict]    # Full conversation history
    usage: dict             # {prompt_tokens, completion_tokens, cache_read_tokens, reasoning_tokens}
    tool_calls_count: int
```

## Checklist for New Projects

1. Copy `api.py` and `agent_core.py` as-is
2. Create `.env` with `OPENROUTER_API_KEY`
3. Define tools for your domain
4. Write system prompt with clear output format
5. Implement tool executor
6. Add validation if output is verifiable
7. Create CLI entry point

## Common Pitfalls

- **Not using `reasoning_effort="high"`** — agents coordinate poorly without it
- **Sync executor with slow I/O** — use async for network/disk operations
- **No validation loop** — agents often need 2-3 attempts for correct format
- **Giant system prompts** — keep <2000 tokens, use tools for reference data

"""OpenRouter API client with retry logic and cache control.

Adapted from recon/src/api.py - this is the canonical API layer.

Features:
- Async httpx client with exponential backoff
- Cache control for Anthropic prompt caching
- Configurable reasoning_effort
- Usage tracking

Example:
    client = OpenRouterClient()
    response = await client.chat(
        model="google/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[...],
    )
    message = response["message"]
    usage = response["usage"]
"""

import os
import json
import asyncio
import random
import sys
from typing import Any

import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_headers() -> dict:
    """Get API request headers."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fs-checking",
    }


def add_cache_control(messages: list[dict]) -> list[dict]:
    """Add cache_control breakpoint to the last cacheable message.

    Finds the last user or system message and adds cache_control to it.
    This tells Anthropic models to cache everything up to this point.
    """
    import copy

    messages = copy.deepcopy(
        messages
    )  # Don't mutate original (deep for nested content)

    # Find last user or system message (going backwards)
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role")
        if role in ("user", "system"):
            content = messages[i].get("content", "")

            # Convert string content to content block format if needed
            if isinstance(content, str):
                messages[i]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list):
                # Add cache_control to last text block
                for j in range(len(content) - 1, -1, -1):
                    if content[j].get("type") == "text":
                        content[j]["cache_control"] = {"type": "ephemeral"}
                        break
            break

    return messages


class OpenRouterClient:
    """Async client for OpenRouter API with reasoning and caching."""

    def __init__(
        self,
        reasoning_effort: str = "high",
        enable_cache: bool = True,
        timeout: float = 600.0,
        max_retries: int = 8,
    ):
        self.reasoning_effort = reasoning_effort
        self.enable_cache = enable_cache
        self.timeout = timeout
        self.max_retries = max_retries

        # Status codes that should NOT be retried
        self.no_retry_codes = {401, 403, 404}

    async def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """
        Make a chat completion request.

        Args:
            model: Model identifier (e.g., "google/gemini-3-flash-preview")
            messages: List of message dicts
            tools: Optional list of tool definitions
            tool_choice: "auto", "none", or "required"
            reasoning_effort: Override instance default

        Returns:
            {
                "message": assistant message dict with 'content' and 'tool_calls',
                "usage": usage dict with token counts
            }
        """
        # Add cache control to messages for Anthropic models
        if self.enable_cache and "anthropic" in model.lower():
            messages = add_cache_control(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        # Add reasoning effort
        effort = reasoning_effort or self.reasoning_effort
        if effort:
            payload["reasoning"] = {"effort": effort}

        backoff = 3.0
        max_backoff = 30.0
        attempt = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                try:
                    response = await client.post(
                        OPENROUTER_API_URL,
                        headers=get_headers(),
                        json=payload,
                    )
                except (httpx.TimeoutException, httpx.RequestError) as e:
                    if attempt < self.max_retries:
                        jitter = random.uniform(0, 3)
                        print(
                            f"[Retry {attempt + 1}/{self.max_retries}] Network error: {type(e).__name__}: {e}",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(backoff + jitter)
                        backoff = min(backoff * 2, max_backoff)
                        attempt += 1
                        continue
                    raise RuntimeError(
                        f"API error after {self.max_retries} retries: {e}"
                    )

                response_text = response.text
                # Retry empty/whitespace-only responses (transient proxy issue)
                if not response_text or not response_text.strip():
                    if attempt < self.max_retries:
                        attempt += 1
                        jitter = random.uniform(0, 3)
                        print(
                            f"[Retry {attempt}/{self.max_retries}] Empty response (status {response.status_code})",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(backoff + jitter)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    raise RuntimeError(
                        f"Empty response after {self.max_retries} retries (status {response.status_code})"
                    )

                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    # Non-empty but malformed — don't retry, let caller handle
                    raise RuntimeError(
                        f"Invalid JSON response (status {response.status_code}): {response_text[:200]}"
                    )

                # Success
                if response.status_code == 200:
                    choice = response_data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    usage = response_data.get("usage", {})

                    return {
                        "message": message,
                        "usage": usage,
                    }

                # Don't retry certain client errors
                if response.status_code in self.no_retry_codes:
                    error_detail = response_data.get("error", {}).get(
                        "message", response_text[:500]
                    )
                    raise RuntimeError(
                        f"OpenRouter API error: {response.status_code} - {response.reason_phrase}: {error_detail}"
                    )

                # Retry all other errors
                if attempt < self.max_retries:
                    error_detail = response_data.get("error", {}).get(
                        "message", response_text[:200]
                    )
                    jitter = random.uniform(0, 5)
                    print(
                        f"[Retry {attempt + 1}/{self.max_retries}] HTTP {response.status_code}: {error_detail}",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(backoff + jitter)
                    backoff = min(backoff * 2, max_backoff)
                    attempt += 1
                    continue

                # Exhausted retries
                error_detail = response_data.get("error", {}).get(
                    "message", response_text[:500]
                )
                raise RuntimeError(
                    f"OpenRouter API error after {self.max_retries} retries: {response.status_code}: {error_detail}"
                )


def create_model_callable(
    client: OpenRouterClient,
    model: str,
    tools: list[dict] | None = None,
):
    """Create a model callable for use with run_agent_loop.

    Returns an async function (messages) -> (assistant_message, usage)
    """

    async def call_model(messages: list[dict]) -> tuple[dict, dict]:
        response = await client.chat(
            model=model,
            messages=messages,
            tools=tools,
        )
        return response["message"], response["usage"]

    return call_model

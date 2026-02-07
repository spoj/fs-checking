"""OpenRouter API client with retry logic.

Async httpx client with exponential backoff, usage tracking,
and configurable reasoning effort. Designed for Gemini Flash/Pro
via OpenRouter.

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

import asyncio
import json
import os
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


def _get_headers() -> dict:
    """Get API request headers."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fs-checking",
    }


class OpenRouterClient:
    """Async client for OpenRouter API with retry and reasoning effort."""

    def __init__(
        self,
        reasoning_effort: str = "high",
        timeout: float = 600.0,
        max_retries: int = 8,
    ):
        self.reasoning_effort = reasoning_effort
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
        """Make a chat completion request with retry.

        Returns:
            {"message": assistant message dict, "usage": usage dict}
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

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
                        headers=_get_headers(),
                        json=payload,
                    )
                except (httpx.TimeoutException, httpx.RequestError) as e:
                    if attempt < self.max_retries:
                        jitter = random.uniform(0, 3)
                        print(
                            f"[Retry {attempt + 1}/{self.max_retries}] "
                            f"Network error: {type(e).__name__}: {e}",
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

                # Empty response — transient proxy/OpenRouter issue, retry
                if not response_text or not response_text.strip():
                    if attempt < self.max_retries:
                        attempt += 1
                        jitter = random.uniform(0, 3)
                        print(
                            f"[Retry {attempt}/{self.max_retries}] "
                            f"Empty response (status {response.status_code})",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(backoff + jitter)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    raise RuntimeError(
                        f"Empty response after {self.max_retries} retries "
                        f"(status {response.status_code})"
                    )

                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    if attempt < self.max_retries:
                        attempt += 1
                        jitter = random.uniform(0, 3)
                        print(
                            f"[Retry {attempt}/{self.max_retries}] "
                            f"Invalid JSON (status {response.status_code}): "
                            f"{response_text[:120]}",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(backoff + jitter)
                        backoff = min(backoff * 2, max_backoff)
                        continue
                    raise RuntimeError(
                        f"Invalid JSON response (status {response.status_code}): "
                        f"{response_text[:200]}"
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
                        f"OpenRouter API error: {response.status_code} - "
                        f"{response.reason_phrase}: {error_detail}"
                    )

                # Retry all other errors (429, 500, 502, 503, etc.)
                if attempt < self.max_retries:
                    error_detail = response_data.get("error", {}).get(
                        "message", response_text[:200]
                    )
                    jitter = random.uniform(0, 5)
                    print(
                        f"[Retry {attempt + 1}/{self.max_retries}] "
                        f"HTTP {response.status_code}: {error_detail}",
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
                    f"OpenRouter API error after {self.max_retries} retries: "
                    f"{response.status_code}: {error_detail}"
                )

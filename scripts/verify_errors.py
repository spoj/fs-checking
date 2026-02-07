"""Verify that injected errors are detectable by sending the PDF + error list to models."""

import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fs_checking.api import OpenRouterClient


VERIFY_PROMPT = """You are an IFRS financial statement auditor. I have a PDF of financial statements that contains 10 known injected numerical errors. For each error below, look at the specific page and location described, and tell me:

1. Can you see the MUTATED (wrong) value in the PDF?
2. Can you verify it doesn't match the ORIGINAL (correct) value that should be there?
3. Is the error detectable by cross-referencing with other pages/notes?

Here are the 10 injected errors:

{errors_text}

For each error, respond with a JSON array of 10 objects:
```json
[
  {{
    "id": "inject_000",
    "mutated_value_visible": true/false,
    "original_value_found_elsewhere": true/false,
    "detectable": true/false,
    "notes": "brief explanation of what you see"
  }},
  ...
]
```

Be precise. Check the actual numbers on the pages specified."""


async def verify(pdf_path: str, ground_truth_path: str, model: str):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    with open(ground_truth_path) as f:
        gt = json.load(f)

    # Format errors for the prompt
    errors_text = ""
    for detail in gt["_injection_details"]:
        errors_text += (
            f"- {detail['id']} (page {detail['page']}): "
            f"{detail['original_text']} → {detail['mutated_text']} "
            f"({detail['mutation_kind']}): {detail['description']}\n"
        )

    prompt = VERIFY_PROMPT.format(errors_text=errors_text)

    pdf_b64 = base64.b64encode(pdf_bytes).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "filename": Path(pdf_path).name,
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    client = OpenRouterClient(timeout=600.0)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Verifying with {model}...", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    response = await client.chat(model=model, messages=messages)
    content = response["message"].get("content", "")
    usage = response.get("usage", {})

    print(
        f"\nTokens: {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out",
        file=sys.stderr,
    )

    # Try to parse JSON from response
    import re

    json_match = re.search(r"\[[\s\S]*\]", content)
    if json_match:
        results = json.loads(json_match.group())
        # Pretty print
        detected = sum(1 for r in results if r.get("detectable"))
        visible = sum(1 for r in results if r.get("mutated_value_visible"))
        print(f"\n{model}:")
        print(f"  Mutated values visible: {visible}/10")
        print(f"  Detectable: {detected}/10")
        print()
        for r in results:
            status = "✓" if r.get("detectable") else "✗"
            vis = "visible" if r.get("mutated_value_visible") else "NOT visible"
            ref = "ref found" if r.get("original_value_found_elsewhere") else "no ref"
            print(f"  {status} {r['id']}: {vis}, {ref}")
            print(f"    {r.get('notes', '')}")
        return results
    else:
        print(f"Could not parse JSON from response. Raw:\n{content[:2000]}")
        return None


async def main():
    pdf = "samples/ar2019.injected_s42.pdf"
    gt = "samples/ar2019.injected_s42.ground_truth.json"

    models = [
        "openai/gpt-5.2",
        "google/gemini-3-pro-preview",
    ]

    # Run both in parallel
    tasks = [verify(pdf, gt, model) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for model, result in zip(models, results):
        if isinstance(result, Exception):
            print(f"\n{model} FAILED: {result}")


if __name__ == "__main__":
    asyncio.run(main())

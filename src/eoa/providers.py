"""Chat adapters for LLMPlanner. Each returns a callable chat(messages, tools) -> {"text", "tool_calls"}.

The Gemini adapter uses the public REST API (generateContent with function calling). The key is read
from the environment variable GEMINI_API_KEY or from the file ~/.gemini_key and is never logged.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        f = Path.home() / ".gemini_key"
        if f.exists():
            k = f.read_text().strip()
    if not k:
        raise RuntimeError("no key: set GEMINI_API_KEY or put it in ~/.gemini_key")
    return k


def to_gemini(messages: list, tools: list) -> dict:
    """Translate the neutral message list into a generateContent request body."""
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    contents = []
    id_to_name: dict = {}
    for m in messages:
        if m["role"] == "user":
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            parts = [{"text": m["content"]}] if m.get("content") else []
            for c in m.get("tool_calls", []):
                id_to_name[c["id"]] = c["name"]
                parts.append({"functionCall": {"name": c["name"], "args": c["args"]}})
            contents.append({"role": "model", "parts": parts})
        elif m["role"] == "tool":
            import json
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": id_to_name.get(m["tool_call_id"], "tool"), "response": {"result": json.loads(m["content"])}}}]})
    body = {"contents": contents,
            "tools": [{"functionDeclarations": [{"name": t["name"], "description": t["description"],
                                                 "parameters": t["parameters"]} for t in tools]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def from_gemini(resp: dict) -> dict:
    parts = (resp.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    calls = [{"id": f"call{i}", "name": p["functionCall"]["name"], "args": dict(p["functionCall"].get("args", {}))}
             for i, p in enumerate(x for x in parts if "functionCall" in x)]
    return {"text": text, "tool_calls": calls}


def gemini_chat(model: str = "gemini-2.5-flash", post=requests.post):
    def chat(messages, tools):
        r = post(GEMINI_URL.format(model=model), json=to_gemini(messages, tools),
                 headers={"x-goog-api-key": _key()}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"model endpoint returned {r.status_code}")
        return from_gemini(r.json())
    return chat

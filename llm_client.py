#!/usr/bin/env python3
"""Shared LLM client for autonovel.

Supports Anthropic Messages API (default) and OpenAI-compatible chat
completions (LM Studio, etc.) via env:

  AUTONOVEL_API_COMPAT=anthropic|openai   (auto: openai if base looks local)
  AUTONOVEL_API_BASE_URL=...
  ANTHROPIC_API_KEY / OPENAI_API_KEY / AUTONOVEL_API_KEY
  AUTONOVEL_WRITER_MODEL / AUTONOVEL_JUDGE_MODEL
  AUTONOVEL_TIMEOUT=seconds (default 300)
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_BASE = Path(__file__).parent
load_dotenv(_BASE / ".env")


def _api_key() -> str:
    return (
        os.environ.get("AUTONOVEL_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or "lm-studio"
    )


def _base_url() -> str:
    return os.environ.get(
        "AUTONOVEL_API_BASE_URL", "https://api.anthropic.com"
    ).rstrip("/")


def _compat() -> str:
    explicit = os.environ.get("AUTONOVEL_API_COMPAT", "").strip().lower()
    if explicit in ("openai", "anthropic"):
        return explicit
    base = _base_url().lower()
    if any(
        x in base
        for x in ("127.0.0.1", "localhost", ":1234", "openai", "lmstudio", "lm-studio")
    ) and "anthropic.com" not in base:
        return "openai"
    return "anthropic"


def _timeout() -> float:
    return float(os.environ.get("AUTONOVEL_TIMEOUT", "300"))


def call_llm(
    prompt: str,
    *,
    model: str,
    system: str = "",
    max_tokens: int = 4000,
    temperature: float = 0.8,
) -> str:
    """Call configured backend; return assistant text."""
    import httpx

    compat = _compat()
    base = _base_url()
    key = _api_key()
    timeout = _timeout()

    if compat == "openai":
        # LM Studio / OpenAI-compatible
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        url = f"{base}/chat/completions"
        if not url.endswith("/chat/completions"):
            # base may already include /v1
            if base.endswith("/v1"):
                url = f"{base}/chat/completions"
            else:
                url = f"{base}/v1/chat/completions"
        resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        if content is None:
            # some servers put text at choice level
            content = choice.get("text") or ""
        if isinstance(content, list):
            # rare multi-part
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        content = (content or "").strip()
        if content:
            return content
        # Thinking models (e.g. some Qwen/Polaris builds) may spend the whole
        # budget in reasoning_content and leave content empty.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if isinstance(reasoning, list):
            reasoning = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in reasoning
            )
        reasoning = (reasoning or "").strip()
        if reasoning:
            # Prefer text after common end-of-think markers if present
            for marker in ("</think>", "</thinking>", "Final answer:", "FINAL:"):
                if marker in reasoning:
                    after = reasoning.split(marker, 1)[-1].strip()
                    if after:
                        return after
            return reasoning
        return ""

    # Anthropic Messages API
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "context-1m-2025-08-07",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    resp = httpx.post(
        f"{base}/v1/messages", headers=headers, json=payload, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

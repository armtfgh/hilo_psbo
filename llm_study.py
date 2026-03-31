
#%%
"""
Minimal multi-LLM study helper.

- Central model registry
- Single entry point to request JSON outputs
- OpenAI Chat Completions + OpenAI-compatible LLaMA servers
- Anthropic Messages API (requires ANTHROPIC_VERSION)

Edit MODEL_REGISTRY to match your exact model IDs.
"""
from __future__ import annotations

import json
import ssl
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_TIMEOUT = 90


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key_header: Optional[str] = None
    extra_headers: Optional[Dict[str, str]] = None


MODEL_REGISTRY: Dict[str, LLMConfig] = {
    # GPT (OpenAI)
    "gpt-4o-mini": LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4o": LLMConfig(
        provider="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4o-2024-05-13": LLMConfig(
        provider="openai",
        model="gpt-4o-2024-05-13",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4.1": LLMConfig(
        provider="openai",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4.1-mini": LLMConfig(
        provider="openai",
        model="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4.1-nano": LLMConfig(
        provider="openai",
        model="gpt-4.1-nano",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    "gpt-4": LLMConfig(
        provider="openai",
        model="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        api_key_header="Authorization",
    ),
    # Claude (Anthropic) - update model IDs to your exact variants
    "claude-3-5-sonnet": LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_header="x-api-key",
        extra_headers={"anthropic-version": os.getenv("ANTHROPIC_VERSION", "")},
    ),
    "claude-3-5-haiku": LLMConfig(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_header="x-api-key",
        extra_headers={"anthropic-version": os.getenv("ANTHROPIC_VERSION", "")},
    ),
    # Claude 4.5 (versioned models you provided)
    "claude-sonnet-4-5-20250929": LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_header="x-api-key",
        extra_headers={"anthropic-version": os.getenv("ANTHROPIC_VERSION", "")},
    ),
    "claude-haiku-4-5-20251001": LLMConfig(
        provider="anthropic",
        model="claude-haiku-4-5-20251001",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_header="x-api-key",
        extra_headers={"anthropic-version": os.getenv("ANTHROPIC_VERSION", "")},
    ),
    "claude-opus-4-5-20251101": LLMConfig(
        provider="anthropic",
        model="claude-opus-4-5-20251101",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        api_key_header="x-api-key",
        extra_headers={"anthropic-version": os.getenv("ANTHROPIC_VERSION", "")},
    ),
    # LLaMA (OpenAI-compatible server)
    "llama-3.1-70b": LLMConfig(
        provider="openai_compat",
        model="llama-3.1-70b",
        base_url=os.getenv("LLAMA_BASE_URL", "http://localhost:8000/v1"),
        api_key_env="LLAMA_API_KEY",
        api_key_header="Authorization",
    ),
    "llama-model-data": LLMConfig(
        provider="openai_compat",
        model="/model_data",
        base_url=os.getenv("LLAMA_BASE_URL", "http://localhost:8000/v1"),
        api_key_env="LLAMA_API_KEY",
        api_key_header="Authorization",
    ),
}


class LLMError(RuntimeError):
    pass


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    s = _strip_code_fences(text)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            return json.loads(s[start : end + 1])
        raise


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        import httpx  # type: ignore

        with httpx.Client(verify=False, timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        pass
    context = ssl._create_unverified_context()  # pragma: no cover
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = str(exc)
        raise LLMError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Connection error: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Non-JSON response: {body[:500]}") from exc


def _build_headers(cfg: LLMConfig, api_key_override: Optional[str] = None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key_env and cfg.api_key_header:
        key = api_key_override if api_key_override is not None else os.getenv(cfg.api_key_env, "").strip()
        if key:
            if cfg.api_key_header.lower() == "authorization":
                headers["Authorization"] = f"Bearer {key}"
            else:
                headers[cfg.api_key_header] = key
    if cfg.extra_headers:
        for k, v in cfg.extra_headers.items():
            if k.lower() == "anthropic-version":
                env_v = os.getenv("ANTHROPIC_VERSION", "").strip()
                if env_v:
                    v = env_v
            if v:
                headers[k] = v
    return headers


def _ensure_anthropic_version(headers: Dict[str, str]) -> None:
    if "anthropic-version" not in headers:
        raise LLMError("Missing anthropic-version header. Set ANTHROPIC_VERSION.")


def _openai_chat(
    cfg: LLMConfig,
    prompt: str,
    *,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    response_format: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    url = f"{cfg.base_url}/chat/completions"
    headers = _build_headers(cfg, api_key_override=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if response_format is not None:
        payload["response_format"] = response_format
    resp = _http_post_json(url, headers, payload, timeout)
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LLMError(f"Unexpected OpenAI response shape: {resp}") from exc


def _anthropic_messages(
    cfg: LLMConfig,
    prompt: str,
    *,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: Optional[str] = None,
) -> str:
    url = f"{cfg.base_url}/messages"
    headers = _build_headers(cfg, api_key_override=api_key)
    _ensure_anthropic_version(headers)
    payload = {
        "model": cfg.model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    resp = _http_post_json(url, headers, payload, timeout)
    blocks = resp.get("content", [])
    if not isinstance(blocks, list):
        raise LLMError(f"Unexpected Anthropic response shape: {resp}")
    text_parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            text_parts.append(b.get("text", ""))
    return "".join(text_parts).strip()


def _openai_compat(
    cfg: LLMConfig,
    prompt: str,
    *,
    system: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    response_format: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> str:
    # For LLaMA servers that expose OpenAI-compatible chat/completions.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _openai_chat(
            cfg,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            api_key=api_key,
        )

    key = api_key if api_key is not None else os.getenv(cfg.api_key_env or "", "").strip()
    if not key:
        key = None
    base_url = os.getenv("LLAMA_BASE_URL", "").strip() or cfg.base_url
    if not base_url:
        raise LLMError("Missing base_url for openai_compat provider.")
    client = OpenAI(base_url=base_url, api_key=key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    content = resp.choices[0].message.content
    if not content:
        raise LLMError("OpenAI-compatible server returned empty response.")
    return content


def get_llm_json(
    model_name: str,
    knowledge: str,
    *,
    system_prompt: Optional[str] = "Return a valid JSON object only. No prose.",
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: int = DEFAULT_TIMEOUT,
    strict_json: bool = True,
    api_key: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if model_name not in MODEL_REGISTRY:
        raise LLMError(f"Unknown model: {model_name}. Available: {sorted(MODEL_REGISTRY)}")

    cfg = MODEL_REGISTRY[model_name]
    provider = cfg.provider.lower()

    if provider == "openai":
        text = _openai_chat(
            cfg,
            knowledge,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            api_key=api_key,
        )
    elif provider == "anthropic":
        text = _anthropic_messages(
            cfg,
            knowledge,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )
    elif provider == "openai_compat":
        text = _openai_compat(
            cfg,
            knowledge,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=None,
            api_key=api_key,
        )
    else:
        raise LLMError(f"Unsupported provider: {provider}")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return _extract_json_from_text(text)
        except json.JSONDecodeError:
            pass
        if strict_json:
            raise LLMError(f"Model did not return valid JSON. Raw text:\n{text}")
        return {"raw_text": text}


def list_models() -> Dict[str, Dict[str, Optional[str]]]:
    return {
        name: {"provider": cfg.provider, "model": cfg.model, "base_url": cfg.base_url}
        for name, cfg in MODEL_REGISTRY.items()
    }


if __name__ == "__main__":
    # Optional smoke tests (disabled by default to avoid network calls on import).
    if os.getenv("LLM_STUDY_RUN_TESTS", "").strip().lower() in {"1", "true", "yes"}:
        readout = get_llm_json(
            "claude-sonnet-4-5-20250929",
            "Return STRICT JSON only (no prose).",
            system_prompt="what is your name?",
        )
        print(readout)

        from openai import OpenAI

        client = OpenAI(base_url="http://10.13.24.169:8000/v1", api_key=None)
        chat_completion = client.chat.completions.create(
            model="/model_data",
            messages=[{"role": "system", "content": "tell me a joke"}],
            temperature=0.8,
            max_tokens=500,
        )
        print(chat_completion.choices[0].message.content)

        os.environ["LLAMA_BASE_URL"] = "http://10.13.24.169:8000/v1"
        out = get_llm_json(
            "llama-model-data",
            "Tell me a joke in JSON: {\"joke\": \"...\"}",
            system_prompt="Return only valid JSON. No code fences.",
        )
        print(out)

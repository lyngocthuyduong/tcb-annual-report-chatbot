"""LLM provider factory. PROVIDER picks exactly one adapter. A failing provider is never replaced by another."""
from __future__ import annotations

import os

from .base import (ConfigError, GenerationError, LLMError, LLMRequest, LLMResponse, Provider, ProviderRejected,
                   ProviderTimeout, ProviderUnavailable)

__all__ = ["get", "ConfigError", "GenerationError", "LLMError", "LLMRequest", "LLMResponse", "Provider",
           "ProviderRejected", "ProviderTimeout", "ProviderUnavailable"]

DEFAULTS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5", "ollama": "qwen2.5:7b"}


def get(env=None) -> Provider:
    e = os.environ if env is None else env
    name = (e.get("PROVIDER") or "openai").strip().lower()
    if name not in DEFAULTS:
        raise ConfigError(name, "-", f"PROVIDER must be one of {sorted(DEFAULTS)}")
    var = {"openai": "OPENAI_MODEL", "anthropic": "ANTHROPIC_MODEL", "ollama": "OLLAMA_MODEL"}[name]
    model = (e.get(var) or DEFAULTS[name]).strip()
    try:
        timeout = float(e.get("LLM_TIMEOUT_S") or 60)
        if timeout <= 0: raise ValueError
    except ValueError:
        raise ConfigError(name, model, "LLM_TIMEOUT_S must be a positive number")
    if name == "ollama":
        from .ollama import OllamaProvider
        try:
            ctx = int(e.get("OLLAMA_NUM_CTX") or 32768)
            if ctx < 2048: raise ValueError
        except ValueError:
            raise ConfigError(name, model, "OLLAMA_NUM_CTX must be an integer >= 2048")
        return OllamaProvider(e.get("OLLAMA_BASE_URL") or "http://host.docker.internal:11434", model, timeout, num_ctx=ctx)
    key_var = "OPENAI_API_KEY" if name == "openai" else "ANTHROPIC_API_KEY"
    key = (e.get(key_var) or "").strip()
    if not key:
        raise ConfigError(name, model, f"{key_var} is not set (copy .env.example to .env). "
                                       "There is no automatic fallback to another provider; set PROVIDER=ollama for a local model.")
    if name == "openai":
        from .oai import OpenAIProvider
        return OpenAIProvider(key, model, timeout)
    from .claude import AnthropicProvider
    return AnthropicProvider(key, model, timeout)


def max_tokens(env=None) -> int:
    e = os.environ if env is None else env
    try:
        return max(100, int(e.get("LLM_MAX_OUTPUT_TOKENS") or 1000))
    except ValueError:
        raise ConfigError("-", "-", "LLM_MAX_OUTPUT_TOKENS must be an integer")

"""Provider interface. One adapter per PROVIDER value; no fallback chain anywhere."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    schema: dict | None = None      # JSON Schema of the expected reply
    max_tokens: int = 1000
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    in_tokens: int | None = None
    out_tokens: int | None = None


class LLMError(RuntimeError):
    """Base. Always names provider + model so a failure is never attributed to the wrong backend."""
    kind = "llm"

    def __init__(self, provider: str, model: str, msg: str):
        super().__init__(f"[{self.kind}] {provider}/{model}: {msg}")
        self.provider, self.model, self.msg = provider, model, msg


class ConfigError(LLMError): kind = "config"                    # missing key / bad env
class ProviderTimeout(LLMError): kind = "timeout"
class ProviderUnavailable(LLMError): kind = "unavailable"       # connect error, 5xx, 429, model not pulled
class ProviderRejected(LLMError): kind = "rejected"             # 400/401/403 etc.
class GenerationError(LLMError): kind = "generation"            # reply unusable after the one repair retry


class Provider(Protocol):
    name: str
    model: str

    def generate(self, req: LLMRequest) -> LLMResponse: ...

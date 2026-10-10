from __future__ import annotations

import json
import time

from .base import LLMRequest, LLMResponse, ProviderRejected, ProviderTimeout, ProviderUnavailable

TOOL = "emit_plan"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float, client=None):
        self.model = model
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)
        self.client = client

    def generate(self, req: LLMRequest) -> LLMResponse:
        import anthropic
        kw = {}
        if req.schema:  # forced tool call = schema-constrained JSON
            kw["tools"] = [{"name": TOOL, "description": "Return the answer plan.", "input_schema": req.schema}]
            kw["tool_choice"] = {"type": "tool", "name": TOOL}
        t = time.time()
        try:
            r = self.client.messages.create(model=self.model, system=req.system, max_tokens=req.max_tokens,
                                            temperature=req.temperature,
                                            messages=[{"role": "user", "content": req.user}], **kw)
        except anthropic.APITimeoutError as e:
            raise ProviderTimeout(self.name, self.model, str(e))
        except anthropic.APIConnectionError as e:
            raise ProviderUnavailable(self.name, self.model, f"connection: {e}")
        except anthropic.APIStatusError as e:
            c = e.status_code
            cls = ProviderUnavailable if c in (404, 408, 409, 429, 529) or c >= 500 else ProviderRejected
            raise cls(self.name, self.model, f"HTTP {c}: {e.message}")
        text = ""
        for b in r.content:
            if getattr(b, "type", "") == "tool_use":
                text = json.dumps(b.input, ensure_ascii=False); break
            if getattr(b, "type", "") == "text":
                text += b.text
        u = r.usage
        return LLMResponse(text, self.name, self.model, time.time() - t,
                           getattr(u, "input_tokens", None), getattr(u, "output_tokens", None))

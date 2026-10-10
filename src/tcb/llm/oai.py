from __future__ import annotations

import time

from .base import LLMRequest, LLMResponse, ProviderRejected, ProviderTimeout, ProviderUnavailable


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float, client=None):
        self.model = model
        if client is None:
            import openai
            client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0)  # no hidden SDK retries
        self.client = client

    def generate(self, req: LLMRequest) -> LLMResponse:
        import openai
        kw = {}
        if req.schema:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "plan", "schema": req.schema, "strict": False}}
        t = time.time()
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=req.temperature, max_completion_tokens=req.max_tokens,
                messages=[{"role": "system", "content": req.system}, {"role": "user", "content": req.user}], **kw)
        except openai.APITimeoutError as e:
            raise ProviderTimeout(self.name, self.model, str(e))
        except openai.APIConnectionError as e:
            raise ProviderUnavailable(self.name, self.model, f"connection: {e}")
        except openai.APIStatusError as e:
            c = e.status_code
            cls = ProviderUnavailable if c in (404, 408, 409, 429) or c >= 500 else ProviderRejected
            raise cls(self.name, self.model, f"HTTP {c}: {e.message}")
        ch = r.choices[0]
        text = ch.message.content or ""
        if not text and getattr(ch.message, "refusal", None):
            raise ProviderRejected(self.name, self.model, f"model refusal: {ch.message.refusal}")
        u = r.usage
        return LLMResponse(text, self.name, self.model, time.time() - t,
                           getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None))

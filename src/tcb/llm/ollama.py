from __future__ import annotations

import time

import httpx

from .base import LLMRequest, LLMResponse, ProviderRejected, ProviderTimeout, ProviderUnavailable


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float, client: httpx.Client | None = None, num_ctx: int = 12288):
        self.model, self.base, self.num_ctx = model, base_url.rstrip("/"), num_ctx   # Ollama's default context (2-4k) would silently truncate our ~5k-token prompts
        self.client = client or httpx.Client(timeout=timeout)

    def generate(self, req: LLMRequest) -> LLMResponse:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
                "options": {"temperature": req.temperature, "num_predict": req.max_tokens, "num_ctx": self.num_ctx}}
        if req.schema:
            body["format"] = req.schema
        t = time.time()
        try:
            r = self.client.post(f"{self.base}/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, self.model, f"{type(e).__name__} (local models are slow on CPU; raise LLM_TIMEOUT_S)")
        except httpx.TransportError as e:
            raise ProviderUnavailable(self.name, self.model, f"cannot reach {self.base}: {e}. Is Ollama running?")
        if r.status_code >= 400:
            msg = r.text[:300]
            cls = ProviderUnavailable if r.status_code == 404 or r.status_code >= 500 else ProviderRejected
            hint = f" (try: ollama pull {self.model})" if r.status_code == 404 else ""
            raise cls(self.name, self.model, f"HTTP {r.status_code}: {msg}{hint}")
        j = r.json()
        return LLMResponse(j.get("message", {}).get("content", ""), self.name, self.model, time.time() - t,
                           j.get("prompt_eval_count"), j.get("eval_count"))

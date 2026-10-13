"""Unit (mocked SDK clients / httpx.MockTransport). Proves adapter + error-mapping + retry logic only.
A pass here is not evidence that OpenAI, Anthropic or Ollama work with this code; see tests/test_live.py."""
import json
from types import SimpleNamespace as NS

import anthropic
import httpx
import openai
import pytest

from tcb import gen, llm, plan
from tcb.llm import (ConfigError, GenerationError, LLMRequest, ProviderRejected, ProviderTimeout, ProviderUnavailable)
from tcb.llm.claude import AnthropicProvider
from tcb.llm.oai import OpenAIProvider
from tcb.llm.ollama import OllamaProvider

pytestmark = pytest.mark.unit
REQ = LLMRequest("sys", "usr", plan.SCHEMA, 500)
GOOD = json.dumps({"status": "answered", "claims": [{"text_vi": "CASA 40,4%", "evidence": ["E1"]}]})
HX = httpx.Request("POST", "http://x")


def resp(code): return httpx.Response(code, request=HX)


# ---------------- factory ----------------
def test_default_is_openai_and_needs_key():
    with pytest.raises(ConfigError) as e: llm.get({})
    assert "OPENAI_API_KEY" in str(e.value) and "no automatic fallback" in str(e.value).lower()
    assert e.value.provider == "openai" and e.value.model == "gpt-4o-mini"


def test_anthropic_needs_key():
    with pytest.raises(ConfigError) as e: llm.get({"PROVIDER": "anthropic"})
    assert "ANTHROPIC_API_KEY" in str(e.value) and e.value.model == "claude-haiku-5-5"


def test_unknown_provider_and_bad_timeout():
    with pytest.raises(ConfigError): llm.get({"PROVIDER": "gemini"})
    with pytest.raises(ConfigError): llm.get({"PROVIDER": "ollama", "LLM_TIMEOUT_S": "abc"})
    with pytest.raises(ConfigError): llm.get({"PROVIDER": "ollama", "LLM_TIMEOUT_S": "0"})


def test_factory_builds_exactly_one_adapter_each():
    o = llm.get({"PROVIDER": "openai", "OPENAI_API_KEY": "k"}); assert (o.name, o.model) == ("openai", "gpt-4o-mini")
    o = llm.get({"PROVIDER": "OpenAI", "OPENAI_API_KEY": "k", "OPENAI_MODEL": "gpt-x"}); assert o.model == "gpt-x"
    a = llm.get({"PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}); assert (a.name, a.model) == ("anthropic", "claude-haiku-5-5")
    l = llm.get({"PROVIDER": "ollama"}); assert (l.name, l.model) == ("ollama", "qwen2.5:7b")
    assert l.base == "http://host.docker.internal:11434"


def test_sdk_clients_have_no_hidden_retries():
    o = llm.get({"PROVIDER": "openai", "OPENAI_API_KEY": "k"}); assert o.client.max_retries == 0
    a = llm.get({"PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}); assert a.client.max_retries == 0


def test_max_tokens_env():
    assert llm.max_tokens({}) == 1000 and llm.max_tokens({"LLM_MAX_OUTPUT_TOKENS": "300"}) == 300
    with pytest.raises(ConfigError): llm.max_tokens({"LLM_MAX_OUTPUT_TOKENS": "x"})


# ---------------- openai ----------------
class FakeOpenAI:
    def __init__(self, result=None, exc=None):
        self.calls, self.result, self.exc = [], result, exc
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc: raise self.exc
        return self.result


def oai_ok(text=GOOD, refusal=None):
    return NS(choices=[NS(message=NS(content=text, refusal=refusal))], usage=NS(prompt_tokens=10, completion_tokens=5))


def test_openai_ok_and_request_shape():
    c = FakeOpenAI(oai_ok()); r = OpenAIProvider("k", "gpt-4o-mini", 5, client=c).generate(REQ)
    assert r.text == GOOD and (r.provider, r.model, r.in_tokens, r.out_tokens) == ("openai", "gpt-4o-mini", 10, 5)
    kw = c.calls[0]
    assert kw["temperature"] == 0 and kw["max_completion_tokens"] == 500
    assert kw["response_format"]["type"] == "json_schema" and kw["response_format"]["json_schema"]["schema"] == plan.SCHEMA


@pytest.mark.parametrize("exc,cls", [
    (openai.APITimeoutError(request=HX), ProviderTimeout),
    (openai.APIConnectionError(request=HX), ProviderUnavailable),
    (openai.AuthenticationError("bad key", response=resp(401), body=None), ProviderRejected),
    (openai.BadRequestError("bad", response=resp(400), body=None), ProviderRejected),
    (openai.RateLimitError("slow", response=resp(429), body=None), ProviderUnavailable),
    (openai.InternalServerError("boom", response=resp(500), body=None), ProviderUnavailable),
    (openai.NotFoundError("no model", response=resp(404), body=None), ProviderUnavailable),
])
def test_openai_errors(exc, cls):
    with pytest.raises(cls) as e: OpenAIProvider("k", "gpt-4o-mini", 5, client=FakeOpenAI(exc=exc)).generate(REQ)
    assert e.value.provider == "openai" and e.value.model == "gpt-4o-mini"


def test_openai_model_refusal():
    with pytest.raises(ProviderRejected): OpenAIProvider("k", "m", 5, client=FakeOpenAI(oai_ok("", "no"))).generate(REQ)


# ---------------- anthropic ----------------
class FakeAnth:
    def __init__(self, result=None, exc=None):
        self.calls, self.result, self.exc = [], result, exc
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc: raise self.exc
        return self.result


def test_anthropic_tool_json():
    res = NS(content=[NS(type="tool_use", input=json.loads(GOOD))], usage=NS(input_tokens=7, output_tokens=3))
    c = FakeAnth(res); r = AnthropicProvider("k", "claude-sonnet-5-5", 5, client=c).generate(REQ)
    assert json.loads(r.text)["status"] == "answered" and r.in_tokens == 7
    kw = c.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "emit_plan"} and "temperature" not in kw and kw["system"] == "sys"


@pytest.mark.parametrize("exc,cls", [
    (anthropic.APITimeoutError(request=HX), ProviderTimeout),
    (anthropic.APIConnectionError(request=HX), ProviderUnavailable),
    (anthropic.AuthenticationError("bad", response=resp(401), body=None), ProviderRejected),
    (anthropic.InternalServerError("boom", response=resp(529), body=None), ProviderUnavailable),
])
def test_anthropic_errors(exc, cls):
    with pytest.raises(cls) as e: AnthropicProvider("k", "claude-sonnet-5-5", 5, client=FakeAnth(exc=exc)).generate(REQ)
    assert e.value.provider == "anthropic"


# ---------------- ollama ----------------
def ollama(handler): return OllamaProvider("http://o:11434/", "qwen2.5:7b", 5, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ollama_ok_and_body():
    seen = {}
    def h(rq):
        seen["url"], seen["body"] = str(rq.url), json.loads(rq.content)
        return httpx.Response(200, json={"message": {"content": GOOD}, "prompt_eval_count": 11, "eval_count": 4})
    r = ollama(h).generate(REQ)
    assert r.text == GOOD and (r.provider, r.model, r.in_tokens, r.out_tokens) == ("ollama", "qwen2.5:7b", 11, 4)
    assert seen["url"] == "http://o:11434/api/chat" and seen["body"]["format"] == plan.SCHEMA
    assert seen["body"]["stream"] is False and seen["body"]["options"]["temperature"] == 0
    assert seen["body"]["options"]["num_ctx"] == 32768          # default Ollama context would truncate the prompt


def test_ollama_num_ctx_env():
    assert llm.get({"PROVIDER": "ollama", "OLLAMA_NUM_CTX": "16384"}).num_ctx == 16384
    with pytest.raises(ConfigError): llm.get({"PROVIDER": "ollama", "OLLAMA_NUM_CTX": "100"})


def test_ollama_unreachable_timeout_missing_model():
    def refuse(rq): raise httpx.ConnectError("refused")
    with pytest.raises(ProviderUnavailable) as e: ollama(refuse).generate(REQ)
    assert "Is Ollama running" in str(e.value)
    def slow(rq): raise httpx.ReadTimeout("slow")
    with pytest.raises(ProviderTimeout): ollama(slow).generate(REQ)
    with pytest.raises(ProviderUnavailable) as e: ollama(lambda rq: httpx.Response(404, text="model not found")).generate(REQ)
    assert "ollama pull qwen2.5:7b" in str(e.value)
    with pytest.raises(ProviderUnavailable): ollama(lambda rq: httpx.Response(500, text="x")).generate(REQ)
    with pytest.raises(ProviderRejected): ollama(lambda rq: httpx.Response(400, text="x")).generate(REQ)


# ---------------- repair retry (same provider, exactly once) ----------------
class Seq:
    """Scripted provider; counts calls. Stands in for ANY real provider."""
    name, model = "scripted", "m1"

    def __init__(self, *texts): self.texts, self.calls = list(texts), []

    def generate(self, req):
        self.calls.append(req)
        return llm.LLMResponse(self.texts[len(self.calls) - 1], self.name, self.model, 0.0)


def test_valid_first_one_call():
    p = Seq(GOOD); g = gen.generate_plan(p, "s", "u")
    assert len(p.calls) == 1 and not g.repaired and g.plan.status == "answered" and g.problems == []


def test_invalid_then_valid_two_calls_and_error_fed_back():
    p = Seq("not json", GOOD); g = gen.generate_plan(p, "s", "u")
    assert len(p.calls) == 2 and g.repaired and g.plan.claims[0].evidence == ["E1"]
    assert "Lỗi cần sửa" in p.calls[1].user and "not json" in p.calls[1].user


def test_schema_invalid_then_valid():
    p = Seq(json.dumps({"status": "maybe", "claims": []}), GOOD); g = gen.generate_plan(p, "s", "u")
    assert len(p.calls) == 2 and "status" in p.calls[1].user and g.plan.status == "answered"


def test_invalid_twice_errors_after_exactly_two_calls_naming_provider():
    p = Seq("x", "{still bad")
    with pytest.raises(GenerationError) as e: gen.generate_plan(p, "s", "u")
    assert len(p.calls) == 2 and e.value.provider == "scripted" and e.value.model == "m1"


def test_semantic_check_triggers_single_retry_and_keeps_best():
    bad = json.dumps({"status": "answered", "claims": [{"text_vi": "x", "evidence": ["E9"]}]})
    chk = lambda pl: [f"unknown evidence {e}" for c in pl.claims for e in c.evidence if e != "E1"]
    p = Seq(bad, GOOD); g = gen.generate_plan(p, "s", "u", check=chk)
    assert len(p.calls) == 2 and g.problems == []
    p = Seq(bad, bad); g = gen.generate_plan(p, "s", "u", check=chk)
    assert len(p.calls) == 2 and g.problems == ["unknown evidence E9"]       # caller drops the claim; no 3rd call
    p = Seq(bad, "garbage"); g = gen.generate_plan(p, "s", "u", check=chk)
    assert len(p.calls) == 2 and g.problems                                  # parseable first plan kept


def test_provider_error_is_not_retried_or_swapped():
    class Boom:
        name, model, n = "boom", "b", 0
        def generate(self, req): self.n += 1; raise ProviderTimeout("boom", "b", "t")
    b = Boom()
    with pytest.raises(ProviderTimeout): gen.generate_plan(b, "s", "u")
    assert b.n == 1


def test_plan_parse_accepts_fenced_json():
    assert plan.parse("```json\n" + GOOD + "\n```").status == "answered"
    with pytest.raises(ValueError): plan.parse(json.dumps({"status": "answered", "claims": [{"text_vi": ""}]}))

"""Dify shows/aggregates `usage.total_price`; it must equal what ZenMux billed when ZenMux reports it."""

import json
from decimal import Decimal
from types import SimpleNamespace

from dify_plugin.entities.model.message import UserPromptMessage

from models._common import apply_reported_cost

MODEL = "openai/gpt-4.1-nano"
COST_USAGE = {
    "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
    "prompt_tokens_details": {"cached_tokens": 800},
    "cost": "0.00031", "cost_details": {"prompt": "0.00002", "input_cache_read": "0.00001", "completion": "0.00028"},
}


class FakeResponse:
    def __init__(self, lines=None, body=None):
        self._lines, self._body = lines or [], body

    def iter_lines(self, decode_unicode=True, delimiter="\n\n"):
        yield from self._lines

    def json(self):
        return self._body


def _openai(llm):
    model = llm._get_model_class_for_model(MODEL)
    model.started_at = 0.0
    return model


def test_reported_cost_overrides_yaml_estimate(llm):
    model = _openai(llm)
    estimate = model._calc_response_usage(MODEL, {}, 1000, 200)
    usage = apply_reported_cost(estimate, COST_USAGE)
    assert usage.total_price == Decimal("0.00031")
    assert usage.completion_price == Decimal("0.00028")
    assert usage.prompt_price == Decimal("0.00003")  # prompt + cache read
    assert usage.prompt_tokens == 1000 and usage.completion_tokens == 200


def test_without_reported_cost_keeps_estimate(llm):
    model = _openai(llm)
    estimate = model._calc_response_usage(MODEL, {}, 1000, 200)
    assert apply_reported_cost(estimate, {"prompt_tokens": 1000}) is estimate
    assert estimate.total_price > 0


def test_openai_non_stream_uses_reported_cost(llm):
    model = _openai(llm)
    creds = {"api_key": "k"}
    model._update_credential(MODEL, creds)
    body = {"id": "x", "model": MODEL, "usage": COST_USAGE,
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
    result = model._handle_generate_response(MODEL, creds, FakeResponse(body=body), [UserPromptMessage(content="hi")])
    assert result.usage.total_price == Decimal("0.00031")


def test_openai_stream_uses_reported_cost(llm):
    model = _openai(llm)
    creds = {"api_key": "k"}
    model._update_credential(MODEL, creds)
    lines = [
        "data: " + json.dumps({"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}),
        "data: " + json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
        "data: " + json.dumps({"choices": [], "usage": COST_USAGE}),
        "data: [DONE]",
    ]
    chunks = list(model._handle_generate_stream_response(MODEL, creds, FakeResponse(lines), [UserPromptMessage(content="hi")]))
    final = chunks[-1].delta.usage
    assert final.total_price == Decimal("0.00031")
    assert final.prompt_tokens == 1000


def test_anthropic_usage_counts_cache_and_cost(llm):
    model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    model.started_at = 0.0
    raw = {"input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000,
           "output_tokens": 50, "cost": "0.0009", "cost_details": {"completion": "0.00025"}}
    usage = model._usage("anthropic/claude-haiku-4.5", {}, raw)
    assert usage.prompt_tokens == 1110
    assert usage.total_price == Decimal("0.0009")
    assert usage.completion_price == Decimal("0.00025")


def test_anthropic_stream_collects_usage_from_message_delta(llm):
    model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    model.started_at = 0.0

    def ev(**kw):
        return SimpleNamespace(**kw)

    def usage_obj(d):
        return SimpleNamespace(model_dump=lambda exclude_none=False: d, **d)

    events = [
        ev(type="message_start", message=ev(usage=usage_obj({"input_tokens": 8, "output_tokens": 1}))),
        ev(type="content_block_start", index=0, content_block=ev(type="text")),
        ev(type="content_block_delta", index=0, delta=ev(type="text_delta", text="hi")),
        ev(type="content_block_stop", index=0),
        ev(type="message_delta", usage=usage_obj({"input_tokens": 8, "output_tokens": 8, "cost": "0.000048",
                                                   "cost_details": {"completion": "0.00004"}})),
        ev(type="message_stop"),
    ]

    class Stream:
        def __enter__(self):
            return iter(events)

        def __exit__(self, *a):
            return False

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream()))
    chunks = list(model._stream_generate("anthropic/claude-haiku-4.5", {}, client, {}, {}, [], False))
    usage = chunks[-1].delta.usage
    assert (usage.prompt_tokens, usage.completion_tokens) == (8, 8)
    assert usage.total_price == Decimal("0.000048")

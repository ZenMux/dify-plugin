"""What a model does when the user leaves a parameter alone must match the default its YAML shows.

The SDK fills in `default` only for required parameters, so a switch the user never touched reaches
the plugin as absent, while Dify keeps showing the declared default. 0.0.9 showed "Thinking Mode: off"
on 81 OpenAI-compatible models that reasoned when the switch was absent, and a switch on 35 models
that cannot turn reasoning off. Now the plugin sends the declared default itself, every switch defaults
to off, and a switch is shown only where off is verified to work. These tests check the whole catalog.
"""

import json
import sys
from types import SimpleNamespace

import pytest
import yaml
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from dify_plugin.entities.model.message import AssistantPromptMessage
from google.genai import types

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import zenmux_models as gen  # noqa: E402
from models._common import OWNED_SWITCHES, strip_reasoning_text  # noqa: E402
from models.llm.anthropic_llm import THINKING_OFF  # noqa: E402
from models.llm.google import ZenMuxGoogleLargeLanguageModel  # noqa: E402

PROBES = json.loads((ROOT / "catalog" / "probes.json").read_text())["models"]
LLMS = [yaml.safe_load(p.read_text()) for p in sorted((ROOT / "models" / "llm").glob("[!_]*.yaml"))]


def _rules(data):
    return {r["name"]: r for r in data.get("parameter_rules") or []}


SWITCHED = [d["model"] for d in LLMS if "agent-thought" in (d.get("features") or []) and gen.reasoning_switch(d["model"])]


def test_models_that_do_not_reason_have_no_thinking_switch():
    for data in LLMS:
        if "agent-thought" not in (data.get("features") or []):
            assert not set(_rules(data)) & {*OWNED_SWITCHES, "exclude_reasoning_tokens"}, data["model"]


@pytest.mark.parametrize("mid", SWITCHED)
def test_thinking_switch_is_shown_only_where_off_works_and_defaults_to_off(mid):
    data = next(d for d in LLMS if d["model"] == mid)
    probe = (PROBES.get(mid) or {}).get("reasoning_switch")
    assert probe, f"{mid}: not probed; run `probe --only '{mid}' --checks reasoning`"
    name, rules = gen.reasoning_switch(mid), _rules(data)
    assert name in OWNED_SWITCHES
    if probe.get("off") is False and True in (probe.get("absent"), probe.get("on")):
        assert rules[name]["default"] is False, (mid, probe)
    else:
        assert name not in rules, (mid, probe)


def test_switch_needs_a_verified_off():
    works = {"absent": True, "off": False, "on": True}
    assert gen.reasoning_rule(gen.ENABLE_THINKING, {"reasoning_switch": works}) == [{**gen.ENABLE_THINKING, "default": False}]
    for probe in ({}, {"reasoning_switch": {**works, "off": "error"}}, {"reasoning_switch": {**works, "off": True}},
                  {"reasoning_switch": {**works, "off": "rejected"}}, {"reasoning_switch": {"absent": False, "off": False, "on": False}}):
        assert gen.reasoning_rule(gen.ENABLE_THINKING, probe) == [], probe


def _sent(llm, monkeypatch, mid):
    """model_parameters the route hands to its request builder for a node the user never touched."""
    model = llm._get_model_class_for_model(mid)
    captured = {}

    def fake_generate(_model, _credentials, _prompt, params, *_a, **_k):
        captured.update(params)
        return LLMResult(model=mid, message=AssistantPromptMessage(content="ok"), usage=LLMUsage.empty_usage())

    monkeypatch.setattr(model, "_generate", fake_generate)
    model._invoke(mid, {"api_key": "x"}, [], llm._validate_and_filter_model_parameters(mid, {}, {}), stream=False)
    return captured


def _claude_request(llm, monkeypatch, mid):
    """The Messages API request the Anthropic route sends for a node the user never touched."""
    model = llm._get_model_class_for_model(mid)
    model.started_at = 0.0
    captured = {}

    def create(**params):
        captured.update(params)
        usage = SimpleNamespace(model_dump=lambda exclude_none=False: {"input_tokens": 1, "output_tokens": 1})
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")], usage=usage)

    monkeypatch.setattr(model, "_get_client", lambda _c: SimpleNamespace(messages=SimpleNamespace(create=create)))
    model._invoke(mid, {"api_key": "x"}, [], llm._validate_and_filter_model_parameters(mid, {}, {}), stream=False)
    return captured


@pytest.mark.parametrize("data", LLMS, ids=[d["model"] for d in LLMS])
def test_untouched_reasoning_parameters_do_what_the_yaml_shows(llm, monkeypatch, data):
    mid, route, rules = data["model"], gen.protocol_of(data["model"]), _rules(data)
    assert rules.get("exclude_reasoning_tokens", {}).get("default", False) is False
    if route == "openai":
        sent = _sent(llm, monkeypatch, mid)
        # Declared off is sent as off; without a switch nothing is sent and the model reasons as upstream does.
        assert sent.get("reasoning") == ({"enabled": False} if "enable_thinking" in rules else None), (mid, sent)
    elif route == "anthropic":
        thinking = _claude_request(llm, monkeypatch, mid).get("thinking")
        if "enable_thinking" in rules:
            # Newer Claude models think unless told not to, so "off" must be explicit.
            assert rules["enable_thinking"]["default"] is False and thinking == {"type": THINKING_OFF[mid]}, mid
        else:
            assert thinking is None, mid
    else:
        sent = _sent(llm, monkeypatch, mid)
        config = types.GenerateContentConfig()
        ZenMuxGoogleLargeLanguageModel._set_thinking_config(config=config, model_parameters=sent, model_name=mid)
        budget = config.thinking_config.thinking_budget if config.thinking_config else None
        assert budget == (0 if "thinking_mode" in rules else None), (mid, budget)
        assert not (config.thinking_config and config.thinking_config.include_thoughts), mid


def test_set_values_win_over_declared_defaults(llm, monkeypatch):
    mid = next(m for m in SWITCHED if gen.protocol_of(m) == "openai" and "enable_thinking" in _rules(next(d for d in LLMS if d["model"] == m)))
    model = llm._get_model_class_for_model(mid)
    captured = {}
    monkeypatch.setattr(model, "_generate", lambda _m, _c, _p, params, *a, **k: captured.update(params) or
                        LLMResult(model=mid, message=AssistantPromptMessage(content="ok"), usage=LLMUsage.empty_usage()))
    model._invoke(mid, {"api_key": "x"}, [], {"max_tokens": 16, "enable_thinking": True}, stream=False)
    assert captured["reasoning"] == {"enabled": True}


THINK = "<think>\nplan\n</think>"


def _openai_invoke(llm, monkeypatch, stream, hide):
    mid = next(d["model"] for d in LLMS if gen.protocol_of(d["model"]) == "openai" and "agent-thought" in d["features"])
    model = llm._get_model_class_for_model(mid)

    def fake_generate(*_a, **_k):
        if stream:
            return (LLMResultChunk(model=mid, delta=LLMResultChunkDelta(index=i, message=AssistantPromptMessage(content=c)))
                    for i, c in enumerate([THINK, "Answer"]))
        return LLMResult(model=mid, message=AssistantPromptMessage(content=THINK + "Answer"), usage=LLMUsage.empty_usage())

    monkeypatch.setattr(model, "_generate", fake_generate)
    result = model._invoke(mid, {}, [], {"exclude_reasoning_tokens": hide}, stream=stream)
    return "".join(c.delta.message.content for c in result) if stream else result.message.content


def _anthropic_invoke(llm, stream, hide):
    mid = "anthropic/claude-haiku-4.5"
    model = llm._get_model_class_for_model(mid)
    model.started_at = 0.0
    usage = SimpleNamespace(model_dump=lambda exclude_none=False: {"input_tokens": 1, "output_tokens": 1})
    if not stream:
        response = SimpleNamespace(usage=usage, content=[SimpleNamespace(type="thinking", thinking="plan"),
                                                         SimpleNamespace(type="text", text="Answer")])
        return model._parse_non_stream(mid, {}, response, [], hide).message.content
    ev = lambda **kw: SimpleNamespace(**kw)  # noqa: E731
    events = [ev(type="message_start", message=ev(usage=usage)),
              ev(type="content_block_start", index=0, content_block=ev(type="thinking")),
              ev(type="content_block_delta", index=0, delta=ev(type="thinking_delta", thinking="plan")),
              ev(type="content_block_stop", index=0),
              ev(type="content_block_start", index=1, content_block=ev(type="text")),
              ev(type="content_block_delta", index=1, delta=ev(type="text_delta", text="Answer")),
              ev(type="content_block_stop", index=1), ev(type="message_stop")]

    class Stream:
        def __enter__(self):
            return iter(events)

        def __exit__(self, *a):
            return False

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream()))
    return "".join(c.delta.message.content or "" for c in model._stream_generate(mid, {}, client, {}, {}, [], hide))


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "blocking"])
@pytest.mark.parametrize("route", sorted(gen.HIDE_REASONING_ROUTES))
def test_hide_thought_process_works_streaming_and_blocking(llm, monkeypatch, route, stream):
    run = (lambda hide: _openai_invoke(llm, monkeypatch, stream, hide)) if route == "openai" else \
        (lambda hide: _anthropic_invoke(llm, stream, hide))
    assert run(True).strip() == "Answer"
    assert "plan" in run(False) and "Answer" in run(False)


def test_hide_drops_a_thought_left_open_by_max_tokens():
    # A reply cut off inside the thought: the stream filter drops it, so the blocking path must too.
    assert strip_reasoning_text("<think>\nprivate plan, cut off") == ""
    assert strip_reasoning_text("Before <think>a</think>\nAnswer <think>open") == "Before Answer "
    assert strip_reasoning_text("No thoughts") == "No thoughts"


def _probe_with(monkeypatch, answers):
    """probe_reasoning_switch over scripted (status, reasons) answers per state."""
    queues = {state: list(seq) for state, seq in answers.items()}
    monkeypatch.setattr(gen, "_thinks", lambda _mid, state, _key: queues[{None: "absent", False: "off", True: "on"}[state]].pop(0))
    return gen.probe_reasoning_switch("vendor/model", "key")


def test_reasoning_probe_needs_two_agreeing_answers_and_ignores_transient_failures(monkeypatch):
    ok, think, no = 200, True, False
    result = _probe_with(monkeypatch, {
        "absent": [(ok, think), (429, None), (ok, think)],           # a rate limit in between is retried
        "off": [(400, None), (400, None)],                           # rejected twice
        "on": [(ok, think), (500, None), (0, None), (429, None)],    # one answer, then only transient failures
    })
    assert result == {"absent": True, "off": "rejected", "on": "error"}
    assert _probe_with(monkeypatch, {"absent": [(ok, think), (ok, no)], "off": [(ok, no)] * 2,
                                     "on": [(ok, think)] * 2})["absent"] == "unstable"


def test_a_transient_probe_failure_keeps_only_that_earlier_answer():
    prev = {"reasoning_switch": {"absent": True, "off": False, "on": True}, "inputs": {"audio": "ok"},
            "input_formats": {"audio": "input_audio"}}
    res = {"reasoning_switch": {"absent": True, "off": "rejected", "on": "error"}, "inputs": {"audio": "error:429 busy"},
           "input_formats": {}}
    merged = gen.merge_probe(prev, res)
    assert merged["reasoning_switch"] == {"absent": True, "off": "rejected", "on": True}
    assert merged["inputs"] == {"audio": "ok"} and merged["input_formats"] == {"audio": "input_audio"}

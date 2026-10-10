"""Contracts between what the generated YAMLs promise and what each route's runtime code does.

Review of 0.0.9 found three gaps that per-feature tests missed: file inputs advertised on a route
whose converter dropped them, a plugin-level option lost when the generator was rewritten, and a
parameter adjustment that left the model's declared range. These tests check the whole catalog.
"""

import base64
import json
import sys

import pytest
import requests
import yaml
from dify_plugin.entities.model.message import (
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    TextPromptMessageContent,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from google.genai import types

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import zenmux_models as gen  # noqa: E402
from models._common import keep_supported_files  # noqa: E402
from models.llm.anthropic_llm import ZenMuxAnthropicLargeLanguageModel, fit_thinking_budget  # noqa: E402
from models.llm.google import ZenMuxGoogleLargeLanguageModel  # noqa: E402
from models.llm.openai import INPUT_FORMAT_CANDIDATES, INPUT_FORMATS, PART_FORMATS, ZenMuxOpenAICCLargeLanguageModel  # noqa: E402

PAYLOAD = base64.b64encode(b"contract-test-payload").decode()
SAMPLES = {
    "image": ImagePromptMessageContent(format="png", mime_type="image/png", base64_data=PAYLOAD),
    "file": DocumentPromptMessageContent(format="pdf", mime_type="application/pdf", base64_data=PAYLOAD, filename="a.pdf"),
    "audio": AudioPromptMessageContent(format="wav", mime_type="audio/wav", base64_data=PAYLOAD, filename="a.wav"),
    "video": VideoPromptMessageContent(format="mp4", mime_type="video/mp4", base64_data=PAYLOAD, filename="a.mp4"),
}


def _llm_yamls():
    for path in sorted((ROOT / "models" / "llm").glob("[!_]*.yaml")):
        yield yaml.safe_load(path.read_text())


def _delivered(route, modality, content=None):
    """Serialized request part(s) the route builds for one sample file."""
    content = content or SAMPLES[modality]
    if route == "openai":
        formats = INPUT_FORMAT_CANDIDATES.get(modality, [None])
        return [json.dumps(ZenMuxOpenAICCLargeLanguageModel._content_part(content, {modality: f})) for f in formats]
    if route == "anthropic":
        cls = ZenMuxAnthropicLargeLanguageModel
        return [json.dumps(cls._build_content_blocks(cls.__new__(cls), [content]))]
    cls = ZenMuxGoogleLargeLanguageModel
    gemini = cls._format_message_to_gemini_content(cls.__new__(cls), UserPromptMessage(content=[content]),
                                                   None, types.GenerateContentConfig())
    return [str([base64.b64encode(p.inline_data.data).decode() for p in gemini.parts if p.inline_data])]


@pytest.mark.parametrize("route,modality", [(r, m) for r, mods in gen.ROUTE_INPUTS.items() for m in sorted(mods)])
def test_route_delivers_every_input_it_may_advertise(route, modality):
    # Every candidate format the route can be configured with must carry the file itself.
    for part in _delivered(route, modality):
        assert PAYLOAD in part, (route, modality, part)


ROUTE_X_INPUT = [(r, m) for r, mods in gen.ROUTE_INPUTS.items() for m in sorted(mods)]


def _serve(monkeypatch, status, body):
    """Make every requests.get return a real Response, so raise_for_status behaves as in production."""
    def fake_get(url, *args, **kwargs):
        response = requests.models.Response()
        response.status_code, response._content, response.url = status, body, url
        response.headers["Content-Type"] = "text/html" if status >= 400 else "application/octet-stream"
        return response

    monkeypatch.setattr("requests.get", fake_get)


def _url_only(modality):
    return SAMPLES[modality].model_copy(update={"base64_data": "", "url": "https://dify.internal/files/x?sign=secret"})


@pytest.mark.parametrize("route,modality", ROUTE_X_INPUT)
def test_url_only_files_are_inlined(monkeypatch, route, modality):
    # Dify with MULTIMODAL_SEND_FORMAT=url passes only its own file URL, which upstreams often cannot reach
    # or refuse ("Invalid file data"), so every route must fetch it and send the bytes.
    _serve(monkeypatch, 200, b"contract-test-payload")
    for part in _delivered(route, modality, _url_only(modality)):
        assert PAYLOAD in part and "dify.internal" not in part, (route, modality, part)


@pytest.mark.parametrize("status", [403, 404, 500])
@pytest.mark.parametrize("route,modality", ROUTE_X_INPUT)
def test_failed_downloads_are_errors_not_content(monkeypatch, route, modality, status):
    # Review of 3d07d3c: an HTML error page was encoded and sent to the model as the file.
    _serve(monkeypatch, status, b"<html>error page</html>")
    with pytest.raises(Exception) as caught:
        _delivered(route, modality, _url_only(modality))
    assert f"HTTP {status}" in str(caught.value) and "sign=secret" not in str(caught.value), caught.value


def test_advertised_inputs_are_deliverable_and_verified():
    probes = json.loads((ROOT / "catalog" / "probes.json").read_text())["models"]
    overrides = yaml.safe_load((ROOT / "catalog" / "overrides.yaml").read_text()) or {}
    for data in _llm_yamls():
        mid, route = data["model"], gen.protocol_of(data["model"])
        forced = set((overrides.get("models") or {}).get(mid, {}).get("features_add") or [])
        for modality, feature in gen.MODAL_FEATURES.items():
            if feature in (data.get("features") or []) and feature not in forced:
                assert modality in gen.ROUTE_INPUTS[route], (mid, feature)
                assert ((probes.get(mid) or {}).get("inputs") or {}).get(modality) == "ok", (mid, feature)


def test_probed_input_formats_exist_and_are_used():
    for mid, formats in INPUT_FORMATS.items():
        for modality, fmt in formats.items():
            assert fmt in PART_FORMATS and fmt in INPUT_FORMAT_CANDIDATES[modality], (mid, modality, fmt)
    msg = UserPromptMessage(content=[TextPromptMessageContent(data="hi"), SAMPLES["audio"], SAMPLES["video"]])
    cls = ZenMuxOpenAICCLargeLanguageModel
    parts = cls._convert_prompt_message_to_dict(cls.__new__(cls), msg, {"input_formats": {"audio": "input_audio_uri"}})["content"]
    assert [p["type"] for p in parts] == ["text", "input_audio", "video_url"]
    assert parts[1]["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_undeclared_files_become_a_note_and_nothing_disappears():
    msg = UserPromptMessage(content=[TextPromptMessageContent(data="summarise"), *SAMPLES.values()])
    kept = keep_supported_files([msg], ["vision", "document"])[0].content
    assert [getattr(p, "type", None) for p in kept] == ["text", "image", "document", "text", "text"]
    assert [p.data for p in kept[3:]] == ["[Audio: a.wav]", "[Video: a.mp4]"]
    assert keep_supported_files([msg], [])[0].content == "summarise [Image] [File: a.pdf] [Audio: a.wav] [Video: a.mp4]"
    assert keep_supported_files([msg], ["vision", "document", "audio", "video"])[0].content == msg.content


def test_hide_thought_process_survives_parameter_filtering(llm):
    # The SDK drops parameters the schema does not list before the plugin sees them.
    for data in _llm_yamls():
        route = gen.protocol_of(data["model"])
        if "agent-thought" not in (data.get("features") or []) or route not in gen.HIDE_REASONING_ROUTES:
            continue
        kept = llm._validate_and_filter_model_parameters(data["model"], {"exclude_reasoning_tokens": True}, {})
        assert kept.get("exclude_reasoning_tokens") is True, data["model"]


def _claude_budget_models():
    for data in _llm_yamls():
        rules = {r["name"]: r for r in data.get("parameter_rules") or []}
        if "reasoning_budget" in rules:
            yield data["model"], rules["max_tokens"], rules["reasoning_budget"]


@pytest.mark.parametrize("model,max_rule,budget_rule", list(_claude_budget_models()))
def test_claude_budget_never_leaves_the_declared_range(model, max_rule, budget_rule):
    cls = ZenMuxAnthropicLargeLanguageModel
    inst = cls.__new__(cls)
    limit = max_rule["max"]
    for max_tokens in {1, max_rule["default"], budget_rule["min"] + 1, limit}:
        for budget in {budget_rule["min"], budget_rule["default"], budget_rule["max"]}:
            params, _ = inst._build_request_params(model, {"max_tokens": max_tokens}, None, [], None, None,
                                                   thinking={"type": "enabled", "budget_tokens": budget},
                                                   max_tokens_limit=limit)
            sent_max, sent_budget = params["max_tokens"], params["thinking"]["budget_tokens"]
            assert sent_budget < sent_max <= limit, (max_tokens, budget, params)
            if budget < max_tokens:
                assert (sent_max, sent_budget) == (max_tokens, budget)


def test_fit_thinking_budget_examples():
    assert fit_thinking_budget(64000, 63999, 64000) == (64000, 63999)
    assert fit_thinking_budget(3000, 2048, 64000) == (3000, 2048)
    assert fit_thinking_budget(2048, 2048, 64000) == (3072, 2048)
    assert fit_thinking_budget(1024, 63999, 64000) == (64000, 63999)


@pytest.mark.parametrize("route", sorted(gen.ROUTE_INPUTS))
def test_errors_while_streaming_become_dify_errors(llm, monkeypatch, route):
    # Gemini and Anthropic send the request on the first iteration; the SDK's own error transform
    # never sees those exceptions, so Dify showed raw tracebacks (e.g. Gemini 400 on audio-only webm).
    from dify_plugin.errors.model import InvokeBadRequestError, InvokeError
    from google.genai import errors as genai_errors

    upstream = {"google": genai_errors.ClientError(400, {"error": {"message": "Request contains an invalid argument."}}),
                "anthropic": ValueError("boom"), "openai": ValueError("boom")}[route]
    model = next(d["model"] for d in _llm_yamls() if gen.protocol_of(d["model"]) == route)

    def failing_stream(*_a, **_k):
        raise upstream
        yield  # noqa: unreachable - makes this a generator

    monkeypatch.setattr(type(llm.model_map[model]), "_invoke", lambda self, *a, **k: failing_stream())
    with pytest.raises(InvokeError) as caught:
        list(llm._invoke(model, {}, [], {}))
    if route == "google":
        assert isinstance(caught.value, InvokeBadRequestError)
        assert "invalid argument" in str(caught.value)

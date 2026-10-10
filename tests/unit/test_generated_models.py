"""Regression guards for model metadata that once broke real requests."""

import json

import yaml

from conftest import ROOT


def _rules(stem):
    data = yaml.safe_load((ROOT / "models" / "llm" / f"{stem}.yaml").read_text())
    return {r["name"]: r for r in data.get("parameter_rules") or []}


def test_gemini_25_pro_has_no_thinking_off_switch():
    # Turning thinking off sends thinking_budget=0, which gemini-2.5-pro rejects with 400.
    assert "thinking_mode" not in _rules("gemini-2.5-pro")
    assert "thinking_budget" in _rules("gemini-2.5-pro")


def test_claude_temperature_only_where_accepted():
    # Newer Claude models reject any non-default temperature; haiku-4.5 caps it at 1.
    assert "temperature" not in _rules("claude-opus-4.8")
    assert _rules("claude-haiku-4.5")["temperature"]["max"] == 1.0


def test_claude_thinking_style_matches_model():
    assert "reasoning_budget" in _rules("claude-haiku-4.5")
    assert "reasoning_budget" not in _rules("claude-opus-4.8")


def _model(folder, stem):
    return yaml.safe_load((ROOT / "models" / folder / f"{stem}.yaml").read_text())


def test_vision_on_retrieval_models_requires_a_passing_image_probe():
    # Dify's knowledge base sends images to embedding/rerank models that list `vision`; the catalog's
    # input modalities alone are not trusted (some models accept images but ignore their content).
    probes = json.loads((ROOT / "catalog" / "probes.json").read_text())["models"]
    forced = {mid for mid, o in ((yaml.safe_load((ROOT / "catalog" / "overrides.yaml").read_text()) or {})
                                 .get("models") or {}).items() if "vision" in (o.get("features_add") or [])}
    for folder in ("text_embedding", "rerank"):
        for path in (ROOT / "models" / folder).glob("[!_]*.yaml"):
            data = yaml.safe_load(path.read_text())
            expected = (probes.get(data["model"]) or {}).get("image") == "ok" or data["model"] in forced
            assert ("vision" in (data.get("features") or [])) == expected, data["model"]
    assert "vision" in _model("rerank", "qwen3-vl-rerank")["features"]


def test_qwen_vl_embedding_batch_limit():
    # ZenMux rejects more than 20 inputs per request for this model (400).
    assert _model("text_embedding", "qwen3-vl-embedding")["model_properties"]["max_chunks"] == 20


def test_free_and_retired_models_are_not_listed():
    names = {p.stem for p in (ROOT / "models" / "llm").glob("*.yaml")}
    assert not any(n.endswith("-free") for n in names)
    assert not {"claude-opus-4.1", "glm-5-turbo", "ernie-x1.1-preview"} & names

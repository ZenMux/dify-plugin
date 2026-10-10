"""The package must load the way the plugin daemon loads it, and the YAMLs must match the catalog."""

import subprocess
import sys

import yaml
from dify_plugin.entities.model import ModelType

from conftest import ROOT


def _yaml_models(folder):
    return {
        yaml.safe_load(p.read_text())["model"]
        for p in (ROOT / "models" / folder).glob("*.yaml")
        if not p.name.startswith("_")
    }


def test_all_model_types_registered(factory):
    assert set(factory.models) == {ModelType.LLM, ModelType.TEXT_EMBEDDING, ModelType.RERANK}


def test_every_yaml_is_a_valid_schema(registration):
    config, _provider, _factory = registration.models_mapping["zenmux"]
    loaded = {(m.model_type, m.model) for m in config.models}
    expected = (
        {(ModelType.LLM, m) for m in _yaml_models("llm")}
        | {(ModelType.TEXT_EMBEDDING, m) for m in _yaml_models("text_embedding")}
        | {(ModelType.RERANK, m) for m in _yaml_models("rerank")}
    )
    assert loaded == expected


def test_position_files_cover_all_models():
    for folder in ("llm", "text_embedding", "rerank"):
        positions = yaml.safe_load((ROOT / "models" / folder / "_position.yaml").read_text())
        assert len(positions) == len(set(positions))
        assert set(positions) == _yaml_models(folder)


def test_generated_yamls_match_catalog():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "zenmux_models.py"), "check"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr

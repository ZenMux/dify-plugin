import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def registration():
    """Load the plugin exactly like the plugin daemon does (manifest -> provider -> YAMLs -> classes)."""
    from dify_plugin.config.config import DifyPluginEnv
    from dify_plugin.core.plugin_registration import PluginRegistration

    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        yield PluginRegistration(DifyPluginEnv())
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def factory(registration):
    _config, _provider, model_factory = registration.models_mapping["zenmux"]
    return model_factory


@pytest.fixture(scope="session")
def llm(factory):
    from dify_plugin.entities.model import ModelType

    return factory.get_instance(ModelType.LLM)

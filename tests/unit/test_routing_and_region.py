import pytest

from models._common import anthropic_base_url, openai_base_url, vertex_base_url
from models.llm.anthropic_llm import ZenMuxAnthropicLargeLanguageModel
from models.llm.google import ZenMuxGoogleLargeLanguageModel
from models.llm.openai import ZenMuxOpenAICCLargeLanguageModel
from models.llm.zenmux import protocol_of


@pytest.mark.parametrize("model,protocol", [
    ("anthropic/claude-opus-4.8", "anthropic"),
    ("google/gemini-3.5-flash", "google"),
    ("google/gemma-4-31b-it", "openai"),
    ("openai/gpt-5.5", "openai"),
    ("moonshotai/kimi-k2.6", "openai"),
])
def test_protocol_of(model, protocol):
    assert protocol_of(model) == protocol


def test_predefined_models_route_to_protocol_class(llm):
    assert isinstance(llm._get_model_class_for_model("anthropic/claude-haiku-4.5"), ZenMuxAnthropicLargeLanguageModel)
    assert isinstance(llm._get_model_class_for_model("google/gemini-2.5-flash"), ZenMuxGoogleLargeLanguageModel)
    assert isinstance(llm._get_model_class_for_model("openai/gpt-4.1"), ZenMuxOpenAICCLargeLanguageModel)


def test_custom_models_fall_back_to_openai_compatible(llm):
    # Customizable models keep the pre-0.0.9 behaviour regardless of prefix.
    assert llm._get_model_class_for_model("anthropic/some-custom-id") is llm.default_model


def test_error_mapping_merges_all_protocols(llm):
    import anthropic

    sources = [s for group in llm._invoke_error_mapping.values() for s in group]
    assert anthropic.RateLimitError in sources


@pytest.mark.parametrize("region,host", [(None, "https://zenmux.ai"), ("global", "https://zenmux.ai"),
                                         ("cn", "https://zenmux.dev"), ("bogus", "https://zenmux.ai")])
def test_region_hosts(region, host):
    creds = {"api_key": "k"} if region is None else {"api_key": "k", "region": region}
    assert openai_base_url(creds) == f"{host}/api/v1"
    assert anthropic_base_url(creds) == f"{host}/api/anthropic"
    assert vertex_base_url(creds) == f"{host}/api/vertex-ai"


def test_openai_path_uses_region(llm):
    model = llm._get_model_class_for_model("openai/gpt-4.1")
    creds = {"api_key": "k", "region": "cn"}
    model._update_credential("openai/gpt-4.1", creds)
    assert creds["endpoint_url"] == "https://zenmux.dev/api/v1"
    assert creds["function_calling_type"] == "tool_call"


def test_anthropic_client_uses_region(llm):
    model = llm._get_model_class_for_model("anthropic/claude-haiku-4.5")
    client = model._get_client({"api_key": "k", "region": "cn"})
    assert str(client.base_url).rstrip("/") == "https://zenmux.dev/api/anthropic"


from collections.abc import Generator

from dify_plugin import LargeLanguageModel
from dify_plugin.entities.model import AIModelEntity, ModelType
from dify_plugin.errors.model import InvokeError

from .anthropic_llm import ZenMuxAnthropicLargeLanguageModel
from .google import ZenMuxGoogleLargeLanguageModel
from .openai import ZenMuxOpenAICCLargeLanguageModel

PROTOCOL_CLASSES = {
    "anthropic": ZenMuxAnthropicLargeLanguageModel,
    "google": ZenMuxGoogleLargeLanguageModel,
    "openai": ZenMuxOpenAICCLargeLanguageModel,
}


def protocol_of(model: str) -> str:
    """Which ZenMux endpoint serves a predefined model.

    Must stay in sync with scripts/zenmux_models.py::protocol_of, which decides
    the parameter template generated for each model.
    """
    if model.startswith("anthropic/"):
        return "anthropic"
    if model.startswith("google/gemini-"):
        return "google"
    return "openai"


class ZenMuxLargeLanguageModel(LargeLanguageModel):
    """
    Model class for zenmux large language model.
    """

    def __init__(self, model_schemas: list[AIModelEntity]) -> None:
        super().__init__(model_schemas)

        grouped: dict[str, list[AIModelEntity]] = {name: [] for name in PROTOCOL_CLASSES}
        for model_schema in model_schemas:
            if model_schema.model_type == ModelType.LLM:
                grouped[protocol_of(model_schema.model)].append(model_schema)

        instances = {name: PROTOCOL_CLASSES[name](schemas) for name, schemas in grouped.items()}
        self.protocol_models = list(instances.values())
        self.default_model = instances["openai"]
        self.model_map = {
            model_schema.model: instances[name]
            for name, schemas in grouped.items()
            for model_schema in schemas
        }

    def _get_model_class_for_model(self, model: str):
        """
        Predefined models are routed by their provider prefix; customizable models
        (anything not in the predefined list) use the OpenAI-compatible endpoint.

        :param model: Model name (e.g., 'deepseek/deepseek-chat:deepseek')
        :return: Model implementation instance
        """
        return self.model_map.get(model, self.default_model)

    def validate_credentials(self, model: str, *args, **kwargs):
        model_obj = self._get_model_class_for_model(model)
        return model_obj.validate_credentials(model, *args, **kwargs)

    @property
    def _invoke_error_mapping(self):
        merged: dict = {}
        for model_obj in self.protocol_models:
            for invoke_error, sources in model_obj._invoke_error_mapping.items():
                bucket = merged.setdefault(invoke_error, [])
                bucket.extend(s for s in sources if s not in bucket)
        return merged

    def _invoke(self, model: str, *args, **kwargs):
        model_obj = self._get_model_class_for_model(model)
        result = model_obj._invoke(model, *args, **kwargs)
        return self._map_stream_errors(result) if isinstance(result, Generator) else result

    def _transform_invoke_error(self, error: Exception) -> InvokeError:
        """Map with the mapping of the protocol whose SDK raised the error (each keeps its own order),
        instead of the merged mapping, where a generic class of one SDK could shadow a specific one."""
        if isinstance(error, InvokeError):
            return error
        for model_obj in self.protocol_models:
            if any(isinstance(error, tuple(sources)) for sources in model_obj._invoke_error_mapping.values()):
                return model_obj._transform_invoke_error(error)
        return super()._transform_invoke_error(error)

    def _map_stream_errors(self, chunks: Generator) -> Generator:
        """Give errors raised while streaming the same Dify error types as errors raised before it.

        The SDK only transforms exceptions from `_invoke` itself; the Gemini and Anthropic SDKs send
        the request on the first iteration, so their HTTP errors surfaced in Dify as raw tracebacks.
        """
        try:
            yield from chunks
        except InvokeError:
            raise
        except Exception as e:
            raise self._transform_invoke_error(e) from e

    def get_num_tokens(self, model: str, *args, **kwargs):
        model_obj = self._get_model_class_for_model(model)
        return model_obj.get_num_tokens(model, *args, **kwargs)

    def get_customizable_model_schema(self, model: str, credentials: dict):
        model_obj = self._get_model_class_for_model(model)
        return model_obj.get_customizable_model_schema(model, credentials)

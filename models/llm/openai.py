import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    UserPromptMessage,
)

from models._common import (
    apply_reported_cost,
    file_base64,
    file_data_uri,
    keep_supported_files,
    openai_base_url,
    strip_reasoning,
)

logger = logging.getLogger(__name__)

REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


# ZenMux passes files through to each upstream, and upstreams disagree on the part format: the
# documented `file` works for PDFs, while videos need `video_url` on most providers and audio is raw
# or data-URI base64 depending on the provider. The probe tries the candidates in order per model
# and scripts/zenmux_models.py writes the winner to _openai_inputs.json; the first is the default.
PART_FORMATS = {
    "file": lambda c: {"type": "file", "file": {"filename": c.filename or f"{c.type.value}.{c.format}",
                                                "file_data": file_data_uri(c)}},
    "video_url": lambda c: {"type": "video_url", "video_url": {"url": file_data_uri(c)}},
    "input_audio": lambda c: {"type": "input_audio", "input_audio": {"data": file_base64(c), "format": c.format}},
    "input_audio_uri": lambda c: {"type": "input_audio", "input_audio": {"data": file_data_uri(c), "format": c.format}},
}
INPUT_FORMAT_CANDIDATES = {"file": ["file"], "video": ["video_url", "file"], "audio": ["input_audio", "input_audio_uri"]}
INPUT_KIND = {PromptMessageContentType.DOCUMENT: "file", PromptMessageContentType.VIDEO: "video",
              PromptMessageContentType.AUDIO: "audio"}
INPUT_FORMATS = json.loads((Path(__file__).parent / "_openai_inputs.json").read_text())


class ZenMuxOpenAICCLargeLanguageModel(OAICompatLargeLanguageModel):

    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = openai_base_url(credentials)
        credentials["mode"] = self.get_model_mode(model).value
        schema = self.get_model_schema(model, credentials)
        if schema and {ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL}.intersection(
            schema.features or []
        ):
            credentials["function_calling_type"] = "tool_call"
        credentials["extra_headers"] = {"HTTP-Referer": "https://dify.ai/", "X-Title": "Dify"}
        credentials["input_formats"] = INPUT_FORMATS.get(model, {})

    @staticmethod
    def _content_part(c, formats: dict | None = None) -> dict:
        """One Dify content item as a ZenMux Chat Completions part, in the format this model accepts."""
        if c.type == PromptMessageContentType.TEXT:
            return {"type": "text", "text": c.data}
        if c.type == PromptMessageContentType.IMAGE:
            return {"type": "image_url", "image_url": {"url": file_data_uri(c), "detail": c.detail.value}}
        kind = INPUT_KIND[c.type]
        return PART_FORMATS[(formats or {}).get(kind) or INPUT_FORMAT_CANDIDATES[kind][0]](c)

    def _convert_prompt_message_to_dict(self, message: PromptMessage, credentials: dict | None = None) -> dict:
        if isinstance(message, UserPromptMessage) and isinstance(message.content, list):
            formats = (credentials or {}).get("input_formats")
            message_dict = {"role": "user", "content": [self._content_part(c, formats) for c in message.content]}
            if message.name:
                message_dict["name"] = message.name
            return message_dict
        return super()._convert_prompt_message_to_dict(message, credentials)

    @staticmethod
    def _set_reasoning_params(model_parameters: dict):
        reasoning = {}
        enable = model_parameters.pop("enable_thinking", None)
        if isinstance(enable, bool):
            reasoning["enabled"] = enable
        elif isinstance(enable, str):
            reasoning["enabled"] = True

        budget = model_parameters.pop("reasoning_budget", None)
        if isinstance(budget, int):
            reasoning["max_tokens"] = budget

        effort = model_parameters.pop("reasoning_effort", None)
        if effort in REASONING_EFFORTS:
            reasoning["effort"] = effort

        if reasoning:
            model_parameters["reasoning"] = reasoning

    @staticmethod
    def _set_json_schema_params(model_parameters: dict):
        if model_parameters.get("response_format") != "json_schema":
            return
        raw = model_parameters.get("json_schema")
        if not raw:
            return
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        model_parameters["json_schema"] = json.dumps({
            "name": "output", "schema": parsed.get("schema", parsed),
        })

    def _invoke(
        self, model, credentials, prompt_messages, model_parameters,
        tools=None, stop=None, stream=True, user=None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)

        schema = self.get_model_schema(model, credentials)
        prompt_messages = keep_supported_files(prompt_messages, schema.features if schema else None)

        # ZenMux ignores `reasoning.exclude`, so hiding is done on our side.
        hide_reasoning = bool(model_parameters.pop("exclude_reasoning_tokens", False))
        self._set_reasoning_params(model_parameters)
        self._set_json_schema_params(model_parameters)

        if stream:
            model_parameters.setdefault("stream_options", {})["include_usage"] = True

        result = self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
        return strip_reasoning(result) if stream and hide_reasoning else result

    def _create_final_llm_result_chunk(self, *args, **kwargs):
        chunk = super()._create_final_llm_result_chunk(*args, **kwargs)
        raw_usage = kwargs["usage"] if "usage" in kwargs else (args[3] if len(args) > 3 else None)
        chunk.delta.usage = apply_reported_cost(chunk.delta.usage, raw_usage)
        return chunk

    def _handle_generate_response(self, model, credentials, response, prompt_messages) -> LLMResult:
        result = super()._handle_generate_response(model, credentials, response, prompt_messages)
        result.usage = apply_reported_cost(result.usage, response.json().get("usage"))
        return result

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(model, credentials)
        return super().validate_credentials(model, credentials)

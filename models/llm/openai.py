import json
import logging
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import (
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    TextPromptMessageContent,
    UserPromptMessage,
)

logger = logging.getLogger(__name__)


class ZenMuxOpenAICCLargeLanguageModel(OAICompatLargeLanguageModel):

    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = "https://zenmux.ai/api/v1"
        credentials["mode"] = self.get_model_mode(model).value
        schema = self.get_model_schema(model, credentials)
        if schema and {ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL}.intersection(
            schema.features or []
        ):
            credentials["function_calling_type"] = "tool_call"
        credentials["extra_headers"] = {"HTTP-Referer": "https://dify.ai/", "X-Title": "Dify"}

    @staticmethod
    def _convert_files_to_text(messages: list[PromptMessage]) -> list[PromptMessage]:
        result = []
        for msg in messages:
            if not (isinstance(msg, UserPromptMessage) and isinstance(msg.content, list)):
                result.append(msg)
                continue
            parts = []
            for c in msg.content:
                if isinstance(c, TextPromptMessageContent):
                    parts.append(c.data)
                elif isinstance(c, ImagePromptMessageContent):
                    parts.append(f"[Image: {c.url}]" if c.url else "[Image]")
                elif c.type == PromptMessageContentType.DOCUMENT:
                    parts.append(f"[File: {getattr(c, 'url', '')}]")
            result.append(UserPromptMessage(content=" ".join(parts)))
        return result

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
        if effort in ("high", "medium", "low", "minimal", "none"):
            reasoning["effort"] = effort

        exclude = model_parameters.pop("exclude_reasoning_tokens", None)
        if isinstance(exclude, bool):
            reasoning["exclude"] = exclude

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
        if not (schema and ModelFeature.VISION in (schema.features or [])):
            prompt_messages = self._convert_files_to_text(prompt_messages)

        self._set_reasoning_params(model_parameters)
        self._set_json_schema_params(model_parameters)

        if stream:
            model_parameters.setdefault("stream_options", {})["include_usage"] = True

        return self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

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

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        if "reasoning" in delta and "reasoning_content" not in delta:
            delta["reasoning_content"] = delta.pop("reasoning")
        return super()._wrap_thinking_by_reasoning_content(delta, is_reasoning)


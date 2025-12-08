import logging
from collections.abc import Generator
from typing import Optional, Union

from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    ModelFeature,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
)
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
    UserPromptMessage,
    TextPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessageContentType,
    AssistantPromptMessage,
    PromptMessageContent,
)

logger = logging.getLogger(__name__)


class ZenmuxLargeLanguageModel(OAICompatLargeLanguageModel):
    """
    Model class for zenmux large language model.
    """

    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = "https://zenmux.ai/api/v1"
        credentials["mode"] = self.get_model_mode(model).value
        schema = self.get_model_schema(model, credentials)
        if schema and {ModelFeature.TOOL_CALL, ModelFeature.MULTI_TOOL_CALL}.intersection(
            schema.features or []
        ):
            credentials["function_calling_type"] = "tool_call"

        # Add OpenRouter specific headers for rankings on openrouter.ai
        credentials["extra_headers"] = {"HTTP-Referer": "https://dify.ai/", "X-Title": "Dify"}

    def _convert_files_to_text(self, messages: list[PromptMessage]) -> list[PromptMessage]:
        """
        Convert any file content in messages to text descriptions to avoid validation issues
        """
        converted_messages = []

        for message in messages:
            if isinstance(message, UserPromptMessage) and isinstance(message.content, list):
                # Process multimodal content
                text_parts = []
                for content in message.content:
                    if isinstance(content, TextPromptMessageContent):
                        text_parts.append(content.data)
                    elif isinstance(content, ImagePromptMessageContent):
                        # Convert image to text description
                        if hasattr(content, "url") and content.url:
                            text_parts.append(f"[Image file uploaded]: {content.url}")
                        else:
                            text_parts.append("[Image file uploaded]")
                    elif (
                        hasattr(content, "type")
                        and content.type == PromptMessageContentType.DOCUMENT
                    ):
                        # Handle any other content types
                        if hasattr(content, "url"):
                            text_parts.append(f"[File uploaded]: {content.url}")
                        else:
                            text_parts.append(str(content))

                # Create new text-only message
                converted_message = UserPromptMessage(content=" ".join(text_parts))
                converted_messages.append(converted_message)
            else:
                # Keep non-multimodal messages as is
                converted_messages.append(message)

        return converted_messages

    @staticmethod
    def _set_reasoning_params(model_parameters: dict):
        reasoning_params = {}

        reasoning_budget = model_parameters.pop("reasnonig_budget", None)
        enable_thinking = model_parameters.pop("enable_thinking", None)
        reasoning_effort = model_parameters.pop("reasoning_effort", None)
        exclude_reasoning_tokens = model_parameters.pop("exclude_reasoning_tokens", None)

        if isinstance(enable_thinking, bool):
            reasoning_params["enabled"] = enable_thinking
        elif isinstance(enable_thinking, str):
            reasoning_params["enabled"] = True

        if isinstance(exclude_reasoning_tokens, bool):
            reasoning_params["exclude"] = exclude_reasoning_tokens

        if isinstance(reasoning_budget, int):
            reasoning_params["max_tokens"] = reasoning_budget

        if reasoning_effort in ["high", "medium", "low", "minimal", "none"]:
            reasoning_params["effort"] = reasoning_effort

        if reasoning_params:
            model_parameters["reasoning"] = reasoning_params

    @staticmethod
    def _set_json_schema_params(model_parameters: dict):
        response_format = model_parameters.get("response_format")
        if response_format and response_format == "json_schema":
            json_schema_str = model_parameters.get("json_schema")
            if json_schema_str:
                json_schema = json.loads(json_schema_str)
                schema = json_schema.get("schema") if "schema" in json_schema else json_schema
                model_parameters["json_schema"] = json.dumps({"name": "output", "schema": schema})

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        self._update_credential(model, credentials)

        # Only convert file content to text descriptions for models that don't support vision
        model_schema = self.get_model_schema(model, credentials)
        if not (model_schema and ModelFeature.VISION in (model_schema.features or [])):
            prompt_messages = self._convert_files_to_text(prompt_messages)

        self._set_reasoning_params(model_parameters)
        self._set_json_schema_params(model_parameters)

        if stream:
            stream_options = model_parameters.setdefault("stream_options", {})
            stream_options["include_usage"] = True

        return self._generate(
            model, credentials, prompt_messages, model_parameters, tools, stop, stream, user
        )

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

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        return super().get_customizable_model_schema(model, credentials)

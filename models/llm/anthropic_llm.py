import base64
import json
import logging
import re
from collections.abc import Generator
from typing import Optional, Union

import requests
import anthropic as anthropic_sdk
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeAuthorizationError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

logger = logging.getLogger(__name__)


class ZenMuxAnthropicLargeLanguageModel(LargeLanguageModel):
    """Claude via ZenMux Anthropic native endpoint."""

    @property
    def _invoke_error_mapping(self) -> dict:
        return {
            InvokeConnectionError: [
                anthropic_sdk.APIConnectionError,
                anthropic_sdk.APITimeoutError,
            ],
            InvokeServerUnavailableError: [anthropic_sdk.InternalServerError],
            InvokeRateLimitError: [anthropic_sdk.RateLimitError],
            InvokeAuthorizationError: [
                anthropic_sdk.AuthenticationError,
                anthropic_sdk.PermissionDeniedError,
            ],
            InvokeBadRequestError: [
                anthropic_sdk.BadRequestError,
                anthropic_sdk.NotFoundError,
            ],
        }

    def _get_client(self, credentials: dict) -> anthropic_sdk.Anthropic:
        return anthropic_sdk.Anthropic(
            api_key=credentials["api_key"],
            base_url="https://zenmux.ai/api/anthropic",
        )

    @staticmethod
    def _fetch_url_image(url: str) -> tuple[str, str]:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        media_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        return base64.standard_b64encode(resp.content).decode(), media_type

    def _build_content_blocks(self, content) -> list[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        blocks = []
        for item in content:
            if item.type == PromptMessageContentType.TEXT:
                blocks.append({"type": "text", "text": item.data})
            elif item.type == PromptMessageContentType.IMAGE:
                if item.base64_data:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mime_type or "image/jpeg",
                            "data": item.base64_data,
                        },
                    })
                elif item.url:
                    try:
                        b64, mime = self._fetch_url_image(item.url)
                        blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": b64},
                        })
                    except Exception as exc:
                        logger.warning(f"Image fetch failed ({item.url}): {exc}")
                        blocks.append({"type": "text", "text": f"[Image unavailable: {item.url}]"})
        return blocks

    def _convert_messages(
        self, prompt_messages: list[PromptMessage]
    ) -> tuple[str | None, list[dict]]:
        system_text: str | None = None
        messages: list[dict] = []

        for msg in prompt_messages:
            if isinstance(msg, SystemPromptMessage):
                if isinstance(msg.content, str):
                    system_text = msg.content
                else:
                    system_text = "\n".join(
                        c.data for c in msg.content if c.type == PromptMessageContentType.TEXT
                    )

            elif isinstance(msg, UserPromptMessage):
                messages.append({
                    "role": "user",
                    "content": self._build_content_blocks(msg.content),
                })

            elif isinstance(msg, AssistantPromptMessage):
                blocks: list[dict] = []
                raw = msg.content or ""
                if isinstance(raw, list):
                    raw = "".join(c.data for c in raw if c.type == PromptMessageContentType.TEXT)
                text = re.sub(r"<think>.*?</think>\s*", "", str(raw), flags=re.DOTALL).strip()
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in msg.tool_calls or []:
                    args = tc.function.arguments
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(args) if isinstance(args, str) else args,
                    })
                messages.append({"role": "assistant", "content": blocks or ""})

            elif isinstance(msg, ToolPromptMessage):
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                    and messages[-1]["content"]
                    and messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append({"role": "user", "content": [result_block]})

        return system_text, messages

    @staticmethod
    def _convert_tools(tools: list[PromptMessageTool]) -> list[dict]:
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.parameters}
            for t in tools
        ]

    def _build_request_params(
        self,
        model: str,
        model_parameters: dict,
        system: str | None,
        messages: list[dict],
        tools: list[dict] | None,
        stop: list[str] | None,
    ) -> tuple[dict, dict]:
        """Returns (sdk_params, extra_body) where extra_body contains ZenMux extensions."""
        params: dict = {
            "model": model,
            "max_tokens": model_parameters.get("max_tokens", 1024),
            "messages": messages,
        }
        extra_body: dict = {}

        if system:
            params["system"] = system
        if stop:
            params["stop_sequences"] = stop

        enable_thinking = bool(model_parameters.get("enable_thinking"))
        if enable_thinking:
            budget = int(model_parameters.get("reasoning_budget", 2000))
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            temp = model_parameters.get("temperature")
            if temp is not None:
                params["temperature"] = float(temp)
            top_p = model_parameters.get("top_p")
            if top_p is not None:
                params["top_p"] = float(top_p)
            top_k = model_parameters.get("top_k")
            if top_k is not None:
                params["top_k"] = int(top_k)

        if tools:
            params["tools"] = tools

        # ZenMux extension: output_config.format for structured output
        fmt = model_parameters.get("response_format")
        if fmt == "json_schema":
            raw = model_parameters.get("json_schema")
            if raw:
                schema = json.loads(raw) if isinstance(raw, str) else raw
                if "schema" in schema:
                    schema = schema["schema"]
                extra_body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        return params, extra_body

    def _parse_non_stream(
        self,
        model: str,
        credentials: dict,
        response: anthropic_sdk.types.Message,
        prompt_messages: list[PromptMessage],
        exclude_thinking: bool,
    ) -> LLMResult:
        text_parts: list[str] = []
        tool_calls: list[AssistantPromptMessage.ToolCall] = []

        for block in response.content:
            if block.type == "thinking" and not exclude_thinking:
                text_parts.insert(0, f"<think>\n{block.thinking}\n</think>")
            elif block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(AssistantPromptMessage.ToolCall(
                    id=block.id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                ))

        content = "\n".join(text_parts)
        message = AssistantPromptMessage(content=content, tool_calls=tool_calls)
        usage = self._calc_response_usage(
            model=model,
            credentials=credentials,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
        return LLMResult(model=model, prompt_messages=prompt_messages, message=message, usage=usage)

    def _stream_generate(
        self,
        model: str,
        credentials: dict,
        client: anthropic_sdk.Anthropic,
        params: dict,
        extra_body: dict,
        prompt_messages: list[PromptMessage],
        exclude_thinking: bool,
    ) -> Generator[LLMResultChunk, None, None]:
        index = 0
        blocks: dict[int, dict] = {}
        thinking_open = False
        input_tokens = 0
        output_tokens = 0

        with client.messages.stream(**params, extra_body=extra_body or None) as stream:
            for event in stream:
                etype = event.type

                if etype == "message_start":
                    input_tokens = event.message.usage.input_tokens

                elif etype == "content_block_start":
                    b = event.content_block
                    btype = b.type
                    bdata: dict = {"type": btype, "buf": ""}
                    if btype == "tool_use":
                        bdata["id"] = b.id
                        bdata["name"] = b.name
                    blocks[event.index] = bdata

                    if btype == "thinking":
                        thinking_open = True
                        if not exclude_thinking:
                            yield LLMResultChunk(
                                model=model,
                                prompt_messages=list(prompt_messages),
                                delta=LLMResultChunkDelta(
                                    index=index,
                                    message=AssistantPromptMessage(content="<think>\n"),
                                ),
                            )
                            index += 1

                elif etype == "content_block_delta":
                    b = blocks.get(event.index, {})
                    delta = event.delta

                    if delta.type == "text_delta":
                        b["buf"] = b.get("buf", "") + delta.text
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=list(prompt_messages),
                            delta=LLMResultChunkDelta(
                                index=index,
                                message=AssistantPromptMessage(content=delta.text),
                            ),
                        )
                        index += 1

                    elif delta.type == "thinking_delta":
                        b["buf"] = b.get("buf", "") + delta.thinking
                        if not exclude_thinking:
                            yield LLMResultChunk(
                                model=model,
                                prompt_messages=list(prompt_messages),
                                delta=LLMResultChunkDelta(
                                    index=index,
                                    message=AssistantPromptMessage(content=delta.thinking),
                                ),
                            )
                            index += 1

                    elif delta.type == "input_json_delta":
                        b["buf"] = b.get("buf", "") + delta.partial_json

                elif etype == "content_block_stop":
                    b = blocks.get(event.index, {})

                    if b.get("type") == "thinking" and thinking_open:
                        thinking_open = False
                        if not exclude_thinking:
                            yield LLMResultChunk(
                                model=model,
                                prompt_messages=list(prompt_messages),
                                delta=LLMResultChunkDelta(
                                    index=index,
                                    message=AssistantPromptMessage(content="\n</think>"),
                                ),
                            )
                            index += 1

                    elif b.get("type") == "tool_use":
                        tc = AssistantPromptMessage.ToolCall(
                            id=b["id"],
                            type="function",
                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                name=b["name"],
                                arguments=b["buf"],
                            ),
                        )
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=list(prompt_messages),
                            delta=LLMResultChunkDelta(
                                index=index,
                                message=AssistantPromptMessage(content="", tool_calls=[tc]),
                            ),
                        )
                        index += 1

                elif etype == "message_delta":
                    output_tokens = event.usage.output_tokens

                elif etype == "message_stop":
                    usage = self._calc_response_usage(
                        model=model,
                        credentials=credentials,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=list(prompt_messages),
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=AssistantPromptMessage(content=""),
                            finish_reason="stop",
                            usage=usage,
                        ),
                    )

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
        client = self._get_client(credentials)
        system, messages = self._convert_messages(prompt_messages)
        anthropic_tools = self._convert_tools(tools) if tools else None
        exclude_thinking = bool(model_parameters.get("exclude_reasoning_tokens", True))

        params, extra_body = self._build_request_params(
            model=model,
            model_parameters=model_parameters,
            system=system,
            messages=messages,
            tools=anthropic_tools,
            stop=stop,
        )

        if stream:
            return self._stream_generate(
                model, credentials, client, params, extra_body, prompt_messages, exclude_thinking
            )

        response = client.messages.create(**params, extra_body=extra_body or None)
        return self._parse_non_stream(model, credentials, response, prompt_messages, exclude_thinking)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        parts = []
        for m in prompt_messages:
            if isinstance(m.content, str):
                parts.append(m.content)
            elif isinstance(m.content, list):
                parts.extend(c.data for c in m.content if c.type == PromptMessageContentType.TEXT)
        return self._get_num_tokens_by_gpt2(" ".join(parts))

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            client = self._get_client(credentials)
            client.messages.create(
                model=model,
                max_tokens=5,
                messages=[{"role": "user", "content": "hi"}],
            )
        except anthropic_sdk.AuthenticationError as exc:
            raise CredentialsValidateFailedError(str(exc))
        except Exception as exc:
            raise CredentialsValidateFailedError(str(exc))

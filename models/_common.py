import base64
from collections.abc import Generator, Mapping
from decimal import Decimal, InvalidOperation

import requests

from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMResultChunk, LLMUsage
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageContentType,
    TextPromptMessageContent,
    UserPromptMessage,
)
from dify_plugin.entities.model.text_embedding import EmbeddingUsage

# ZenMux serves the same API from two regions; `region` comes from the provider credentials.
REGION_HOSTS = {
    "global": "https://zenmux.ai",
    "cn": "https://zenmux.dev",
}
PRICE_UNIT = Decimal("0.000001")


def zenmux_host(credentials: Mapping | None) -> str:
    region = (credentials or {}).get("region") or "global"
    return REGION_HOSTS.get(region, REGION_HOSTS["global"])


def openai_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/v1"


def anthropic_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/anthropic"


def vertex_base_url(credentials: Mapping | None) -> str:
    return f"{zenmux_host(credentials)}/api/vertex-ai"


_IMAGE_SIGNATURES = ((b"\x89PNG", "image/png"), (b"\xff\xd8\xff", "image/jpeg"), (b"GIF8", "image/gif"), (b"BM", "image/bmp"))


def image_data_uri(content: str) -> str:
    """Dify hands knowledge-base images over as bare base64; ZenMux wants a data URI (or a URL)."""
    if content.startswith(("data:", "http://", "https://")):
        return content
    head = base64.b64decode(content[:24])
    mime = "image/webp" if head[:4] == b"RIFF" and head[8:12] == b"WEBP" else next(
        (m for sig, m in _IMAGE_SIGNATURES if head.startswith(sig)), "image/png")
    return f"data:{mime};base64,{content}"


def _decimal(value) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _unit_price(amount: Decimal, tokens: int) -> Decimal:
    return (amount / (tokens * PRICE_UNIT)).quantize(Decimal("0.000001")) if tokens else Decimal(0)


def apply_reported_cost(usage, raw_usage: Mapping | None):
    """Replace the YAML-price estimate with the cost ZenMux reports for the request.

    ZenMux returns `usage.cost` (USD) and `usage.cost_details` synchronously on the
    OpenAI-compatible and Anthropic endpoints. It equals the bill's `originAmount`:
    tiered pricing, cache reads/writes and web search are applied, promotional
    discounts are not (verified 2026-10-07 against /management/generation). Dify's flat
    input/output price cannot express tiers or caching. Without it we keep the estimate.
    """
    cost = _decimal((raw_usage or {}).get("cost"))
    if usage is None or cost is None:
        return usage
    if isinstance(usage, EmbeddingUsage):
        return usage.model_copy(update={
            "total_price": cost, "unit_price": _unit_price(cost, usage.tokens),
            "price_unit": PRICE_UNIT, "currency": "USD",
        })
    if not isinstance(usage, LLMUsage):
        return usage
    completion = min(_decimal(((raw_usage or {}).get("cost_details") or {}).get("completion")) or Decimal(0), cost)
    prompt = cost - completion
    return usage.model_copy(update={
        "prompt_price": prompt, "prompt_unit_price": _unit_price(prompt, usage.prompt_tokens),
        "prompt_price_unit": PRICE_UNIT,
        "completion_price": completion, "completion_unit_price": _unit_price(completion, usage.completion_tokens),
        "completion_price_unit": PRICE_UNIT,
        "total_price": cost, "currency": "USD",
    })


def file_base64(c) -> str:
    """The file's bytes as base64, fetched here if Dify sent only a URL.

    With MULTIMODAL_SEND_FORMAT=url Dify passes its own signed file URL, which upstreams often cannot
    reach (self-hosted, internal) or refuse ("Invalid file data"), so the plugin always inlines the file.
    """
    if c.base64_data:
        return c.base64_data
    response = requests.get(c.url, timeout=(10, 120))
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        # The signed URL stays out of the message; it would end up in Dify's logs.
        raise ValueError(f"Could not download {c.filename or c.type.value} for the model: HTTP {response.status_code}") from e
    return base64.b64encode(response.content).decode()


def file_data_uri(c) -> str:
    return f"data:{c.mime_type};base64,{file_base64(c)}"


FILE_FEATURES = {
    PromptMessageContentType.IMAGE: (ModelFeature.VISION, "Image"),
    PromptMessageContentType.DOCUMENT: (ModelFeature.DOCUMENT, "File"),
    PromptMessageContentType.AUDIO: (ModelFeature.AUDIO, "Audio"),
    PromptMessageContentType.VIDEO: (ModelFeature.VIDEO, "Video"),
}


def keep_supported_files(messages: list[PromptMessage], features) -> list[PromptMessage]:
    """Replace files of a type the model does not declare with a short text note.

    Every route sends what the model declares and nothing is dropped silently: an undeclared
    file becomes e.g. "[Audio: memo.mp3]". Messages left with only text collapse to a string.
    """
    features = {getattr(f, "value", f) for f in features or []}
    result = []
    for msg in messages:
        if not (isinstance(msg, UserPromptMessage) and isinstance(msg.content, list)):
            result.append(msg)
            continue
        parts = []
        for c in msg.content:
            feature, label = FILE_FEATURES.get(c.type, (None, None))
            if feature is None or feature.value in features:
                parts.append(c)
            else:
                name = c.filename or c.url
                parts.append(TextPromptMessageContent(data=f"[{label}: {name}]" if name else f"[{label}]"))
        if all(isinstance(p, TextPromptMessageContent) for p in parts):
            parts = " ".join(p.data for p in parts)
        result.append(msg.model_copy(update={"content": parts}))
    return result


def strip_reasoning(chunks: Generator[LLMResultChunk, None, None]) -> Generator[LLMResultChunk, None, None]:
    """Drop `<think>...</think>` segments from a stream of chunks.

    ZenMux ignores `reasoning.exclude`, so "hide the thought process" is enforced here.
    The SDK always emits the tags whole, so per-chunk scanning with carried state is enough.
    """
    inside = False
    for chunk in chunks:
        content = chunk.delta.message.content
        if not isinstance(content, str) or not content:
            if not inside or chunk.delta.message.tool_calls or chunk.delta.usage:
                yield chunk
            continue
        kept, rest = [], content
        while rest:
            if inside:
                end = rest.find("</think>")
                if end < 0:
                    rest = ""
                else:
                    rest, inside = rest[end + len("</think>"):].lstrip("\n"), False
            else:
                start = rest.find("<think>")
                if start < 0:
                    kept.append(rest)
                    rest = ""
                else:
                    kept.append(rest[:start])
                    rest, inside = rest[start + len("<think>"):], True
        text = "".join(kept)
        if text or chunk.delta.message.tool_calls or chunk.delta.usage or chunk.delta.finish_reason:
            chunk.delta.message.content = text
            yield chunk

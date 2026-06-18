"""Shared helpers for the ZenMux image tools (text2image / image_edit).

ZenMux exposes image generation through the Vertex AI protocol (google-genai),
the same client construction used by ``models/llm/google.py``:

* Google image models (``google/*``, e.g. Nano Banana Pro) use the
  ``generate_content`` API with ``response_modalities=["TEXT", "IMAGE"]``; the
  result is read from ``response`` parts (``part.text`` / ``part.inline_data``).
* Every other model (``openai/*``, ``qwen/*``) uses the ``generate_images`` /
  ``edit_image`` APIs. ZenMux internally translates these Vertex AI calls into
  the OpenAI image-generation format, so OpenAI-specific options (``imageSize``,
  ``quality``) are passed through ``config.http_options.extra_body``.

See https://zenmux.ai/docs/guide/advanced/image-generation.html
"""

from google import genai
from google.genai import types

ZENMUX_VERTEX_BASE_URL = "https://zenmux.ai/api/vertex-ai"

# Default / flagship model: Nano Banana Pro.
DEFAULT_IMAGE_MODEL = "google/gemini-3-pro-image-preview"

# All image models reachable through the Vertex AI protocol. Keep in sync with
# the ``model`` select options in text2image.yaml / image_edit.yaml.
IMAGE_MODELS = (
    "google/gemini-3-pro-image-preview",
    "openai/gpt-image-2",
    "openai/gpt-image-1.5",
    "qwen/qwen-image-2.0",
)

_OUTPUT_FORMAT_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


def build_client(credentials: dict) -> genai.Client:
    """Construct a google-genai client pointed at ZenMux's Vertex AI endpoint."""
    api_key = credentials.get("api_key")
    if not api_key:
        raise ValueError("ZenMux API key is required")
    return genai.Client(
        api_key=api_key,
        vertexai=True,
        http_options=types.HttpOptions(
            api_version="v1", base_url=ZENMUX_VERTEX_BASE_URL
        ),
    )


def is_google_model(model: str) -> bool:
    """Google image models use generate_content; everything else uses generate_images."""
    return model.startswith("google/")


def output_mime_type(output_format: str | None) -> str | None:
    """Translate an ``output_format`` parameter (png/jpeg/webp) into a MIME type."""
    if not output_format:
        return None
    return _OUTPUT_FORMAT_MIME.get(output_format.lower())


def images_from_generate_content(
    response: types.GenerateContentResponse,
) -> tuple[list[tuple[str, bytes]], str]:
    """Collect ``(mime_type, bytes)`` images and any text from a generate_content response.

    The genai SDK returns ``inline_data.data`` as raw bytes, ready to be handed
    straight to ``Tool.create_blob_message``.
    """
    images: list[tuple[str, bytes]] = []
    texts: list[str] = []
    for candidate in response.candidates or []:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            if part.text:
                texts.append(part.text)
            inline = part.inline_data
            if inline and inline.data and (inline.mime_type or "").startswith("image/"):
                images.append((inline.mime_type, inline.data))
    return images, "".join(texts)


def images_from_generate_images(response) -> list[tuple[str, bytes]]:
    """Collect ``(mime_type, bytes)`` images from a generate_images / edit_image response."""
    images: list[tuple[str, bytes]] = []
    for generated in response.generated_images or []:
        image = generated.image
        if image and image.image_bytes:
            images.append((image.mime_type or "image/png", image.image_bytes))
    return images

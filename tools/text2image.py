from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from google.genai import types

from ._common import (
    DEFAULT_IMAGE_MODEL,
    build_client,
    images_from_generate_content,
    images_from_generate_images,
    is_google_model,
    output_mime_type,
)


class ZenMuxText2ImageTool(Tool):
    """Generate images from a text prompt via ZenMux image models."""

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        prompt = (tool_parameters.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")
        model = tool_parameters.get("model") or DEFAULT_IMAGE_MODEL

        client = build_client(self.runtime.credentials)

        if is_google_model(model):
            yield from self._generate_with_gemini(client, model, prompt, tool_parameters)
        else:
            yield from self._generate_with_images_api(
                client, model, prompt, tool_parameters
            )

    def _generate_with_gemini(
        self, client, model: str, prompt: str, params: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        config = types.GenerateContentConfig(
            response_modalities=[types.Modality.TEXT.value, types.Modality.IMAGE.value]
        )
        if aspect_ratio := params.get("aspect_ratio"):
            config.image_config = types.ImageConfig(aspect_ratio=aspect_ratio)

        response = client.models.generate_content(
            model=model, contents=[prompt], config=config
        )
        images, text = images_from_generate_content(response)
        if text:
            yield self.create_text_message(text)
        if not images:
            raise ValueError("ZenMux returned no image for this prompt")
        for mime_type, data in images:
            yield self.create_blob_message(blob=data, meta={"mime_type": mime_type})

    def _generate_with_images_api(
        self, client, model: str, prompt: str, params: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        config = types.GenerateImagesConfig()
        if (n := params.get("n")) is not None:
            config.number_of_images = int(n)
        if mime := output_mime_type(params.get("output_format")):
            config.output_mime_type = mime

        # OpenAI-specific options are passed through extra_body (ZenMux passthrough).
        extra_body: dict = {}
        if size := params.get("size"):
            extra_body["imageSize"] = size
        if quality := params.get("quality"):
            extra_body["quality"] = quality
        if extra_body:
            config.http_options = types.HttpOptions(extra_body=extra_body)

        response = client.models.generate_images(
            model=model, prompt=prompt, config=config
        )
        images = images_from_generate_images(response)
        if not images:
            raise ValueError("ZenMux returned no image for this prompt")
        for mime_type, data in images:
            yield self.create_blob_message(blob=data, meta={"mime_type": mime_type})

from collections.abc import Generator

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File
from google.genai import types

from ._common import (
    DEFAULT_IMAGE_MODEL,
    build_client,
    images_from_generate_content,
    images_from_generate_images,
    is_google_model,
    output_mime_type,
)


class ZenMuxImageEditTool(Tool):
    """Edit / transform existing images with a text prompt (image-to-image)."""

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        prompt = (tool_parameters.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("`prompt` is required")

        images = self._as_file_list(tool_parameters.get("image"))
        if not images:
            raise ValueError("at least one input `image` is required")
        mask = tool_parameters.get("mask")

        model = tool_parameters.get("model") or DEFAULT_IMAGE_MODEL
        client = build_client(self.runtime.credentials)

        if is_google_model(model):
            yield from self._edit_with_gemini(client, model, prompt, images)
        else:
            yield from self._edit_with_images_api(
                client, model, prompt, images, mask, tool_parameters
            )

    @staticmethod
    def _as_file_list(value) -> list[File]:
        """Normalize a ``files`` (list) or ``file`` (single) parameter into a list."""
        if value is None:
            return []
        if isinstance(value, list):
            return [f for f in value if f is not None]
        return [value]

    @staticmethod
    def _mime_of(file: File) -> str:
        return file.mime_type or "image/png"

    def _edit_with_gemini(
        self, client, model: str, prompt: str, images: list[File]
    ) -> Generator[ToolInvokeMessage, None, None]:
        # Google models take the reference image(s) inline in `contents`.
        contents: list = [
            types.Part.from_bytes(data=f.blob, mime_type=self._mime_of(f))
            for f in images
        ]
        contents.append(prompt)
        config = types.GenerateContentConfig(
            response_modalities=[types.Modality.TEXT.value, types.Modality.IMAGE.value]
        )
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
        result_images, text = images_from_generate_content(response)
        if text:
            yield self.create_text_message(text)
        if not result_images:
            raise ValueError("ZenMux returned no image for this edit")
        for mime_type, data in result_images:
            yield self.create_blob_message(blob=data, meta={"mime_type": mime_type})

    def _edit_with_images_api(
        self,
        client,
        model: str,
        prompt: str,
        images: list[File],
        mask: File | None,
        params: dict,
    ) -> Generator[ToolInvokeMessage, None, None]:
        reference_images: list = []
        ref_id = 0
        for f in images:
            ref_id += 1
            reference_images.append(
                types.RawReferenceImage(
                    reference_id=ref_id,
                    reference_image=types.Image(
                        image_bytes=f.blob, mime_type=self._mime_of(f)
                    ),
                )
            )
        if mask is not None:
            ref_id += 1
            reference_images.append(
                types.MaskReferenceImage(
                    reference_id=ref_id,
                    reference_image=types.Image(
                        image_bytes=mask.blob, mime_type=self._mime_of(mask)
                    ),
                    config=types.MaskReferenceConfig(
                        mask_mode="MASK_MODE_USER_PROVIDED"
                    ),
                )
            )

        config = types.EditImageConfig()
        if (n := params.get("n")) is not None:
            config.number_of_images = int(n)
        if mime := output_mime_type(params.get("output_format")):
            config.output_mime_type = mime

        extra_body: dict = {}
        if size := params.get("size"):
            extra_body["imageSize"] = size
        if quality := params.get("quality"):
            extra_body["quality"] = quality
        if extra_body:
            config.http_options = types.HttpOptions(extra_body=extra_body)

        response = client.models.edit_image(
            model=model,
            prompt=prompt,
            reference_images=reference_images,
            config=config,
        )
        result_images = images_from_generate_images(response)
        if not result_images:
            raise ValueError("ZenMux returned no image for this edit")
        for mime_type, data in result_images:
            yield self.create_blob_message(blob=data, meta={"mime_type": mime_type})

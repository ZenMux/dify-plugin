
from decimal import Decimal

import requests
from dify_plugin import OAICompatEmbeddingModel
from dify_plugin.entities.model.text_embedding import MultiModalContentType, MultiModalEmbeddingResult

from models._common import apply_reported_cost, image_data_uri, openai_base_url


class ZenMuxTextEmbeddingModel(OAICompatEmbeddingModel):

    def _invoke(self, model, credentials, texts, user=None, input_type=None):
        credentials["endpoint_url"] = openai_base_url(credentials)
        return super()._invoke(model, credentials, texts, user, input_type)

    def _invoke_multimodal(self, model, credentials, documents, user=None, input_type=None):
        """Used by Dify's multimodal knowledge base for models whose YAML lists `vision`.

        ZenMux takes images for these models as data-URI strings in the plain `input` list
        (object inputs are rejected). The SDK text path is not reused because it truncates
        long inputs by GPT-2 token count, which would cut a data URI in half.
        """
        inputs = [d.content if d.content_type == MultiModalContentType.TEXT else image_data_uri(d.content)
                  for d in documents]
        max_chunks = self._get_max_chunks(model, credentials)
        embeddings, tokens, costs = [], 0, []
        for i in range(0, len(inputs), max_chunks):
            response = requests.post(
                f"{openai_base_url(credentials)}/embeddings",
                headers={"Authorization": f"Bearer {credentials['api_key']}"},
                json={"model": model, "input": inputs[i:i + max_chunks], "encoding_format": "float"},
                timeout=(10, 300),
            )
            response.raise_for_status()
            data = response.json()
            embeddings += [item["embedding"] for item in data["data"]]
            tokens += data["usage"]["total_tokens"]
            costs.append(data["usage"].get("cost"))
        usage = self._calc_response_usage(model=model, credentials=credentials, tokens=tokens)
        if None not in costs:
            usage = apply_reported_cost(usage, {"cost": sum(Decimal(str(c)) for c in costs)})
        return MultiModalEmbeddingResult(model=model, embeddings=embeddings, usage=usage)

    def validate_credentials(self, model, credentials):
        credentials["endpoint_url"] = openai_base_url(credentials)
        return super().validate_credentials(model, credentials)

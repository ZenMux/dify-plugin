import requests
from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelType
from dify_plugin.entities.model.rerank import MultiModalRerankResult, RerankDocument, RerankResult
from dify_plugin.entities.model.text_embedding import MultiModalContentType
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.rerank_model import RerankModel

from models._common import image_data_uri, openai_base_url


class ZenMuxRerankModel(RerankModel):
    """ZenMux /api/v1/rerank (DashScope-style `input` + `parameters` body).

    Scores are already relevance probabilities in [0, 1], so they are passed through
    unchanged; min-max normalisation would break `score_threshold` semantics.
    """

    def _invoke(self, model, credentials, query, docs, score_threshold=None, top_n=None, user=None) -> RerankResult:
        return RerankResult(model=model, docs=self._rank(model, credentials, query, docs, docs, score_threshold, top_n))

    def _invoke_multimodal(self, model, credentials, query, docs, score_threshold=None, top_n=None, user=None):
        """Used by Dify's multimodal knowledge base for models whose YAML lists `vision`."""

        def item(content):
            if content.content_type == MultiModalContentType.TEXT:
                return {"text": content.content}
            return {"image": image_data_uri(content.content)}

        ranked = self._rank(model, credentials, item(query), [item(d) for d in docs], [d.content for d in docs],
                            score_threshold, top_n)
        return MultiModalRerankResult(model=model, docs=ranked)

    def _rank(self, model, credentials, query, documents, texts, score_threshold, top_n) -> list[RerankDocument]:
        if not documents:
            return []
        parameters = {"return_documents": False}
        if top_n:
            parameters["top_n"] = top_n
        response = requests.post(
            f"{openai_base_url(credentials)}/rerank",
            headers={"Authorization": f"Bearer {credentials['api_key']}", "Content-Type": "application/json"},
            json={"model": model, "input": {"query": query, "documents": documents}, "parameters": parameters},
            timeout=(10, 120),
        )
        if response.status_code >= 400:
            error = (
                InvokeAuthorizationError if response.status_code in (401, 403)
                else InvokeRateLimitError if response.status_code == 429
                else InvokeServerUnavailableError if response.status_code >= 500
                else InvokeBadRequestError
            )
            raise error(f"ZenMux rerank HTTP {response.status_code}: {response.text[:500]}")
        results = response.json().get("results") or []
        ranked = [
            RerankDocument(index=r["index"], text=texts[r["index"]], score=float(r["relevance_score"]))
            for r in results
            if score_threshold is None or float(r["relevance_score"]) >= score_threshold
        ]
        ranked.sort(key=lambda d: d.score, reverse=True)
        return ranked

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._invoke(model, credentials, query="ping", docs=["pong"])
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        return AIModelEntity(
            model=model, label=I18nObject(en_us=model), model_type=ModelType.RERANK,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL, model_properties={},
        )

    @property
    def _invoke_error_mapping(self):
        # Errors raised by _invoke itself must be listed too, or the SDK flattens them to InvokeError.
        return {
            InvokeConnectionError: [InvokeConnectionError, requests.ConnectionError, requests.Timeout],
            InvokeServerUnavailableError: [InvokeServerUnavailableError],
            InvokeRateLimitError: [InvokeRateLimitError],
            InvokeAuthorizationError: [InvokeAuthorizationError],
            InvokeBadRequestError: [InvokeBadRequestError, KeyError, ValueError],
        }

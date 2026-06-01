
from dify_plugin import OAICompatEmbeddingModel


class ZenMuxTextEmbeddingModel(OAICompatEmbeddingModel):

    def _invoke(self, model, credentials, texts, user=None, input_type=None):
        credentials["endpoint_url"] = "https://zenmux.ai/api/v1"
        return super()._invoke(model, credentials, texts, user, input_type)

    def validate_credentials(self, model, credentials):
        credentials["endpoint_url"] = "https://zenmux.ai/api/v1"
        return super().validate_credentials(model, credentials)

from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError


class ZenMuxImageToolProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        # Lightweight check only: a ZenMux API key is a non-empty string. The key
        # is the same one used by the ZenMux model provider, and its real validity
        # is surfaced by the image API on the first tool invocation. Avoiding a
        # network round-trip here keeps credential saving fast and free.
        api_key = credentials.get("api_key")
        if not api_key or not str(api_key).strip():
            raise ToolProviderCredentialValidationError("API key is required")

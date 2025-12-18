
from .zenmux import ZenMuxLargeLanguageModel


def _register_models():
    from .zenmux import MODEL_CLASS_MAP
    from .openai import ZenMuxOpenAICCLargeLanguageModel
    from .google import ZenMuxGoogleLargeLanguageModel

    MODEL_CLASS_MAP.update({
        'anthropic/claude-3.7-sonnet': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-sonnet-4': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-opus-4': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-opus-4.1': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-sonnet-4.5': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-opus-4.5': ZenMuxOpenAICCLargeLanguageModel,
        'anthropic/claude-haiku-4.5': ZenMuxOpenAICCLargeLanguageModel,
        'google/gemini-2.5-flash-lite': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-flash': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-flash-image': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-pro': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3-pro-preview': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3-pro-image-preview': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3-flash-preview': ZenMuxGoogleLargeLanguageModel,
        'openai/gpt-4o-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4o': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1-nano': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1': ZenMuxOpenAICCLargeLanguageModel,
        'openai/o4-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-nano': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-pro': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.1': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2-pro': ZenMuxOpenAICCLargeLanguageModel,
        '*': ZenMuxOpenAICCLargeLanguageModel
    })

_register_models()

__all__ = [ZenMuxLargeLanguageModel]

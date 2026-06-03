
from .zenmux import ZenMuxLargeLanguageModel


def _register_models():
    from .zenmux import MODEL_CLASS_MAP
    from .openai import ZenMuxOpenAICCLargeLanguageModel
    from .google import ZenMuxGoogleLargeLanguageModel
    from .anthropic_llm import ZenMuxAnthropicLargeLanguageModel

    MODEL_CLASS_MAP.update({
        # Anthropic
        'anthropic/claude-opus-4.8': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-opus-4.7': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-opus-4.6': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-sonnet-4.6': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-opus-4.5': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-sonnet-4.5': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-opus-4.1': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-opus-4': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-sonnet-4': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-haiku-4.5': ZenMuxAnthropicLargeLanguageModel,
        'anthropic/claude-3.5-haiku': ZenMuxAnthropicLargeLanguageModel,

        # Google
        'google/gemini-3.5-flash': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3.1-pro-preview': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3.1-flash-lite-preview': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3.1-flash-lite': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-3-flash-preview': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-pro': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-flash': ZenMuxGoogleLargeLanguageModel,
        'google/gemini-2.5-flash-lite': ZenMuxGoogleLargeLanguageModel,
        'google/gemma-3-12b-it': ZenMuxOpenAICCLargeLanguageModel,

        # OpenAI
        'openai/gpt-5.5-pro': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.5': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.4-pro': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.4': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.4-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.4-nano': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.3-codex': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.3-chat': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2-pro': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2-codex': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2-chat': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.2': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.1-codex-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.1-codex': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.1-chat': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5.1': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-pro': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-codex': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-chat': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-nano': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-5-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/o4-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4.1-nano': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4o': ZenMuxOpenAICCLargeLanguageModel,
        'openai/gpt-4o-mini': ZenMuxOpenAICCLargeLanguageModel,
        'openai/chat-latest': ZenMuxOpenAICCLargeLanguageModel,

        # DeepSeek
        'deepseek/deepseek-v4-pro': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-v4-flash': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-v3.2-exp': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-v3.2': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-chat-v3.1': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-chat': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-r1-0528': ZenMuxOpenAICCLargeLanguageModel,
        'deepseek/deepseek-reasoner': ZenMuxOpenAICCLargeLanguageModel,

        # Z.AI
        'z-ai/glm-5.1': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-5-turbo': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-5': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-5v-turbo': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.7': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.7-flashx': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.6': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.6v': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.6v-flash': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.5': ZenMuxOpenAICCLargeLanguageModel,
        'z-ai/glm-4.5-air': ZenMuxOpenAICCLargeLanguageModel,

        # Qwen
        'qwen/qwen3.7-plus': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.7-max': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.6-max-preview': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.6-plus': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.6-flash': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.5-plus': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3.5-flash': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-coder-plus': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-coder': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-max': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-vl-plus': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-235b-a22b-thinking-2507': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-235b-a22b-2507': ZenMuxOpenAICCLargeLanguageModel,
        'qwen/qwen3-14b': ZenMuxOpenAICCLargeLanguageModel,

        # Moonshot
        'moonshotai/kimi-k2.6': ZenMuxOpenAICCLargeLanguageModel,
        'moonshotai/kimi-k2.5': ZenMuxOpenAICCLargeLanguageModel,

        # ByteDance
        'bytedance/doubao-seed-2.0-pro': ZenMuxOpenAICCLargeLanguageModel,
        'bytedance/doubao-seed-2.0-code': ZenMuxOpenAICCLargeLanguageModel,
        'bytedance/doubao-seed-2.0-mini': ZenMuxOpenAICCLargeLanguageModel,
        'bytedance/doubao-seed-2.0-lite': ZenMuxOpenAICCLargeLanguageModel,
        'bytedance/doubao-seed-code': ZenMuxOpenAICCLargeLanguageModel,
        'bytedance/doubao-seed-1.8': ZenMuxOpenAICCLargeLanguageModel,

        # Baidu
        'baidu/ernie-5.1': ZenMuxOpenAICCLargeLanguageModel,
        'baidu/ernie-x1.1-preview': ZenMuxOpenAICCLargeLanguageModel,
        'baidu/ernie-5.0-thinking-preview': ZenMuxOpenAICCLargeLanguageModel,

        # Xiaomi
        'xiaomi/mimo-v2.5-pro': ZenMuxOpenAICCLargeLanguageModel,
        'xiaomi/mimo-v2.5': ZenMuxOpenAICCLargeLanguageModel,
        'xiaomi/mimo-v2-flash': ZenMuxOpenAICCLargeLanguageModel,

        # X.AI
        'x-ai/grok-4.3': ZenMuxOpenAICCLargeLanguageModel,
        'x-ai/grok-4.2-fast': ZenMuxOpenAICCLargeLanguageModel,
        'x-ai/grok-4.2-fast-non-reasoning': ZenMuxOpenAICCLargeLanguageModel,

        # MiniMax
        'minimax/minimax-m3': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2.7-highspeed': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2.7': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2.5-lightning': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2.5': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2.1': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2': ZenMuxOpenAICCLargeLanguageModel,
        'minimax/minimax-m2-her': ZenMuxOpenAICCLargeLanguageModel,

        # Meta
        'meta/llama-3.3-70b-instruct': ZenMuxOpenAICCLargeLanguageModel,
        'meta/llama-4-scout-17b-16e-instruct': ZenMuxOpenAICCLargeLanguageModel,

        # Mistral
        'mistralai/mistral-large-2512': ZenMuxOpenAICCLargeLanguageModel,

        # InclusionAI
        'inclusionai/ring-2.6-1t': ZenMuxOpenAICCLargeLanguageModel,
        'inclusionai/ring-1t': ZenMuxOpenAICCLargeLanguageModel,
        'inclusionai/ling-2.6-1t': ZenMuxOpenAICCLargeLanguageModel,
        'inclusionai/ling-2.6-flash': ZenMuxOpenAICCLargeLanguageModel,
        'inclusionai/ling-1t': ZenMuxOpenAICCLargeLanguageModel,
        'inclusionai/llada2.1-flash': ZenMuxOpenAICCLargeLanguageModel,

        # Tencent
        'tencent/hy3-preview': ZenMuxOpenAICCLargeLanguageModel,
        'tencent/hunyuan-2.0-thinking': ZenMuxOpenAICCLargeLanguageModel,

        # StepFun
        'stepfun/step-3.7-flash': ZenMuxOpenAICCLargeLanguageModel,
        'stepfun/step-3.5-flash': ZenMuxOpenAICCLargeLanguageModel,
        'stepfun/step-3': ZenMuxOpenAICCLargeLanguageModel,

        # Sapiens AI
        'sapiens-ai/agnes-2.0-flash': ZenMuxOpenAICCLargeLanguageModel,

        # Kuaishou
        'kuaishou/kat-coder-pro-v2': ZenMuxOpenAICCLargeLanguageModel,

        '*': ZenMuxOpenAICCLargeLanguageModel
    })

_register_models()

__all__ = [ZenMuxLargeLanguageModel]

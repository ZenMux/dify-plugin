## ZenMux

**Author:** zenmux
**Version:** 0.0.9
**Type:** model

### Description

Model provider plugin for Dify to access the models served by [ZenMux](https://zenmux.ai).

- **LLM** — the ZenMux text model catalog (Anthropic, Google, OpenAI, DeepSeek, Qwen, Kimi, GLM, MiniMax, Doubao, …),
  including vision / video / audio / document input, tool calling, structured output and reasoning controls where the
  model supports them. Claude models use ZenMux's Anthropic-native endpoint and Gemini models the Vertex AI endpoint;
  everything else uses the OpenAI-compatible endpoint.
- **Text embedding** and **rerank** models for knowledge bases.
- **Region**: choose *Global (zenmux.ai)* or *Mainland China (zenmux.dev)* when configuring the API key.
- **Cost tracking**: on the OpenAI-compatible and Anthropic endpoints the cost recorded in Dify is the cost ZenMux
  reports per request (tiered prices and prompt caching included, promotional discounts not deducted); other routes
  use the listed base price.

Free and zero-priced models are not listed. Any other ZenMux model id can be added as a custom model.

### Install

Install **ZenMux** from the Dify Marketplace, or download the `.difypkg` from the
[GitHub releases](https://github.com/ZenMux/dify-plugin/releases) and use *Plugins → Install plugin → Local package file*.

Then open *Settings → Model Provider → ZenMux*, paste your ZenMux API key and pick the region.
Requires Dify 1.6.0 or later.

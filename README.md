## ZenMux

**Author:** zenmux
**Version:** 0.0.8
**Type:** model

### Description

This is a model provider plugin for Dify to access models provided by ZenMux which created by zenmux.ai team. The plugin is currently under heavy development for general public release.

### Install Instruction

Ensure you have the basic knowledgement of Dify and it's plugin system before start. If not, read [Get Start > Dify Plugin](https://docs.dify.ai/en/develop-plugin/getting-started/getting-started-dify-plugin) at first.

To install this plugin, open Plugins page of your Dify installation, click + Install plugin button and select GitHub. Type URL of the current repository `https://github.com/ZenMux/dify-plugin` in the promped form then click Next. Select version `v0.0.1` and package `zenmux-dify-plugin_0.0.1.difypkg` and click Next again. Check the plugin information in the next pannel and click Install to start the installation process. It will took minutes to to finish the task.

After successful installation, you will find ZenMux in your installed Plugins page.

### Image Tools

Besides the LLM and text-embedding models, this plugin also ships two **tools** for image generation, available under the **ZenMux Image** tool provider:

- **Text to Image** — generate images from a text prompt.
- **Image Edit** — edit / transform existing images with a prompt (image-to-image), with an optional mask.

Both tools support these ZenMux image models (via the Vertex AI protocol):

- `google/gemini-3-pro-image-preview` (Nano Banana Pro, default)
- `openai/gpt-image-2`
- `openai/gpt-image-1.5`
- `qwen/qwen-image-2.0`

Parameters: `model`, `size` (pixel size such as `1024x1024`, for OpenAI / Qwen models), `aspect_ratio` (for Gemini models, text-to-image), `quality` (`auto` / `low` / `medium` / `high`, OpenAI / Qwen), `n` (number of images, OpenAI / Qwen), and `output_format` (`png` / `jpeg` / `webp`).

> **Note:** In Dify, tool credentials are independent from model-provider credentials. You need to configure your ZenMux API Key once more for the **ZenMux Image** tool, even if you already configured it for the ZenMux model provider. Get your key from [zenmux.ai/settings/keys](https://zenmux.ai/settings/keys).

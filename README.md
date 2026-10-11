# Path tracer AI benchmark

In the fractal chats community we came up with a simple benchmark to quickly measure how well new LLMs perform on coding tasks.
It's just one prompt that is intentionally vague for additional challenge.

```
Please write a clear and concisely-commented C++ path tracer with support for direct lighting / NEE, which renders an 8-bit sRGB 512x512 resolution output.ppm file. The scene is a specularly-reflective sphere of radius 1 in a diffuse box of radius 2, viewed open from the front with a quad light source pointing down from the ceiling.
```

Feel free to open a PR with your results! (For a level playing field, make sure to have a c++ compiler available on the runner, so agents don't need to make up workarounds like producing the image with python...)

Thanks to contributors! bezo for maintenance, lyc for the prompt+evaluation, and everyone who submitted results!

## Results

Please see the details to evaluate individual runs:

| Model                 | Quant      | Harness                            | Effort  | Scene correct | Code correct | Details                              |
| --------------------- | ---------- | ---------------------------------- | ------- | ------------- | ------------ | ------------------------------------ |
| Qwen3.6-27B           | UD-Q8_K_XL | pi (no extensions)                 | default | ✅            | -            | [Details](/qwen3.6-27B-Q8/)          |
| Hy3 (295B-A21B)       | IQ1_M      | pi (no extensions)                 | low     | ❌            | -            | [Details](/Hy3/)                     |
| Claude Opus 5         | -          | Claude Code CLI                    | default | ✅            | -            | [Details](/Claude-Opus-5-CC/)        |
| Laguna S 2.1          | UD-Q4_K_XL | pi (no extensions)                 | default | ❌            | -            | [Details](/Laguna-S-2.1/)            |
| Deepseek 4 Flash 0731 | UD-IQ2_XXS | pi (no extensions)                 | high    | ✅            | -            | [Details](/Deepseek-4-Flash-0731/)   |
| Qwen3.8-27B           | UD-Q8_K_XL | pi (no extensions)                 | high    | ✅            | -            | [Details](/qwen3.8-27B-Q8/)          |
| Qwen3.8-27B           | UD-Q8_K_XL | pi (no extensions)                 | low     | ✅            | -            | [Details](/qwen3.8-27B-Q8-low/)      |
| Qwen3.8-27B           | UD-Q8_K_XL | DeepSeek Harness (default plugins) | low     | ✅            | -            | [Details](/qwen3.8-27B-Q8-low-dsh/)  |
| Claude Opus 5         | -          | Ultracode                          | default | ✅            | -            | [Details](/Claude-Opus-5-Ultracode/) |
| Qwen3.8-Flash-Next    | UD-IQ4_XS  | pi (no extensions)                 | high    | ✅            | -            | [Details](/qwen3.8-flash-next/)      |
| Claude Opus 5.5       | -          | Claude Code CLI                    | default | ✅            | -            | [Details](/Claude-Opus-5.5-CLI/)     |
| Qwen3.8-Flash-Next    | UD-Q4_K_XL | pi (no extensions)                 | max     | ✅            | -            | [Details](/qwen3.8-flash-next-gufo/) |

- For local models, default `llama.cpp` arguments and recommended params for coding tasks are used unless stated otherwise in the run's README.

## Failed attempts

Some models failed to complete the task (no files produced, blank image, diverting from task, etc.) - listed here separately:

- Deepseek-4-Flash-preview - ran out of context during initial thinking, then loop.

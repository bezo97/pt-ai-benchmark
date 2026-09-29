# Path tracer AI benchmark

In the fractal chats community we came up with a simple benchmark to quickly measure how well new LLMs perform on coding tasks.
It's just one prompt that is intentionally vague for additional challenge.

```
Please write a clear and concisely-commented C++ path tracer with support for direct lighting / NEE, which renders an 8-bit sRGB 512x512 resolution output.ppm file. The scene is a specularly-reflective sphere of radius 1 in a diffuse box of radius 2, viewed open from the front with a quad light source pointing down from the ceiling.
```

Feel free to open a PR with your results! (For a level playing field, make sure to have a c++ compiler available on the runner, so agents don't need to make up workarounds like producing the image with python...)

## Results

Please see outputs to evaluate quality.

| Nr. | Model                 | Quant      | Harness                            | Tokens       | Context utilization | Details                             | Code produced | Binaries built | Image produced | Scene correct | Output                              |
| --- | --------------------- | ---------- | ---------------------------------- | ------------ | ------------------- | ----------------------------------- | ------------- | -------------- | -------------- | ------------- | ----------------------------------- |
| 1   | Qwen3.6-27B           | UD-Q8_K_XL | pi (no extensions)                 | ↑15k ↓52k    | 26.1%/256k          | [Details](#1-qwen36-27b)            | ✅            | ✅             | ✅             | ✅            | [Output](/qwen3.6-27B-Q8/)          |
| 2   | Hy3 (295B-A21B)       | IQ1_M      | pi (no extensions)                 | -            | 64k                 | [Details](#2-Hy3)                   | ✅            | ✅             | ✅             | ❌            | [Output](/Hy3/)                     |
| 3   | Claude Opus 5         | -          | Claude Code CLI                    | -            | -                   | [Details](#3-Claude-Opus-5-CC)      | ✅            | ✅             | ✅             | ✅            | [Output](/Claude-Opus-5-CC/)        |
| 4   | Laguna S 2.1          | UD-Q4_K_XL | pi (no extensions)                 | ↑7.2k ↓18k   | 9.7%/256k           | [Details](#4-Laguna-S-21)           | ✅            | ❌             | ✅             | ❌            | [Output](/Laguna-S-2.1/)            |
| 5   | Deepseek 4 Flash 0731 | UD-IQ2_XXS | pi (no extensions)                 | ↑57k ↓56k    | 93.8%/64k           | [Details](#5-Deepseek-4-Flash-0731) | ✅            | ✅             | ✅             | ✅            | [Output](/Deepseek-4-Flash-0731/)   |
| 6   | Qwen3.8-27B (high)    | UD-Q8_K_XL | pi (no extensions)                 | ↑75k ↓149k   | 63.6%/256k          | [Details](#6-qwen38-27b-high-pi)    | ✅            | ✅             | ✅             | ✅            | [Output](/qwen3.8-27B-Q8/)          |
| 7   | Qwen3.8-27B (low)     | UD-Q8_K_XL | pi (no extensions)                 | ↑17k ↓47k    | 25.2%/256k          | [Details](#7-qwen38-27b-low-pi)     | ✅            | ✅             | ✅             | ✅            | [Output](/qwen3.8-27B-Q8-low/)      |
| 8   | Qwen3.8-27B (low)     | UD-Q8_K_XL | DeepSeek Harness (default plugins) | ↑2.8m ↓95.6k | 37.3%/256k          | [Details](#8-qwen38-27b-low-dsh)    | ✅            | ✅             | ✅             | ✅            | [Output](/qwen3.8-27B-Q8-low-dsh/)  |
| 9   | Claude Opus 5         | -          | Ultracode                          | -            | -                   | -                                   | ✅            | ✅             | ✅             | ✅            | [Output](/Claude-Opus-5-Ultracode/) |
| 10  | Qwen3.8-Flash-Next    | UD-IQ4_XS  | pi (no extensions)                 | ↑162k ↓67k   | 36.2%/256k          | [Details](#10-qwen38-flash-next)    | ✅            | ✅             | ✅             | ✅            | [Output](/qwen3.8-flash-next/)      |
| 11  | Claude Opus 5.5       | -          | Claude Code CLI                    | -            | -                   | [Details](#11-claude-opus-55-cli)  | ✅            | ✅             | ✅             | ✅            | [Output](/Claude-Opus-5.5-CLI/)     |
## Failed attempts

Some models failed to complete the task (no files produced, blank image, diverting from task, etc.) - listed here separately:

- Deepseek-4-Flash-preview - ran out of context during initial thinking, then loop.

## Details

For local models, default `llama.cpp` arguments and recommended params for coding tasks are used unless stated.

### 1 Qwen3.6-27B

```ini
temperature = 0.6
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
chat-template-kwargs = {"preserve_thinking": true, "reasoning_preserve": true}
```

Note: The agent created a by-product file named `nul`, I renamed it to `nul_` to be able to commit full output.

### 2 Hy3

```ini
cache-type-k = q4_0
cache-type-v = q4_0
ctkd = q4_0
ctvd = q4_0
ctx-size = 64000
chat-template-kwargs = {"reasoning_effort": "low"}
```

Note: Due to hardware limits I had to run this with Q1 quantization and also very limited context size, so there were multiple context compressions during the run.

### 3 Claude Opus 5 (CC)

```
Total cost: $0.69
Total duration (API): 2m 57s
Total duration (wall): 6m 8s
Total code changes: 323 lines added, 0 lines removed

claude-haiku-4-5: 601 input, 21 output, 0 cache read, 0 cache write ($0.0007)
claude-opus-5: 19 input, 13.5k output, 275.1k cache read, 21.1k cache write ($0.69)
```

Note: result was correctly produced in PPM format, only the submission was provided in png. Thanks to rychveldir for the run.

### 4 Laguna-S-2.1

Note: Failed to compile the c++ code so it rewrote the code in python to produce the image, which is blank. Technically it did not completely fail because c++ code was produced and it creatively used available tools, but the overall solution is not desirable.

### 5 Deepseek-4-Flash-0731

```ini
reasoning = on
cache-type-k = q4_0
cache-type-v = q4_0
ctx-size = 64000
temperature = 1.0
top-p = 0.95
chat-template-kwargs = {"reasoning_effort": "high"}
```

Notes:

- Due to hardware limits I had to run this with heavily quantized weights and kv-cache, and also a small context size (but no context compression was needed)
- The results were produced after `[↑38k ↓39k]` tokens, then the agent proceeded to verify the resulting image which turned into a sort of thinking loop.
- The png file is a byproduct, not sure why the agent converted the ppm

### 6 Qwen3.8-27B (high, pi)

```ini
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
chat-template-kwargs = {"preserve_thinking": true, "reasoning_preserve": true}
```

Pi harness effort: high

Note: Agent created a temporary debugger cpp script and used ffmpeg on path to convert ppm to png for visual verification.

### 7 Qwen3.8-27B (low, pi)

```ini
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
chat-template-kwargs = {"preserve_thinking": true, "reasoning_preserve": true, "reasoning_effort": "low"}
```

Pi harness effort: low

Note: Agent created a temporary debugger cpp script and used ffmpeg on path to convert ppm to png for visual verification.

### 8 Qwen3.8-27B (low, dsh)

```ini
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
presence-penalty = 0.0
chat-template-kwargs = {"preserve_thinking": true, "reasoning_preserve": true, "reasoning_effort": "low"}
```

Note: Input token count is very big, I think dsh is counting that in a different way. In the thinking traces I didn't see much difference compared to the pi run.

### 9 Claude Opus 5 (Ultracode)

Note: Arguably failed the "concisely-commented" requirement, Opus 5 feels verbose in general. Thanks to lycium for the run.

### 10 Qwen3.8-Flash-Next

```ini
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
```

Note: The agent produced working code and rendered image after just 17k generated tokens. Then proceeded to validate with vision, fix bugs, "decorate" the scene aesthetically, produce documentation etc. It even noticed that the wall's shading is off and pointed out the rendering bug. Overall very satisfied, beats the 27B model by a big margin in all aspects.

### 11 Claude Opus 5.5 (CLI)

```Total cost:            $0.3231
Total duration (API):  1m 13s
Total code changes:    224 lines added, 0 lines removed
Usage by model:
    claude-haiku-4-5:  977 input, 16 output, 0 cache read, 0 cache write ($0.0011)
     claude-opus-5-5:  610 input, 6.8k output, 214.7k cache read, 17.5k cache write ($0.3220)```

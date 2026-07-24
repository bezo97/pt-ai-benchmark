# Path tracer AI benchmark

In the fractal chats community we came up with a simple benchmark to quickly measure how well new LLMs perform on coding tasks.
It's just one prompt that is intentionally vague for additional challenge.

```
Please write a clear and concisely-commented C++ path tracer with support for direct lighting / NEE, which renders an 8-bit sRGB 512x512 resolution output.ppm file. The scene is a specularly-reflective sphere of radius 1 in a diffuse box of radius 2, viewed open from the front with a quad light source pointing down from the ceiling.
```

Feel free to open a PR with your results!

## Results

| Nr. | Model           | Quant      | Harness            | Tokens    | Context utilization | Details                  | Output                     |
| --- | --------------- | ---------- | ------------------ | --------- | ------------------- | ------------------------ | -------------------------- |
| 1   | Qwen3.6-27B     | UD-Q8_K_XL | pi (no extensions) | ↑15k ↓52k | 26.1%/256k          | [Details](#1-qwen36-27b) | [Output](/qwen3.6-27B-Q8/) |
| 2   | Hy3 (295B-A21B) | IQ1_M      | pi (no extensions) | -         | 64k                 | [Details](#2-Hy3)        | [Output](/Hy3/)            |

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

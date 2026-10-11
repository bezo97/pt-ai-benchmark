# Qwen3.8-Flash-Next · pi (no extensions)

![Qwen3.8-Flash-Next render result](preview.png)

## Run details

| Field               | Value              |
| ------------------- | ------------------ |
| Model               | Qwen3.8-Flash-Next |
| Effort              | high               |
| Quant               | UD-IQ4_XS          |
| Harness             | pi (no extensions) |
| Tokens              | ↑162k ↓67k         |
| Context utilization | 36.2%/256k         |
| Code produced       | ✅                 |
| Binaries built      | ✅                 |
| Image produced      | ✅                 |
| Code correct        | -                  |
| Scene correct       | ✅                 |

## Inference config

Local model served by llama.cpp with default arguments plus:

```ini
temperature = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0
```

## Notes

- The agent produced working code and rendered image after just 17k generated tokens. Then proceeded to validate with vision, fix bugs, "decorate" the scene aesthetically, produce documentation etc. It even noticed that the wall's shading is off and pointed out the rendering bug. Overall very satisfied, beats the 27B model by a big margin in all aspects.
- The agent also wrote build scripts (`build.bat` for MSVC, `Makefile` for gcc/clang) and documentation for its solution, kept verbatim as `README_.md` — note the trailing underscore, `README.md` is ours.
- `preview.png` is the agent's own png, not a conversion.

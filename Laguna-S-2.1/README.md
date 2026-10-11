# Laguna S 2.1 · pi (no extensions)

![Laguna S 2.1 render result](preview.png)

## Run details

| Field               | Value              |
| ------------------- | ------------------ |
| Model               | Laguna S 2.1       |
| Effort              | default            |
| Quant               | UD-Q4_K_XL         |
| Harness             | pi (no extensions) |
| Tokens              | ↑7.2k ↓18k         |
| Context utilization | 9.7%/256k          |
| Code produced       | ✅                 |
| Binaries built      | ❌                 |
| Image produced      | ✅ (blank)         |
| Code correct        | -                  |
| Scene correct       | ❌                 |

## Inference config

Local model served by llama.cpp with default arguments plus the recommended coding params.

## Notes

- Failed to compile the c++ code so it rewrote the code in python to produce the image, which is blank. Technically it did not completely fail because c++ code was produced and it creatively used available tools, but the overall solution is not desirable.
- `output.ppm` is in ASCII (P3) format, unlike most other runs which write binary P6.

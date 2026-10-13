# lfm2moe reference capture (llama.cpp oracle)

Golden data for the LFM2.5 (`lfm2moe`) port, captured with a stock llama.cpp
build so the qwencpp graph can be diffed against a known-good one. Same
procedure as `tests/step-ref` (on the step35 branch).

- llama.cpp `1b70350c9` (master, 2026-10-14), RTX 4060 Ti, CUDA
- model: `LiquidAI/LFM2.5-8B-A1B-GGUF` **Q4_K_M** (5.16 GB)
- prompt for the tensor/logit captures: `The capital of France is`
  -> 6 tokens `[124894, 597, 5205, 302, 3980, 355]` (llama.cpp adds BOS for
  `tokenizer.ggml.pre = lfm2` even though the file has no `add_bos_token`)
- every run uses `-ngl 99 -dev CUDA0 -c 4096 -fa off --no-warmup`

| file | what it pins down |
|---|---|
| `04-tensors.log` | layers 0 (conv + dense FFN), 2 (attention + MoE), 3 (conv + MoE) and `result_*` |
| `05-logits/` | final logits (`.bin` = raw f32 x 128000) and the prompt's token ids |
| `06-lout-cuda.log` | `l_out`, mixer outputs and `ffn_moe_out` for all 24 layers, CUDA |
| `07-lout-cpu.log` | the same capture on llama.cpp's CPU backend: the noise floor |
| `10-qw-tensors.log`, `11-qw-lout-all.log` | qwencpp's side of `04` and `06` |
| `tok-test.txt` | tokenizer edge cases (digits, CJK punctuation, CRLF, specials, emoji) |

## Comparing

```
python cmp.py 04-tensors.log 10-qw-tensors.log
python cmp_logits.py 05-logits/llamacpp-LFM2.5-8B-A1B-Q4_K_M.bin <qw-cli --dump-logits file>
```

`cmp.py` here carries an alias table: llama.cpp labels the lfm2 graph after
the HF module names (`model.layers.{}.conv.in_proj`, ...), and it hangs
`ffn_out` on the **ffn_norm** output, not the FFN output.

## What was found

- Layers 0-3 agree to 1e-4..1e-3 relative; the final logits agree on the top
  token with KL(llama||qwencpp) = 0.027 and max |dlogit| = 1.65.
- That is inside llama.cpp's own noise: its CPU and CUDA backends differ by
  KL 0.125 and max |dlogit| 2.47 on the same prompt (`06` vs `07`). Sigmoid
  routing over 32 experts flips selections on tiny differences, so the
  per-layer sums wander more than on the Qwen models; read the logits, not the
  late-layer sums (which also cancel heavily).
- Resident and offload (RAM tier) prefill are bit-identical tensor for tensor
  with `QWEN_NO_FLASH=1`. With flash attention on they differ slightly once the
  prompt is long enough for the resident graph's 32-row KV bucket and the
  offload graph's exact KV length to pick different FA kernels.
- Tokenization matches `llama-tokenize --no-escape` exactly on `tok-test.txt`
  and on 55 KB of README.ja.md + C++ source, for this model and for Qwen3 /
  Qwen3.6 (the CJK-punctuation fix applied to the qwen2 split as well).
- The chat prompt for one user turn matches llama.cpp's rendering of the
  GGUF template token for token.

## Caveats

- LFM2.5 always opens `<think>` itself. An empty `<think></think>` prefill does
  not stop it (it reopens the block or prints a stray `</think>`, in llama.cpp
  too), so `--reasoning off` is a no-op for this architecture and the server
  routes the leading block to `reasoning_content` regardless.
- Tool definitions are serialised with nlohmann's `dump()` (sorted keys, no
  spaces), as the Qwen path already does, not Jinja's `tojson`.

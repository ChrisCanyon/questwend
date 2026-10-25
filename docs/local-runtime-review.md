# Local runtime review guide

This branch preserves the Open Agents runtime source so its changes can be
reviewed in ChrisCanyon's fork. It is not a proposal to merge the complete
patch into upstream QuestWend. The existing serving binary was not replaced.

## Review in this order

1. Review the small tokenizer and buffered-stream PRs independently. They are
   the ordinary bug fixes and do not need the cache or RPC experiments.
2. Read `core/host_expert_cache.h`, `gpu_lru_index.h`, `expert_io_workers.h`,
   `expert_readiness.h`, `prefill_cache_policy.h` and their model-free tests.
   These hold the RAM tier, eviction indexing, worker ownership and prefill
   isolation policies. Read their call sites in `core/expert_cache.cpp` next.
3. Read `core/hybrid_schedule.h`, the bounded CPU-miss path in `runtime.cpp`,
   and the attribution files. These are local adaptations of published work,
   not a new implementation of the complete MoE-Infinity engine.
4. Review Q8 storage/checkpoint code separately from cache scheduling. Q8 KV
   is a representation change, not a free proof of identical output quality.
5. Review the opt-in CUDA get-rows/exact-pool paths with independent operator
   tests. The masked-zero/finite-input experiment remains outside the exact
   production claim.
6. Leave private RPC, grouped block execution, speculation and layer-window
   experiments until last. They have no demonstrated distributed speedup and
   are still mixed into eight source files. Live migration is not implemented.

## What is verified

The exact preserved source builds a CPU server in a clean checkout. Fourteen
fresh model-free cache/I/O/memory/concurrency tests passed, including 16,000
threaded host-cache operations, exclusive promotion, bounded eviction, worker
failure/cleanup, checkpoint limits and exact expert-byte audits. The existing
Flash-Next runtime has separately recorded long-context and tool integration
results. Those are different scopes of evidence.

The tokenizer also passed 26 synthetic cases across all supported pre-tokenizer
modes and 88 stock-llama.cpp parity cases across four actual vocabularies.

## Review findings and limits

- This branch is too broad for one upstream PR. It deliberately preserves the
  legacy state; it does not claim that inactive experiments have been removed
  from the executable.
- Some legacy environment flags use presence rather than parsing `0`/`1`.
  Benchmark arms must clear inherited switches before selecting a policy.
- The pinned llama.cpp metadata/parser helper is the proposed reuse boundary;
  it has not replaced this branch's custom tokenizer/renderer/parser. Malformed
  tool-output semantics still differ and must be resolved before migration.
- The available Qwen3-Next 80B GGUF uses legacy fused `ssm_in` and `ssm_ba`
  tensors. Both stock b005 and the current custom runtime fail on its first
  inference with missing `blk.0.attn_qkv.weight`. Their loaders accept metadata
  before discovering this. This is an inherited compatibility failure, not a
  custom-runtime speed result. No cross-model throughput generality is claimed.
- A model execution failure can currently end in zero output tokens after only
  logging the error. That deserves its own explicit-error regression/fix.
- Fresh CUDA/Metal/HIP/Vulkan matrix testing and full 80B inference parity are
  not established by the CPU build. Preserve rollback binaries and do not
  activate this review build as a side effect of merging a source snapshot.

## Maintenance direction

Keep a narrow native memory/SSD extension, use upstream libraries for model
formats, and move experiments behind separately buildable boundaries. A future
standalone runtime repository should carry benchmark fixtures, pinned source
and licenses rather than the cluster dashboard or private machine settings.
SSD expert streaming and activation-aware caching already exist in QuestWend
and MoE-Infinity; the engineering integration and reproducible results are the
candidate contribution. A novelty claim needs a separate literature review.

AI assistance: Codex helped implement and review these changes. This is an
owner-fork review document, not an upstream submission or maintainer response.

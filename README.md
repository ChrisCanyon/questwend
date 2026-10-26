# QuestWend production core for Open Agents

This personal fork contains the measured single-machine cache/runtime extensions
used by Open Agents. QuestWend supplies the inference engine and SSD expert
streaming; llama.cpp supplies chat templates, prompt tokenization and response
parsing. This branch is a focused production candidate, not an upstream release.

## Source boundaries

| Directory | Responsibility |
| --- | --- |
| `core/` | Model execution, GPU/RAM/SSD caches, CPU overlap, MTP, KV and prompt state |
| `server/` | HTTP/SSE transport and the upstream chat process bridge |
| `tests/` | Independent correctness fixtures and explicit diagnostic tools |
| `cmake/` | Optional tests, diagnostics and matching-backend SDK support |
| `third_party/ggml/` | Backend plus the separately measured kernel changes |

Private RPC, remote block execution, external draft experiments, QSA capture
hooks and their obsolete probes are deleted from this branch. Git history and
the previous personal review branch retain them; there is no second legacy tree.
The private RPC backend changes were reverted to the original upstream files.

The duplicated Jinja/chat renderer, prompt BPE implementation and hand-written
tool parsers are deleted. The production server requires Open Agents' pinned
`integrations/llama-chat` adapter. `TokenPieces` is only a small decoder from
existing token IDs to their original bytes; it performs no prompt encoding or
model-format parsing. UTF-8 stream buffering remains in the transport.

Prompt-state serialization is in `core/runtime_state.cpp.inc`, included after
the private Runtime implementation. Complete snapshots retain PLE convolution,
n-gram history, QSA keys and logits. Their existing payload format is preserved.
Trained MTP is retained; removing unrelated external drafts does not disable it.

## Build

The native process bridge is currently qualified on Windows. Build the pinned
helper in Open Agents first, then configure this source with its adapter path:

```powershell
python C:/workspace/open-agents/integrations/llama-chat/build.py --source path/to/llama.cpp
cmake -S . -B build -DQW_CUDA=ON -DQW_CHAT_ADAPTER_DIR=C:/workspace/open-agents/integrations/llama-chat
cmake --build build --config Release --target qw-server
```

Package `open-agents-llama-chat.exe` beside `qw-server.exe`, with the required
matching ggml/CUDA DLLs and dependency licenses. Open Agents records the source
commit, build options and binary hashes. Do not package RPC or diagnostic binaries.

`BUILD_TESTING` and `QW_BUILD_DIAGNOSTICS` default to OFF. Production builds contain
only the server and required libraries. To build and run regression fixtures:

```powershell
cmake -S . -B build-tests -DBUILD_TESTING=ON -DQW_CHAT_ADAPTER_DIR=C:/workspace/open-agents/integrations/llama-chat
cmake --build build-tests --config Release
ctest --test-dir build-tests -C Release --output-on-failure
```

`QW_BACKEND_BUILD` is an optional matching-backend SDK for host-only refactor
comparisons. It is not a substitute for checking a source build. Open Agents
validates both paths; GPU measurements retain the recorded numerical DLL family.
Diagnostics require `-DQW_BUILD_DIAGNOSTICS=ON`; they never run during a build.
The obsolete multi-platform release workflow was removed rather than publishing
an unqualified bridge on unsupported platforms.

## Serving scope

This candidate serves text and structured tools through the upstream adapter.
Its unused native CLI, vision encoder and image-only libraries were removed.
The existing `--no-mmproj` launch argument remains a no-op for profile compatibility;
image requests and `--mmproj` fail explicitly. Open Agents' consumer CLI and the
other aliases' llama.cpp vision support are separate and unchanged.

This is a refactor, not a new speed or model-quality claim. Keep the preceding
runtime available until native, inference, cache and maintained-client checks
pass. Existing background design documents describe upstream context; this README
and the actual build options define the production branch's supported surface.

## Attribution

Original engine: [kishida/questwend](https://github.com/kishida/questwend).
Chat components: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
Local policy ports retain the Caffeine, HybriMoE and MoE-Infinity references and
licenses in `third_party/`. SSD offload and activation-aware caching are established
techniques; this project claims measured implementation work, not their invention.

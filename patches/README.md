# Bundled backend changes

`third_party/ggml/GGML_VERSION` identifies the original upstream backend.
The current tree also includes the local CUDA operator changes preserved in
this personal fork; it is not a pristine upstream copy. Review them with Git
against the recorded QuestWend base rather than applying a historical patch pile.

This production-core cleanup restores the original RPC files and disables RPC
builds. Its measured GPU validation deliberately reuses the exact previous
numerical DLL family. Independent shape/stride/platform validation is still
required before proposing the local CUDA changes upstream.

Use the production source manifest and Open Agents build receipt for exact
reproduction. Never replace backend DLLs beside a running server or mix a
new server with unrecorded numerical libraries.

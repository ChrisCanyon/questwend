#pragma once

// Inference hooks for the optional expert visualizer. The normal build has
// inline no-ops, so it does not poll recorder state on every layer/token.

#include "expert_debug.h"

namespace questwend::expert_debug_hooks {

#if QW_EXPERT_VISUALIZATION
inline void initialize(int layers, int experts) { ExpertDebug::instance().initialize(layers, experts); }
inline void observe(int layer, const int32_t * ids, int count) { ExpertDebug::instance().observe(layer, ids, count); }
inline void layer_stage(int layer, const char * stage, int tokens) { ExpertDebug::instance().layer_stage(layer, stage, tokens); }
inline void scratch(int layer, const std::vector<int> & parts) { ExpertDebug::instance().scratch(layer, parts); }
inline void execution(int layer, const char * device, bool begin, const int32_t * experts, int count) {
    ExpertDebug::instance().execution(layer, device, begin, experts, count);
}
inline void maybe_capture(const ExpertCache & cache, bool force = false) {
    ExpertDebug::instance().maybe_capture(cache, force);
}
inline void set_phase(ExpertDebug::Phase phase) { ExpertDebug::instance().set_phase(phase); }
inline ExpertDebug::Mode mode() { return ExpertDebug::instance().mode(); }
#else
inline void initialize(int, int) noexcept {}
inline void observe(int, const int32_t *, int) noexcept {}
inline void layer_stage(int, const char *, int) noexcept {}
inline void scratch(int, const std::vector<int> &) noexcept {}
inline void execution(int, const char *, bool, const int32_t *, int) noexcept {}
inline void maybe_capture(const ExpertCache &, bool = false) noexcept {}
inline void set_phase(ExpertDebug::Phase) noexcept {}
inline ExpertDebug::Mode mode() noexcept { return ExpertDebug::Off; }
#endif

} // namespace questwend::expert_debug_hooks

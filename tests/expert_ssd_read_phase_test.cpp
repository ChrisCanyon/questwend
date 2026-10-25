// CPU-only policy, persistent-handle and production direct-read correctness gate.
#include "expert_ssd_read_phase.h"
#include "expert_io_workers.h"
#include "expert_ssd_windows.h"
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using questwend::ExpertSsdReadPhase;
using questwend::ExpertIOFiles;
using questwend::PersistentExpertIO;
namespace fs = std::filesystem;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void check_mode(ExpertSsdReadPhase & policy, bool configured, bool expected) {
    const auto fetch = policy.begin_fetch(configured);
    require(fetch.direct() == expected, "Incorrect effective read mode");
}

static void policy_checks() {
    for (const char * flag : {static_cast<const char *>(nullptr), "0", "1"}) {
        ExpertSsdReadPhase policy(flag);
        const bool enabled = flag && std::strcmp(flag, "1") == 0;
        // Initial/default phase preserves the configured mode.
        check_mode(policy, true, true);
        for (bool configured : {false, true}) {
            for (bool source : {true, false, true, false}) {
                policy.enter_phase(source);
                check_mode(policy, configured, configured && !(enabled && source));
            }
        }
    }
    for (const char * invalid : {"", "true", "2", "01", " 1", "1 ", "-1"}) {
        bool rejected = false;
        try { ExpertSsdReadPhase policy(invalid); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Invalid explicit phase flag accepted");
    }
    ExpertSsdReadPhase policy("1");
    policy.enter_phase(true);
    bool owner_failed = false;
    try {
        const auto fetch = policy.begin_fetch(true);
        require(!fetch.direct(), "Source snapshot is not buffered");
        bool rejected = false;
        try { policy.enter_phase(false); }
        catch (const std::logic_error &) { rejected = true; }
        require(rejected && !fetch.direct(), "In-flight phase change accepted");
        throw std::runtime_error("Intentional owner failure");
    } catch (const std::runtime_error & error) {
        owner_failed = std::strcmp(error.what(), "Intentional owner failure") == 0;
    }
    require(owner_failed, "Owner failure did not unwind the fetch guard correctly");
    policy.enter_phase(false);
    check_mode(policy, true, true);
    // A verification batch is explicitly non-source, regardless of token count.
    policy.enter_phase(false);
    check_mode(policy, true, true);
    require(policy.scope_counts().source_buffered == 1 &&
            policy.scope_counts().non_source_direct == 2,
            "Failed fetch scope or phase recovery was miscounted");
}

static std::string profile_text(const ExpertSsdReadPhase & policy, bool configured_direct) {
    FILE * file = std::tmpfile();
    require(file != nullptr, "Cannot create temporary profile fixture");
    policy.append_json(file, configured_direct);
    std::rewind(file);
    std::array<char, 2048> buffer{};
    const size_t length = std::fread(buffer.data(), 1, buffer.size(), file);
    const bool complete = !std::ferror(file) && length < buffer.size();
    std::fclose(file);
    require(complete, "Profile fixture was truncated");
    return std::string(buffer.data(), length);
}

static void profile_checks() {
    ExpertSsdReadPhase policy("1");
    policy.enter_phase(true);
    check_mode(policy, true, false);
    check_mode(policy, false, false);
    policy.enter_phase(false);
    check_mode(policy, true, true);
    check_mode(policy, false, false);
    const auto & scopes = policy.scope_counts();
    require(scopes.source_direct == 0 && scopes.source_buffered == 2 &&
            scopes.non_source_direct == 1 && scopes.non_source_buffered == 1,
            "Owner scope counters did not match effective modes");
    const auto non_source = profile_text(policy, true);
    for (const char * expected : {"\"buffered_source_option\":true", "\"configured_direct\":true",
            "\"current_source_prefill\":false", "\"effective_direct\":true",
            "\"counter_kind\":\"owner_fetch_scopes_not_actual_ssd_reads\"",
            "\"source_direct_scopes\":0", "\"source_buffered_scopes\":2",
            "\"non_source_direct_scopes\":1", "\"non_source_buffered_scopes\":1"})
        require(non_source.find(expected) != std::string::npos, "Missing/misleading profile field");
    policy.enter_phase(true);
    const auto source = profile_text(policy, true);
    require(source.find("\"current_source_prefill\":true") != std::string::npos &&
            source.find("\"effective_direct\":false") != std::string::npos,
            "Source profile mode is wrong");
    require(scopes.source_buffered == 2, "Profile export incremented a scope counter");
    ExpertSsdReadPhase disabled(nullptr);
    disabled.enter_phase(true);
    check_mode(disabled, true, true);
    require(disabled.scope_counts().source_direct == 1, "Disabled policy changed source direct scopes");
    require(profile_text(disabled, true).find("\"buffered_source_option\":false") != std::string::npos,
            "Default option reported enabled");
}

static uint8_t expected(size_t index) { return static_cast<uint8_t>((index * 17 + 73) % 251); }
static constexpr size_t file_size = 32768;
static std::atomic<int> buffered_opens{0}, direct_opens{0};
static std::atomic<int> buffered_closes{0}, direct_closes{0};

static void read_fixture(ExpertIOFiles & files, const fs::path & path,
                         bool direct, size_t offset, size_t bytes) {
    std::vector<uint8_t> output(bytes);
#ifdef _WIN32
    if (direct) {
        void *& handle = files.direct[path.string()];
        if (!handle) {
            handle = questwend::win_direct_open(path.string());
            require(handle != nullptr, "Production direct open failed");
            ++direct_opens;
        }
        require(questwend::win_direct_read(handle, output.data(), bytes, offset, files.aligned),
                "Production aligned-covering direct read failed");
    } else
#endif
    {
        FILE *& file = files.buffered[path.string()];
        if (!file) {
            file = std::fopen(path.string().c_str(), "rb");
            require(file != nullptr, "Buffered open failed");
            ++buffered_opens;
        }
        require(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0,
                "Buffered fixture seek failed");
        require(std::fread(output.data(), 1, bytes, file) == bytes, "Buffered fixture short read");
    }
    for (size_t index = 0; index < bytes; ++index)
        require(output[index] == expected(offset + index), "Fixture byte mismatch");
}

static void file_checks(const fs::path & path) {
    questwend::ExpertIOClosers closers;
    closers.buffered = [](FILE * file) { ++buffered_closes; return std::fclose(file); };
#ifdef _WIN32
    closers.direct = [](void * handle) { ++direct_closes; CloseHandle(static_cast<HANDLE>(handle)); };
#endif
    ExpertSsdReadPhase policy("1");
    {
        PersistentExpertIO pool(4, closers);
        for (bool source : {true, false, true, false}) {
            policy.enter_phase(source);
            const auto phase = policy.begin_fetch(true);
            const bool direct = phase.direct();
            // Exactly the same mode snapshot is captured by every CPU reader.
            pool.run(4, [&, direct](int worker, ExpertIOFiles & files) {
                read_fixture(files, path, direct, 4096 * worker, 4096);
                read_fixture(files, path, direct, 4093 + worker, 5007);
                read_fixture(files, path, direct, file_size - 137, 137);
            });
        }
        // Exercise the other synchronous owner-progress barrier and recovery.
        for (bool source : {false, true}) {
            policy.enter_phase(source);
            bool caught = false;
            try {
                const auto phase = policy.begin_fetch(true);
                const bool direct = phase.direct();
                pool.run_with_owner_progress(4, [&, direct](int worker, ExpertIOFiles & files) {
                    read_fixture(files, path, direct, worker * 31, 129);
                    if (worker == 1) throw std::runtime_error("Intentional reader failure");
                }, [] {});
            } catch (const std::runtime_error & error) {
                caught = std::strcmp(error.what(), "Intentional reader failure") == 0;
            }
            require(caught, "Worker failure did not propagate");
            policy.enter_phase(!source); // No reader remains active after failure.
            const auto phase = policy.begin_fetch(true);
            const bool direct = phase.direct();
            pool.run(4, [&, direct](int worker, ExpertIOFiles & files) {
                read_fixture(files, path, direct, worker * 97, 257);
            });
        }
        require(buffered_opens == 4, "Buffered handles were not worker-local/reused");
#ifdef _WIN32
        require(direct_opens == 4, "Direct handles were not worker-local/reused");
        // EOF failure uses the actual covering-range helper; reuse remains valid.
        policy.enter_phase(false);
        const auto phase = policy.begin_fetch(true);
        pool.run(4, [&](int, ExpertIOFiles & files) {
            std::array<uint8_t, 32> bytes{};
            require(!questwend::win_direct_read(files.direct.at(path.string()), bytes.data(),
                bytes.size(), file_size - 4, files.aligned), "Direct EOF overread accepted");
            read_fixture(files, path, phase.direct(), 17, 32);
        });
#endif
    }
    require(buffered_opens == buffered_closes, "Buffered handles leaked");
    require(direct_opens == direct_closes, "Direct handles leaked");
}

int main(int argc, char ** argv) {
    try {
        policy_checks();
        profile_checks();
        const fs::path root = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() /
            ("qw-ssd-phase-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(fs::create_directory(root), "Use a new fixture directory");
        const auto path = root / "weights.bin";
        {
            std::vector<uint8_t> bytes(file_size);
            for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = expected(index);
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            require(bool(file), "Fixture creation failed");
        }
        file_checks(path);
        require(fs::remove(path) && fs::remove(root), "Fixture cleanup failed (live handle?)");
        std::cout << "{\"state\":\"passed\",\"phase_policy\":true,\"persistent_workers\":4,"
                  << "\"direct_alignment_cases\":"
#ifdef _WIN32
                  << "true"
#else
                  << "false"
#endif
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "SSD read phase test failed: " << error.what() << std::endl;
        return 1;
    }
}

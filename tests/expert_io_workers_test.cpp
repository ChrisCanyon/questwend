// Synthetic correctness/lifetime test. No GPU, model or inference dependency.
#include "expert_io_workers.h"
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

using questwend::ExpertIOClosers;
using questwend::ExpertIOFiles;
using questwend::PersistentExpertIO;
namespace fs = std::filesystem;
static constexpr size_t file_size = 32768;
static std::atomic<int> closes{0};
static std::atomic<int> direct_closes{0};

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static uint8_t expected(size_t position, int seed) {
    return static_cast<uint8_t>((position * 31 + seed * 37) % 251);
}

static void fixture(const fs::path & path, int seed) {
    std::vector<uint8_t> bytes(file_size);
    for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = expected(index, seed);
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!output) throw std::runtime_error("Cannot write synthetic fixture");
}

static int close_buffered(FILE * file) {
    ++closes;
    return std::fclose(file);
}

static FILE * open_buffered(ExpertIOFiles & files, const fs::path & path, std::atomic<int> & opens) {
    FILE *& file = files.buffered[path.string()];
    if (!file) {
        file = std::fopen(path.string().c_str(), "rb");
        if (!file) throw std::runtime_error("Expected synthetic open failure");
        ++opens;
    }
    return file;
}

static void read_and_check(FILE * file, long offset, int seed) {
    std::array<uint8_t, 128> bytes{};
    require(std::fseek(file, offset, SEEK_SET) == 0, "Synthetic seek failed");
    require(std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size(), "Synthetic short read");
    for (size_t index = 0; index < bytes.size(); ++index)
        require(bytes[index] == expected(offset + index, seed), "Synthetic byte mismatch");
}

static void buffered_checks(const std::array<fs::path, 3> & paths) {
    ExpertIOClosers closers;
    closers.buffered = close_buffered;
    std::atomic<int> opens{0};
    std::array<std::thread::id, 4> worker_ids{};
    std::array<FILE *, 12> handles{};
    {
        PersistentExpertIO pool(4, closers);
        for (int batch = 0; batch < 12; ++batch) {
            pool.run(4, [&](int tid, ExpertIOFiles & files) {
                const int shard = batch % 3;
                if (batch == 0) worker_ids[tid] = std::this_thread::get_id();
                require(worker_ids[tid] == std::this_thread::get_id(), "Worker identity changed");
                FILE * file = open_buffered(files, paths[shard], opens);
                FILE *& first = handles[tid * 3 + shard];
                if (!first) first = file;
                require(first == file, "Buffered handle was not reused");
                read_and_check(file, (batch * 977 + tid * 4096) % (file_size - 128), shard);
            });
        }
        require(opens == 12, "Expected exactly one handle per worker/shard");
        std::atomic<int> completed{0};
        bool caught = false;
        try {
            pool.run(4, [&](int tid, ExpertIOFiles &) {
                if (tid == 0) {
                    ++completed;
                    throw std::runtime_error("Intentional worker exception");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                ++completed;
            });
        } catch (const std::runtime_error &) { caught = true; }
        require(caught && completed == 4, "Failure escaped before all active workers completed");
        pool.run(4, [&](int tid, ExpertIOFiles & files) {
            read_and_check(open_buffered(files, paths[0], opens), tid * 128, 0);
        });
        require(opens == 12, "Exception recovery reopened good handles");
        pool.run(0, [](int, ExpertIOFiles &) { throw std::runtime_error("Empty batch ran"); });
        caught = false;
        try { pool.run(5, [](int, ExpertIOFiles &) {}); }
        catch (const std::invalid_argument &) { caught = true; }
        require(caught, "Oversized worker batch was accepted");
    }
    require(closes == 12, "Destructor did not close every buffered handle");
}

static void failure_reuse_checks(const fs::path & late_file) {
    ExpertIOClosers closers;
    closers.buffered = close_buffered;
    std::atomic<int> opens{0};
    const int before = closes;
    {
        PersistentExpertIO pool(1, closers);
        bool caught = false;
        try { pool.run(1, [&](int, ExpertIOFiles & files) { open_buffered(files, late_file, opens); }); }
        catch (const std::runtime_error &) { caught = true; }
        require(caught, "Missing file did not propagate failure");
        fixture(late_file, 9);
        pool.run(1, [&](int, ExpertIOFiles & files) {
            read_and_check(open_buffered(files, late_file, opens), 0, 9);
        });
        caught = false;
        try {
            pool.run(1, [&](int, ExpertIOFiles & files) {
                FILE * file = open_buffered(files, late_file, opens);
                require(std::fseek(file, file_size - 4, SEEK_SET) == 0, "EOF seek failed");
                std::array<uint8_t, 64> bytes{};
                if (std::fread(bytes.data(), 1, bytes.size(), file) != bytes.size())
                    throw std::runtime_error("Intentional short read");
            });
        } catch (const std::runtime_error &) { caught = true; }
        require(caught, "Short read failure was lost");
        pool.run(1, [&](int, ExpertIOFiles & files) {
            // Production seeks before every buffered read, clearing EOF.
            read_and_check(open_buffered(files, late_file, opens), 128, 9);
        });
        require(opens == 1, "Failure/reuse did not preserve the one good handle");
    }
    require(closes == before + 1, "Recovery handle leaked on destruction");
}

#ifdef _WIN32
static void close_direct(void * handle) {
    ++direct_closes;
    CloseHandle(static_cast<HANDLE>(handle));
}

static void direct_checks(const fs::path & path) {
    ExpertIOClosers closers;
    closers.direct = close_direct;
    std::atomic<int> opens{0};
    {
        PersistentExpertIO pool(4, closers);
        for (int batch = 0; batch < 3; ++batch) pool.run(4, [&](int, ExpertIOFiles & files) {
            void *& handle = files.direct[path.string()];
            if (!handle) {
                HANDLE opened = CreateFileA(path.string().c_str(), GENERIC_READ, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
                if (opened == INVALID_HANDLE_VALUE) throw std::runtime_error("Synthetic direct open failed");
                handle = opened;
                ++opens;
            }
            files.aligned.resize(8192);
            auto * aligned = reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(files.aligned.data()) + 4095) & ~uintptr_t(4095));
            OVERLAPPED offset{};
            DWORD got = 0;
            require(ReadFile(static_cast<HANDLE>(handle), aligned, 4096, &got, &offset) && got == 4096,
                "Synthetic aligned direct read failed");
            for (size_t index = 0; index < 4096; ++index)
                require(aligned[index] == expected(index, 0), "Direct byte mismatch");
        });
        require(opens == 4, "Direct handles were not worker-local/reused");
    }
    require(direct_closes == 4, "Direct handles leaked on destruction");
}
#endif

int main(int argc, char ** argv) {
    try {
        const fs::path root = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() /
            ("qw-expert-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(fs::create_directory(root), "Use a new synthetic test directory");
        std::array<fs::path, 3> paths{{root / "a.bin", root / "b.bin", root / "c.bin"}};
        for (int index = 0; index < 3; ++index) fixture(paths[index], index);
        buffered_checks(paths);
        const fs::path late = root / "late.bin";
        failure_reuse_checks(late);
#ifdef _WIN32
        direct_checks(paths[0]);
#endif
        bool rejected = false;
        try { PersistentExpertIO invalid(0); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Unbounded/zero worker configuration was accepted");
        rejected = false;
        try { PersistentExpertIO invalid(65); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Oversized worker configuration was accepted");
        // Only remove the four known files created inside this new directory.
        for (const auto & path : paths) require(fs::remove(path), "Fixture removal failed (open handle?)");
        require(fs::remove(late) && fs::remove(root), "Test directory cleanup failed");
        std::cout << "{\"state\":\"passed\",\"persistent_workers\":4,\"buffered_closes\":"
            << closes << ",\"direct_closes\":" << direct_closes << "}" << std::endl;
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "expert I/O workers test failed: " << error.what() << std::endl;
        return 1;
    }
}

// Experimental matching RPC worker for the vendored QuestWend ggml revision.
// Always loopback-only; remote access uses the existing authenticated SSH path.
#include "ggml-backend.h"
#include "backend_loader.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    try {
        int port = 50064;
        size_t threads = 6;
        std::string device_names = "CUDA0,CPU", cache, model_index, model_root;
        bool list = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--list-devices") list = true;
            else if (i + 1 < argc && arg == "--port") port = std::stoi(argv[++i]);
            else if (i + 1 < argc && arg == "--threads") threads = std::stoul(argv[++i]);
            else if (i + 1 < argc && arg == "--devices") device_names = argv[++i];
            else if (i + 1 < argc && arg == "--cache-dir") cache = argv[++i];
            else if (i + 1 < argc && arg == "--model-index") model_index = argv[++i];
            else if (i + 1 < argc && arg == "--model-root") model_root = argv[++i];
            else throw std::runtime_error("Unknown or missing worker argument: " + arg);
        }
        if (port < 1 || port > 65535 || threads < 1 || threads > 256)
            throw std::runtime_error("Invalid worker port or thread count");
#ifdef _WIN32
        if (!model_index.empty()) _putenv_s("GGML_RPC_MODEL_INDEX", model_index.c_str());
        if (!model_root.empty()) _putenv_s("GGML_RPC_MODEL_ROOT", model_root.c_str());
#else
        if (!model_index.empty()) setenv("GGML_RPC_MODEL_INDEX", model_index.c_str(), 1);
        if (!model_root.empty()) setenv("GGML_RPC_MODEL_ROOT", model_root.c_str(), 1);
#endif
        questwend::initialize_backends();
        if (list) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto dev = ggml_backend_dev_get(i);
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(dev, &free, &total);
                std::cout << ggml_backend_dev_name(dev) << " : " << ggml_backend_dev_description(dev)
                          << " : " << free << " / " << total << " bytes" << std::endl;
            }
            return 0;
        }
        std::vector<ggml_backend_dev_t> devices;
        for (size_t start = 0; start < device_names.size();) {
            size_t end = device_names.find(',', start);
            std::string name = device_names.substr(start, end == std::string::npos ? end : end - start);
            auto dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) throw std::runtime_error("Worker device unavailable: " + name);
            devices.push_back(dev);
            if (end == std::string::npos) break;
            start = end + 1;
        }
        if (devices.empty()) throw std::runtime_error("No worker devices selected");
        auto reg = ggml_backend_reg_by_name("RPC");
        if (!reg) throw std::runtime_error("Build and deploy the matching ggml-rpc backend");
        using Start = void (*)(const char *, const char *, size_t, size_t, ggml_backend_dev_t *);
        auto start = reinterpret_cast<Start>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server"));
        if (!start) throw std::runtime_error("Matching RPC server entry point is unavailable");
        if (!cache.empty()) std::filesystem::create_directories(cache);
        std::string endpoint = "127.0.0.1:" + std::to_string(port);
        std::cout << "Experimental QuestWend RPC worker: " << endpoint << std::endl;
        start(endpoint.c_str(), cache.empty() ? nullptr : cache.c_str(), threads, devices.size(), devices.data());
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}

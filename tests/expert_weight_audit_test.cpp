#include "expert_weight_audit.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "expert-weight-audit-fixture.bin";
        uint8_t file_bytes[32];
        for (size_t index = 0; index < sizeof(file_bytes); ++index) file_bytes[index] = uint8_t(index * 3);
        FILE * file = std::fopen(path.c_str(), "wb");
        require(file != nullptr, "Cannot create byte audit fixture");
        const bool written = std::fwrite(file_bytes, 1, sizeof(file_bytes), file) == sizeof(file_bytes);
        require(std::fclose(file) == 0 && written, "Cannot write byte audit fixture");
        questwend::ExpertWeightAudit audit(1, 2, 1, true);
        require(!audit.begin(0, 10, 10), "Wrong layer triggered audit");
        require(!audit.begin(1, 1, 10), "Proactive single-expert fill consumed a full-group skip");
        require(!audit.begin(1, 10, 10) && !audit.begin(1, 10, 10), "Warmup skip count wrong");
        require(audit.begin(1, 10, 10), "Selected warmed group not audited");
        audit.source(path, 4, file_bytes + 4, 16, 0, 17, 3);
        bool copied = false;
        audit.destination(path, 8, 8, 1, 18, 9, [&](void * destination, size_t bytes) {
            std::memcpy(destination, file_bytes + 8, bytes);
            copied = true;
        });
        require(copied, "Destination callback was not checked");
        uint8_t corrupted[16];
        std::memcpy(corrupted, file_bytes + 4, sizeof(corrupted));
        corrupted[7] ^= 1;
        bool mismatch = false;
        try { audit.source(path, 4, corrupted, 16, 0, 17, 3); }
        catch (const std::runtime_error &) { mismatch = true; }
        require(mismatch, "A one-byte source mismatch was accepted");
        audit.end();
        require(!audit.begin(1, 10, 10), "Audit exceeded its full-group limit");
        require(!audit.active() && !audit.wants_gpu(), "Inactive audit retained GPU instrumentation");
        std::cout << "authoritative expert byte audit checks passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

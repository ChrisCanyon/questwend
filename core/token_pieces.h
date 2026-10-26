#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace questwend {
struct Vocab;
// Decode existing token IDs to their original bytes. No prompt encoding,
// merge tables, regex pre-tokenization, chat formatting or tool parsing.
class TokenPieces {
public:
    explicit TokenPieces(const Vocab & vocab);
    std::string decode(int32_t token) const;
    std::string decode(const std::vector<int32_t> & tokens) const;
private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<uint32_t,uint8_t> uni_to_byte_;
};
}

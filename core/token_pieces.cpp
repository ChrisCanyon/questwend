// Decode-only vocabulary adapter. Prompt BPE and templates live in llama.cpp.
#include "token_pieces.h"
#include "model.h"
namespace questwend {
struct CP { uint32_t cp; size_t off; size_t len; };

static std::vector<CP> utf8_decode(const std::string & s) {
    std::vector<CP> out;
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t) s[i];
        uint32_t cp; size_t len;
        if (c < 0x80)        { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; len = 4; }
        else                 { cp = c; len = 1; }
        for (size_t k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | ((uint8_t) s[i + k] & 0x3F);
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

// ---- GPT-2 byte<->unicode ----
static void build_byte_unicode(std::unordered_map<uint32_t, uint8_t> & u2b) {
    auto in_printable = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp;
        if (in_printable(b)) {
            cp = (uint32_t) b;
        } else {
            cp = (uint32_t) (256 + n);
            ++n;
        }
        u2b[cp] = (uint8_t) b;
    }
}

TokenPieces::TokenPieces(const Vocab & vocab) : id_to_token_(vocab.tokens) {
    build_byte_unicode(uni_to_byte_);
}
std::string TokenPieces::decode(int32_t token) const {
    if (token < 0 || token >= (int32_t) id_to_token_.size()) return "";
    const std::string & t = id_to_token_[token];
    std::string out;
    for (auto & cp : utf8_decode(t)) {
        auto it = uni_to_byte_.find(cp.cp);
        if (it != uni_to_byte_.end()) out.push_back((char) it->second);
    }
    return out;
}

std::string TokenPieces::decode(const std::vector<int32_t> & tokens) const {
    std::string out;
    for (int32_t t : tokens) out += decode(t);
    return out;
}

} // namespace questwend

#include "token_pieces.h"
#include "model.h"
#include <stdexcept>

int main() {
    questwend::Vocab vocabulary;
    // Golden byte-level vocabulary fixtures: ASCII, leading space, newline,
    // a multibyte character split across tokens, and a literal control marker.
    vocabulary.tokens = {"Hello", "\xc4\xa0world", "\xc4\x8a",
                         "\xc3\xa7", "\xc4\xae\xc2\xab", "<|im_end|>"};
    questwend::TokenPieces pieces(vocabulary);
    if (pieces.decode({0,1,2}) != "Hello world\n")
        throw std::runtime_error("ASCII/space/newline bytes changed");
    if (pieces.decode({3,4}) != "\xe7\x8c\xab")
        throw std::runtime_error("Split UTF-8 bytes changed");
    if (pieces.decode(5) != "<|im_end|>")
        throw std::runtime_error("Control marker bytes changed");
    if (!pieces.decode(-1).empty() || !pieces.decode(6).empty())
        throw std::runtime_error("Invalid token ID must produce no bytes");
}

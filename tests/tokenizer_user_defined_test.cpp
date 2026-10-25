// Model-free regression for GGUF USER_DEFINED versus CONTROL token semantics.
#include "model.h"
#include "tokenizer.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace questwend;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        Vocab vocab;
        for (int c = 33; c < 127; ++c) {
            vocab.tokens.push_back(std::string(1, static_cast<char>(c)));
            vocab.token_types.push_back(1);
        }
        const auto add = [&](const std::string & text, int type) {
            int id = static_cast<int>(vocab.tokens.size());
            vocab.tokens.push_back(text);
            vocab.token_types.push_back(type);
            return id;
        };
        const int control = add("<|im_end|>", 3);
        const int short_user = add("<tool", 4);
        const int user = add("<tool_call>", 4);
        int checks = 0;
        for (const auto & pre : {"qwen2", "deepseek-v3", "lfm2"}) {
        vocab.pre = pre;
        Tokenizer tokenizer(vocab);
        require(tokenizer.encode("<tool_call>", false, false) == std::vector<int32_t>{user},
                "USER_DEFINED must remain atomic with parse_special=false");
        require(tokenizer.encode("<tool", false, false) == std::vector<int32_t>{short_user},
                "Short USER_DEFINED token must remain atomic");
        const auto literal = tokenizer.encode("<|im_end|>", false, false);
        require(std::find(literal.begin(), literal.end(), control) == literal.end(),
                "CONTROL must remain disabled in literal message text");
        require(tokenizer.encode("<|im_end|>", false, true) == std::vector<int32_t>{control},
                "CONTROL must be enabled with parse_special=true");
        require(tokenizer.encode("x<tool_call>y", false, false) ==
                std::vector<int32_t>{tokenizer.token_to_id("x"), user, tokenizer.token_to_id("y")},
                "USER_DEFINED scanning must preserve surrounding text");
        require(tokenizer.encode("", false, false).empty(), "Empty input must remain empty");
        const auto plain = tokenizer.encode("abc123!", false, false);
        require(plain == tokenizer.encode("abc123!", false, true), "Ordinary text must not depend on special parsing");
        require(tokenizer.decode(plain) == "abc123!", "Ordinary text must round trip");
        checks += 8;
        }
        vocab.bos_id = control;
        Tokenizer with_bos(vocab);
        require(with_bos.encode("", true, false) == std::vector<int32_t>{control},
                "Explicit BOS insertion must be unaffected");
        vocab.token_types.clear();
        Tokenizer missing_types(vocab);
        require(missing_types.encode("abc", false, false) == missing_types.encode("abc", false, true),
                "Missing token-type metadata must retain ordinary encoding behavior");
        std::cout << checks + 2 << " tokenizer regressions passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

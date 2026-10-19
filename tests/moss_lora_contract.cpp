/* SPDX-License-Identifier: MIT */
#include "moss_lora_contract.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <set>
int main() {
    using namespace ps5_ai_moss_lora;
    unsigned accepted = 0, rejected = 0;
    std::set<std::string> targets;
    for (const auto& [name, shape] : expected()) {
        validate_tensor(name, true, 2, shape.input, shape.output);
        ++accepted; targets.insert(shape.model_weight);
        try { validate_tensor(name, false, 2, shape.input, shape.output); assert(false); } catch (const std::runtime_error&) { ++rejected; }
        try { validate_tensor(name, true, 2, shape.input + 1, shape.output); assert(false); } catch (const std::runtime_error&) { ++rejected; }
    }
    assert(accepted == 504 && targets.size() == 168 && rejected == 1008);
    validate_counts(504, 504, 0, 0, false);
    validate_counts(504, 504, 504, 0, true);
    for (auto values : {std::array<size_t,4>{503,504,504,0}, {504,503,504,0}, {504,504,503,0}, {504,504,504,1}, {504,504,505,0}}) {
        try { validate_counts(values[0],values[1],values[2],values[3],true); assert(false); } catch (const std::runtime_error&) { ++rejected; }
    }
    std::cout << "MOSS LoRA contract PASS accepted504 rejected" << rejected << " target_weights168; no GPU/model execution\n";
}

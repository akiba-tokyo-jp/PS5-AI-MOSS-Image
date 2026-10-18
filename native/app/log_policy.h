/* SPDX-License-Identifier: MIT */
#ifndef PS5_AI_LOG_POLICY_H
#define PS5_AI_LOG_POLICY_H
#ifdef __cplusplus
#include "json.hpp"
#include <stdexcept>
#include <string>
/* Read once before AGC/worker initialization. Missing policy means normal. */
static bool ps5_ai_log_policy_debug(const nlohmann::json& config) {
    if (!config.contains("logging")) return false;
    const auto& logging=config.at("logging");
    if (!logging.is_object()) throw std::runtime_error("logging must be an object");
    const auto level=logging.value("level",std::string("normal"));
    if (level!="normal" && level!="debug")
        throw std::runtime_error("logging.level must be normal or debug");
    return level=="debug";
}
#endif
#endif

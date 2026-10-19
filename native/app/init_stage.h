/* SPDX-License-Identifier: MIT -- finite context-init diagnostics, no cancellation. */
#ifndef PS5_AI_INIT_STAGE_H
#define PS5_AI_INIT_STAGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#define PS5_AI_INIT_NOEXCEPT noexcept
extern "C" {
#else
#define PS5_AI_INIT_NOEXCEPT
#endif
#define PS5_AI_INIT_MAX_RECORDS 32u
#define PS5_AI_INIT_RECORD_BYTES 384u
#define PS5_AI_INIT_JSON_BYTES 640u
enum ps5_ai_init_stage_id {
    PS5_AI_INIT_NOT_STARTED,
    PS5_AI_INIT_PREAMBLE,
    PS5_AI_INIT_PRECONTEXT_EVENT,
    PS5_AI_INIT_CONTEXT,
    PS5_AI_INIT_HEADERS,
    PS5_AI_INIT_DIT_HEADER,
    PS5_AI_INIT_QWEN_HEADER,
    PS5_AI_INIT_VAE_HEADER,
    PS5_AI_INIT_BACKENDS,
    PS5_AI_INIT_FILE_IO,
    PS5_AI_INIT_CORE_RUNNERS,
    PS5_AI_INIT_VAE_RUNNER,
    PS5_AI_INIT_VALIDATE,
    PS5_AI_INIT_TE_FLASH,
    PS5_AI_INIT_VAE_FLASH,
    PS5_AI_INIT_DIT_FLASH,
    PS5_AI_INIT_DENOISER,
    PS5_AI_INIT_STAGE_COUNT
};
enum ps5_ai_init_stage_state {
    PS5_AI_INIT_STATE_NONE,
    PS5_AI_INIT_LOG_PENDING,
    PS5_AI_INIT_OPERATION_STARTED,
    PS5_AI_INIT_OPERATION_RETURNED,
    PS5_AI_INIT_OPERATION_EXCEPTION
};
struct ps5_ai_init_stage_snapshot {
    uint64_t state_changed_ms;
    uint32_t stage, state, clock_valid;
    uint32_t journal_records_reserved, journal_calls_completed, journal_records_suppressed;
};
/* No lock is held across logging or model operations. Snapshot never logs,
 * waits, allocates, touches the model, queries a GPU, or accesses files. */
void ps5_ai_init_stage_enter(unsigned stage) PS5_AI_INIT_NOEXCEPT;
void ps5_ai_init_stage_returned(unsigned stage,int exception) PS5_AI_INIT_NOEXCEPT;
void ps5_ai_init_stage_snapshot(struct ps5_ai_init_stage_snapshot *out) PS5_AI_INIT_NOEXCEPT;
int ps5_ai_init_stage_json(char *out,size_t bytes) PS5_AI_INIT_NOEXCEPT;
#ifdef __cplusplus
}
#include <exception>
class PS5AIInitStageScope {
    unsigned stage_;
    int exceptions_;
public:
    explicit PS5AIInitStageScope(unsigned stage) noexcept
        : stage_(stage), exceptions_(std::uncaught_exceptions()) { ps5_ai_init_stage_enter(stage_); }
    ~PS5AIInitStageScope() noexcept {
        ps5_ai_init_stage_returned(stage_,std::uncaught_exceptions()>exceptions_);
    }
    PS5AIInitStageScope(const PS5AIInitStageScope&)=delete;
    PS5AIInitStageScope& operator=(const PS5AIInitStageScope&)=delete;
};
template<class F> auto ps5_ai_init_stage_call(unsigned stage,F&& function) -> decltype(function()) {
    PS5AIInitStageScope scope(stage);
    return function();
}
#endif
#undef PS5_AI_INIT_NOEXCEPT
#endif

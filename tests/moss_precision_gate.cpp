/* SPDX-License-Identifier: GPL-3.0-or-later
 * Host-only synthetic backend. Real GGML metadata/quantizer; no GPU or model.
 * Synthetic graph outputs test admission/refusal, not GPU numerical accuracy.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "probe.h"
#include "json.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static ggml_backend_dev_t gpu_device = reinterpret_cast<ggml_backend_dev_t>(uintptr_t(42));
static std::string scenario;
static int graphs = 0, buffer_frees = 0, backend_frees = 0, events = 0;
static int target_case() { return scenario.find("-q4-matrix-large") != std::string::npos ? 5 : scenario.find("-q4-matrix-small") != std::string::npos ? 4 : scenario.find("-bf16-matrix") != std::string::npos ? 3 : scenario.find("-bf16") != std::string::npos ? 2 : scenario.find("-flash") != std::string::npos ? 1 : 0; }
static std::string fault() { const auto pos = std::min({scenario.find("-bf16"), scenario.find("-q4-matrix"), scenario.find("-flash")}); return scenario.substr(0, pos); }
static ps5_ai_ggml_dispatch_stats stats{10, 10, 10, 0, 1};
static void write_event(const nlohmann::json &value) {
    assert(value.at("CPU_fallback") == false && value.at("model_inference") == false); ++events;
    assert(value.at("CPU_wall_time") == true);
    const double elapsed = value.at("graph_and_wait_seconds").get<double>();
    assert(std::isfinite(elapsed) && elapsed >= 0);
#if defined(PS5_AI_ENABLE_FP16) && PS5_AI_ENABLE_FP16
    assert(value.at("requested_fp16_math") == true);
#else
    assert(value.at("requested_fp16_math") == false);
#endif
    if (value.at("case") == "F32_Q_F16_KV_head128_heads24_tokens8")
        assert(value.at("FA_accumulation") == "f32");
}
#include "moss_precision_gate.inc"

struct ggml_backend_buffer { std::vector<void *> allocations; };
extern "C" {
ggml_backend_t ggml_backend_vk_init(size_t device) { assert(device == 0); return scenario == "no-backend" ? nullptr : reinterpret_cast<ggml_backend_t>(uintptr_t(1)); }
bool ggml_backend_is_vk(ggml_backend_t) { return scenario != "not-vulkan"; }
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t) { return scenario == "wrong-device" ? nullptr : gpu_device; }
void ggml_backend_free(ggml_backend_t) { ++backend_frees; }
void ggml_backend_synchronize(ggml_backend_t) {}
bool ggml_backend_supports_op(ggml_backend_t, const ggml_tensor *op) {
    // Source-linked regression: omitting the model's explicit F32 FA policy
    // must refuse the ordinary success path in either requested math mode.
    if (op->op == GGML_OP_FLASH_ATTN_EXT) assert(op->op_params[3] == GGML_PREC_F32);
    return !(fault() == "unsupported" && graphs == target_case());
}
ggml_backend_buffer_t ggml_backend_alloc_ctx_tensors(ggml_context *ctx, ggml_backend_t) {
    if (fault() == "allocation" && graphs == target_case()) return nullptr;
    auto *result = new ggml_backend_buffer;
    for (auto *t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        t->data = std::malloc(ggml_nbytes(t)); assert(t->data); result->allocations.push_back(t->data);
    }
    return result;
}
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    assert(!stats.work_pending); for (void *p : buffer->allocations) std::free(p); delete buffer; ++buffer_frees;
}
void ggml_backend_tensor_set(ggml_tensor *t, const void *p, size_t offset, size_t bytes) { assert(offset + bytes <= ggml_nbytes(t)); std::memcpy(static_cast<char *>(t->data) + offset, p, bytes); }
void ggml_backend_tensor_get(const ggml_tensor *t, void *p, size_t offset, size_t bytes) { assert(offset + bytes <= ggml_nbytes(t)); std::memcpy(p, static_cast<const char *>(t->data) + offset, bytes); }
void ggml_backend_tensor_memset(ggml_tensor *t, uint8_t value, size_t offset, size_t bytes) { assert(offset + bytes <= ggml_nbytes(t)); std::memset(static_cast<char *>(t->data) + offset, value, bytes); }
ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { return GGML_BACKEND_BUFFER_USAGE_COMPUTE; }
ggml_status ggml_backend_graph_compute(ggml_backend_t, ggml_cgraph *graph) {
    auto *out = ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1);
    const auto f = moss_precision::fixture_for_case(unsigned(graphs));
    assert(size_t(ggml_nelements(out)) == (graphs == 1 ? f.expected.size() : size_t(f.left_rows) * f.right_cols));
    auto *data = static_cast<float *>(out->data);
    std::fill(data, data + ggml_nelements(out), 0.f);
    for (size_t i = 0; i < f.expected.size(); ++i) data[f.sampled_output_indices.empty() ? i : f.sampled_output_indices[i]] = float(f.expected[i]);
    const bool active = graphs == target_case();
    if (fault() == "mismatch" && active) data[0] += 1.f;
    if (fault() == "nan" && active) data[0] = std::numeric_limits<float>::quiet_NaN();
    if (fault() == "nan-unsampled" && active) data[2] = std::numeric_limits<float>::quiet_NaN();
    ++graphs;
    if (fault() != "no-dispatch" || !active) ++stats.dispatches;
    if (fault() != "no-submit" || !active) ++stats.submits;
    if (fault() != "no-fence" || !active) ++stats.fenced_completions;
    if (fault() == "unverified" && active) stats.hardware_verified = 0;
    if (fault() == "pending" && active) stats.work_pending = 1;
    return fault() == "compute-failed" && active ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS;
}
void ps5_ai_ggml_dispatch_snapshot(ps5_ai_ggml_dispatch_stats *out) { *out = stats; }
int ps5_ai_ggml_wait_backend(const char *phase, int) {
    return !(fault() == "upload-unknown" && graphs == target_case() && std::strstr(phase, "upload")) &&
           !(fault() == "result-unknown" && graphs == target_case()+1 && std::strstr(phase, "result")) &&
           !(fault() == "readback-unknown" && graphs == target_case()+1 && std::strstr(phase, "readback")) &&
           !(fault() == "cleanup-unknown" && graphs == target_case()+1 && std::strstr(phase, "cleanup"));
}
__attribute__((noreturn)) void ps5_ai_ggml_retain(const char *why) {
    // A process exit models the production non-returning retain. In every
    // uncertain-completion case no graph-owned buffer/backend may be freed.
    assert(buffer_frees == target_case() && backend_frees == 0); std::printf("RETAIN %s current_graph_not_freed=1 previous_completed_graphs_freed=%d\n", why, buffer_frees); std::exit(42);
}
void ps5_ai_ggml_report(const char *, ...) {}
int ps5_ai_ggml_log_healthy() { return !(fault() == "log-unhealthy" && graphs == target_case()+1); }
}

static void oracle_checks() {
    using namespace moss_precision;
    const auto a = matmul_fixture(), repeat = matmul_fixture();
    assert(a.quantized.size() == 204 && a.expected.size() == 119);
    assert(std::memcmp(a.quantized.data(), repeat.quantized.data(), a.quantized.size() * sizeof(block_q4_K)) == 0);
    assert(a.expected == repeat.expected);
    for (int row = 0; row < m; ++row) {
        std::vector<float> decoded(k); dequantize_row_q4_K(a.quantized.data() + row * (k / QK_K), decoded.data(), k);
        for (int d = 0; d < k; ++d) assert(decoded[d] == a.a[size_t(row) * k + d]);
        for (int col = 0; col < n; ++col) {
            long double expected = 0;
            for (int d = k - 1; d >= 0; --d) expected += static_cast<long double>(decoded[d]) * a.b[size_t(col) * k + d];
            assert(std::fabs(double(expected) - a.expected[size_t(col) * m + row]) < 1.e-9);
        }
    }
    const auto f = flash_fixture(); assert(f.expected.size() == 24576);
    double minimum = 1.e10, maximum = -1.e10;
    for (int row = 0; row < tokens; ++row) for (int head = 0; head < heads; ++head) for (int d = 0; d < dim; ++d) {
        long double numerator = 0, denominator = 0;
        float lo = 1.e10f, hi = -1.e10f;
        for (int col = 0; col < tokens; ++col) {
            long double dot = 0;
            for (int x = dim - 1; x >= 0; --x)
                dot += static_cast<long double>(f.a[(size_t(head) * tokens + row) * dim + x]) * f.b[(size_t(head) * tokens + col) * dim + x];
            const long double weight = std::exp(dot * flash_scale);
            const float value = ggml_fp16_to_fp32(f.value[(size_t(head) * tokens + col) * dim + d]);
            numerator += weight * value; denominator += weight; lo = std::min(lo, value); hi = std::max(hi, value);
        }
        const double value = f.expected[(size_t(row) * heads + head) * dim + d];
        assert(std::fabs(double(numerator / denominator) - value) < 1.e-12 && value >= lo && value <= hi);
        minimum = std::min(minimum, value); maximum = std::max(maximum, value);
    }
    assert(maximum - minimum > 0.1);
    for (int cols : {7, 9}) {
        const auto bf = bf16_matmul_fixture(cols);
        assert(bf.expected.size() == size_t(m * cols));
        size_t rounded = 0;
        for (size_t i = 0; i < bf.b.size(); ++i) rounded += bf.b[i] != bf.bf16_rounded_rhs[i];
        assert(cols == 7 ? rounded == 0 : rounded > bf.b.size() / 2);
        for (int col = 0; col < cols; ++col) for (int row = 0; row < m; ++row) {
            long double sum = 0;
            for (int d = k - 1; d >= 0; --d) {
                const float weight = ggml_bf16_to_fp32(bf.bf16_weight[size_t(row) * k + d]);
                assert(weight == bf.a[size_t(row) * k + d]);
                const float rhs = cols == 7 ? bf.b[size_t(col) * k + d] : ggml_bf16_to_fp32(ggml_fp32_to_bf16(bf.b[size_t(col) * k + d]));
                sum += static_cast<long double>(weight) * rhs;
            }
            assert(std::fabs(double(sum) - bf.expected[size_t(col) * m + row]) < 1.e-9);
        }
    }
    const auto q9 = matmul_fixture(m, 9);
    assert(q9.expected.size() == 153 && q9.sampled_output_indices.empty());
    for (int col = 0; col < 9; ++col) for (int row = 0; row < m; ++row) {
        long double sum = 0;
        for (int d = k - 1; d >= 0; --d) sum += static_cast<long double>(q9.a[size_t(row) * k + d]) * q9.b[size_t(col) * k + d];
        assert(std::fabs(double(sum) - q9.expected[size_t(col) * m + row]) < 1.e-9);
    }
    const auto large = matmul_fixture(3072, 304, true);
    assert(large.a.size() == k && large.b.size() == size_t(k) * 304 && large.quantized.size() == size_t(k / QK_K) * 3072);
    assert(large.expected.size() == 99 && large.sampled_output_indices.size() == 99);
    std::vector<size_t> indices = large.sampled_output_indices;
    std::sort(indices.begin(), indices.end());
    assert(indices.front() == 0 && indices.back() == size_t(3072) * 304 - 1 && std::adjacent_find(indices.begin(), indices.end()) == indices.end());
    std::vector<float> large_actual(size_t(3072) * 304, 0.f), decoded(k);
    for (size_t i = 0; i < large.expected.size(); ++i) {
        const size_t row = large.sampled_output_indices[i] % 3072, col = large.sampled_output_indices[i] / 3072;
        dequantize_row_q4_K(large.quantized.data() + row * (k / QK_K), decoded.data(), k);
        long double sum = 0;
        for (int d = k - 1; d >= 0; --d) sum += static_cast<long double>(decoded[d]) * large.b[col * k + d];
        assert(std::fabs(double(sum) - large.expected[i]) < 1.e-9);
        large_actual[large.sampled_output_indices[i]] = float(large.expected[i]);
    }
    assert(!compare_fixture(large_actual, large, .001, .01).mismatches);
    large_actual[large.sampled_output_indices.back()] += 1.f;
    assert(compare_fixture(large_actual, large, .001, .01).mismatches == 1);
    large_actual[2] = std::numeric_limits<float>::infinity();
    assert(compare_fixture(large_actual, large, .001, .01).invalid == 1); // un-sampled output is still checked.
    auto bad_sample = large; bad_sample.sampled_output_indices.back() = large_actual.size();
    bool bad_index = false; try { compare_fixture(large_actual, bad_sample, .001, .01); } catch (const std::runtime_error &) { bad_index = true; } assert(bad_index);
    std::vector<float> actual; for (double x : f.expected) actual.push_back(float(x));
    assert(!compare(actual, f.expected, .002, .02).mismatches);
    actual[0] += .5f; assert(compare(actual, f.expected, .002, .02).mismatches == 1);
    actual[0] = std::numeric_limits<float>::infinity(); assert(compare(actual, f.expected, .002, .02).invalid == 1);
    actual[1] = std::numeric_limits<float>::quiet_NaN(); assert(compare(actual, f.expected, .002, .02).invalid == 2);
    bool bad_shape = false; try { compare({}, f.expected, .002, .02); } catch (const std::runtime_error &) { bad_shape = true; } assert(bad_shape);
    std::puts("ORACLES Q4_K=119 FA=24576 BF16_vector=119 BF16_matrix=153 Q4_matrix_small=153 Q4_matrix_large=99 full_finite_check=933888 reference/controlflow_only GPU_calls=0 model_inference=0");
}

int main(int argc, char **argv) {
    assert(argc == 2); scenario = argv[1];
    if (scenario == "oracles") { oracle_checks(); return 0; }
    try {
        moss_precision_readback_gate(); assert(scenario == "success");
        assert(graphs == 6 && events == 6 && buffer_frees == 6 && backend_frees == 1);
        std::puts("SYNTHETIC_ACCEPT graphs=6 GPU_calls=0");
    } catch (const std::runtime_error &e) {
        assert(scenario != "success"); assert(!stats.work_pending);
        assert(backend_frees == (scenario == "no-backend" ? 0 : 1));
        std::printf("REFUSE %s GPU_calls=0\n", e.what());
    }
}

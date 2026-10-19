/* SPDX-License-Identifier: GPL-3.0-or-later
 * Fixed-data MOSS operator admission, not a model inference or capacity proof.
 * The CPU only constructs numerical references; all tested graphs must run on
 * the already selected shared Vulkan device with positive hardware receipts.
 */
#include "ggml-quants.h"

namespace moss_precision {
constexpr int k = 3072, m = 17, n = 7;
constexpr int dim = 128, heads = 24, tokens = 8;
constexpr float flash_scale = 0.08838834764831845f;

struct Fixture {
    int left_rows = m;
    int right_cols = n;
    std::vector<float> a, b, v;
    std::vector<block_q4_K> quantized;
    std::vector<ggml_bf16_t> bf16_weight;
    std::vector<float> bf16_rounded_rhs;
    std::vector<ggml_fp16_t> key, value;
    std::vector<double> expected;
    std::vector<size_t> sampled_output_indices;
};

static Fixture matmul_fixture(int left_rows = m, int right_cols = n, bool sampled = false) {
    static_assert(k % QK_K == 0, "Q4_K rows must contain whole quantization blocks");
    Fixture f;
    if (left_rows <= 0 || right_cols <= 0 || (sampled && (left_rows != 3072 || right_cols != 304)))
        throw std::runtime_error("MOSS fixed Q4_K fixture shape mismatch");
    f.left_rows = left_rows; f.right_cols = right_cols;
    // The large diagnostic keeps one source row, not a full F32 model-sized
    // matrix. Only the stored Q4_K matrix, RHS and GPU output are complete.
    f.a.resize(sampled ? size_t(k) : size_t(k) * left_rows);
    f.b.resize(size_t(k) * right_cols);
    f.quantized.resize(size_t(k / QK_K) * left_rows);
    uint32_t state = 0x6d6f7373u;
    auto next = [&]() { state = state * 1664525u + 1013904223u;
        return float(int((state >> 16) & 1023u) - 511) / 1024.f; };
    for (int row = 0; row < left_rows; ++row) {
        float *source = f.a.data() + (sampled ? 0 : size_t(row) * k);
        for (int d = 0; d < k; ++d) source[d] = next();
        block_q4_K *q = f.quantized.data() + size_t(row) * (k / QK_K);
        quantize_row_q4_K_ref(source, q, k);
        // The oracle uses exactly the stored Q4_K weights, not their FP32 source.
        if (!sampled) dequantize_row_q4_K(q, source, k);
    }
    for (float &x : f.b) x = next() * 0.5f;
    if (sampled) {
        std::vector<float> decoded(k);
        for (int row : {0, 1, 15, 16, 31, 32, 127, 128, 1535, 1536, 3071}) {
            dequantize_row_q4_K(f.quantized.data() + size_t(row) * (k / QK_K), decoded.data(), k);
            for (int col : {0, 1, 7, 8, 31, 32, 127, 128, 303}) {
                double sum = 0;
                for (int d = 0; d < k; ++d) sum += double(decoded[d]) * f.b[size_t(col) * k + d];
                f.sampled_output_indices.push_back(size_t(col) * left_rows + row);
                f.expected.push_back(sum);
            }
        }
    } else {
        f.expected.resize(size_t(left_rows) * right_cols);
        for (int col = 0; col < right_cols; ++col) for (int row = 0; row < left_rows; ++row) {
            double sum = 0;
            for (int d = 0; d < k; ++d) sum += double(f.a[size_t(row) * k + d]) * f.b[size_t(col) * k + d];
            f.expected[size_t(col) * left_rows + row] = sum;
        }
    }
    return f;
}

static Fixture flash_fixture() {
    Fixture f;
    constexpr size_t count = size_t(dim) * heads * tokens;
    f.a.resize(count); f.b.resize(count); f.v.resize(count);
    f.key.resize(count); f.value.resize(count); f.expected.resize(count);
    for (size_t i = 0; i < count; ++i) {
        f.a[i] = float(int((i * 17 + i / dim * 3) % 41) - 20) / 32.f;
        f.key[i] = ggml_fp32_to_fp16(float(int((i * 11 + i / dim) % 37) - 18) / 32.f);
        f.value[i] = ggml_fp32_to_fp16(float(int((i * 7 + i / dim * 5) % 43) - 21) / 32.f);
        f.b[i] = ggml_fp16_to_fp32(f.key[i]); f.v[i] = ggml_fp16_to_fp32(f.value[i]);
    }
    for (int head = 0; head < heads; ++head) for (int row = 0; row < tokens; ++row) {
        double score[tokens], maximum = -1.e300, denominator = 0;
        for (int col = 0; col < tokens; ++col) {
            double dot = 0;
            for (int d = 0; d < dim; ++d)
                dot += double(f.a[(size_t(head) * tokens + row) * dim + d]) * f.b[(size_t(head) * tokens + col) * dim + d];
            score[col] = dot * double(flash_scale); maximum = std::max(maximum, score[col]);
        }
        for (double &s : score) { s = std::exp(s - maximum); denominator += s; }
        for (int d = 0; d < dim; ++d) {
            double sum = 0;
            for (int col = 0; col < tokens; ++col)
                sum += score[col] / denominator * f.v[(size_t(head) * tokens + col) * dim + d];
            // flash_attn_ext output is [dim, heads, query_tokens, batch].
            f.expected[(size_t(row) * heads + head) * dim + d] = sum;
        }
    }
    return f;
}

static Fixture bf16_matmul_fixture(int right_cols) {
    Fixture f;
    f.right_cols = right_cols;
    f.a.resize(size_t(k) * m); f.b.resize(size_t(k) * right_cols);
    f.bf16_weight.resize(f.a.size()); f.bf16_rounded_rhs.resize(f.b.size());
    f.expected.resize(size_t(m) * right_cols);
    uint32_t state = 0x62663136u;
    auto next = [&]() { state = state * 1664525u + 1013904223u;
        return float(int((state >> 8) & 65535u) - 32767) / 65536.f; };
    for (size_t i = 0; i < f.a.size(); ++i) {
        f.bf16_weight[i] = ggml_fp32_to_bf16(next());
        f.a[i] = ggml_bf16_to_fp32(f.bf16_weight[i]);
    }
    for (size_t i = 0; i < f.b.size(); ++i) {
        f.b[i] = next() * 0.5f;
        // Pinned Vulkan uses its vector path through N=8 with F32 RHS intact.
        // The matrix path converts F32 RHS to BF16, then sums in FP32. Check
        // both actual paths without assuming native BF16 arithmetic support.
        f.bf16_rounded_rhs[i] = right_cols > 8 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(f.b[i])) : f.b[i];
    }
    for (int col = 0; col < right_cols; ++col) for (int row = 0; row < m; ++row) {
        double sum = 0;
        for (int d = 0; d < k; ++d)
            sum += double(f.a[size_t(row) * k + d]) * f.bf16_rounded_rhs[size_t(col) * k + d];
        f.expected[size_t(col) * m + row] = sum;
    }
    return f;
}

struct Comparison { uint64_t invalid = 0, mismatches = 0; double maximum = 0; };
static Comparison compare(const std::vector<float> &actual, const std::vector<double> &expected,
                          double absolute, double relative) {
    if (actual.size() != expected.size() || actual.empty()) throw std::runtime_error("MOSS oracle shape mismatch");
    Comparison result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool finite = std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double error = finite ? std::fabs(double(actual[i]) - expected[i]) : 0;
        result.invalid += !finite;
        result.mismatches += !finite || error > absolute + relative * std::fabs(expected[i]);
        result.maximum = std::max(result.maximum, error);
    }
    return result;
}
static Comparison compare_fixture(const std::vector<float> &actual, const Fixture &fixture,
                                  double absolute, double relative) {
    if (fixture.sampled_output_indices.empty()) return compare(actual, fixture.expected, absolute, relative);
    if (actual.size() != size_t(fixture.left_rows) * fixture.right_cols ||
        fixture.sampled_output_indices.size() != fixture.expected.size() || fixture.expected.empty())
        throw std::runtime_error("MOSS sampled oracle shape mismatch");
    Comparison result;
    // Every GPU output must be finite, including the un-compared values.
    for (float x : actual) result.invalid += !std::isfinite(x);
    for (size_t i = 0; i < fixture.expected.size(); ++i) {
        if (fixture.sampled_output_indices[i] >= actual.size()) throw std::runtime_error("MOSS sampled oracle index mismatch");
        const float value = actual[fixture.sampled_output_indices[i]];
        const bool finite = std::isfinite(value) && std::isfinite(fixture.expected[i]);
        const double error = finite ? std::fabs(double(value) - fixture.expected[i]) : 0;
        result.invalid += !std::isfinite(fixture.expected[i]);
        result.mismatches += !finite || error > absolute + relative * std::fabs(fixture.expected[i]);
        result.maximum = std::max(result.maximum, error);
    }
    return result;
}
static Fixture fixture_for_case(unsigned test) {
    if (test == 0) return matmul_fixture();
    if (test == 1) return flash_fixture();
    if (test == 2 || test == 3) return bf16_matmul_fixture(test == 2 ? 7 : 9);
    if (test == 4) return matmul_fixture(m, 9);
    if (test == 5) return matmul_fixture(3072, 304, true);
    throw std::runtime_error("MOSS unknown fixed oracle case");
}
} // namespace moss_precision

static void moss_precision_readback_gate() {
    struct Owned {
        ggml_backend_t backend = nullptr; ggml_context *ctx = nullptr; ggml_backend_buffer_t buffer = nullptr;
        void clear() noexcept {
            if (!backend) return;
            ggml_backend_synchronize(backend);
            ps5_ai_ggml_dispatch_stats state{}; ps5_ai_ggml_dispatch_snapshot(&state);
            if (state.work_pending || !ps5_ai_ggml_wait_backend("moss_precision_cleanup", 0))
                ps5_ai_ggml_retain("moss_precision_cleanup_completion_unknown");
            if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
        }
        ~Owned() noexcept { if (backend) { clear(); ggml_backend_free(backend); } }
    } r;
    // vk_init(0) obtains another backend handle for the existing shared device.
    // It must not select a CPU backend, a scheduler, or a second VkDevice.
    r.backend = ggml_backend_vk_init(0);
    if (!r.backend || !ggml_backend_is_vk(r.backend) || ggml_backend_get_device(r.backend) != gpu_device)
        throw std::runtime_error("MOSS precision gate requires the existing shared hardware Vulkan device");
    for (unsigned test = 0; test < 6; ++test) {
        using namespace moss_precision;
        Fixture fixture = fixture_for_case(test);
        const bool bf16 = test == 2 || test == 3;
        const char *case_name = test == 1 ? "F32_Q_F16_KV_head128_heads24_tokens8" : test == 2 ? "BF16_weight_F32_RHS_K3072_M17_N7_vector" : test == 3 ? "BF16_weight_F32_RHS_K3072_M17_N9_matrix" : test == 4 ? "Q4_K_K3072_M17_N9_matrix" : test == 5 ? "Q4_K_K3072_M3072_N304_matrix_sample99" : "Q4_K_K3072_M17_N7";
        r.ctx = ggml_init({1 << 20, nullptr, true});
        if (!r.ctx) throw std::runtime_error("MOSS precision gate metadata unavailable");
        ggml_tensor *a, *b, *v = nullptr, *out;
        if (test != 1) {
            a = ggml_new_tensor_2d(r.ctx, bf16 ? GGML_TYPE_BF16 : GGML_TYPE_Q4_K, k, fixture.left_rows);
            b = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, k, fixture.right_cols);
            out = ggml_mul_mat(r.ctx, a, b);
        } else {
            a = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, dim, tokens, heads);
            b = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F16, dim, tokens, heads);
            v = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F16, dim, tokens, heads);
            out = ggml_flash_attn_ext(r.ctx, a, b, v, nullptr, flash_scale, 0.f, 0.f);
        }
        ggml_set_input(a); ggml_set_input(b); if (v) ggml_set_input(v); ggml_set_output(out);
        auto *graph = ggml_new_graph_custom(r.ctx, 16, false); ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (!ggml_backend_supports_op(r.backend, ggml_graph_node(graph, i)))
                throw std::runtime_error("MOSS precision operation unsupported; CPU fallback forbidden");
        const size_t output_values = test == 1 ? fixture.expected.size() : size_t(fixture.left_rows) * fixture.right_cols;
        if (size_t(ggml_nelements(out)) != output_values)
            throw std::runtime_error("MOSS precision output layout mismatch");
        r.buffer = ggml_backend_alloc_ctx_tensors(r.ctx, r.backend);
        if (!r.buffer) throw std::runtime_error("MOSS precision buffer unavailable");
        if (test != 1) {
            if (bf16) ggml_backend_tensor_set(a, fixture.bf16_weight.data(), 0, fixture.bf16_weight.size() * sizeof(ggml_bf16_t));
            else ggml_backend_tensor_set(a, fixture.quantized.data(), 0, fixture.quantized.size() * sizeof(block_q4_K));
            ggml_backend_tensor_set(b, fixture.b.data(), 0, fixture.b.size() * sizeof(float));
        } else {
            ggml_backend_tensor_set(a, fixture.a.data(), 0, fixture.a.size() * sizeof(float));
            ggml_backend_tensor_set(b, fixture.key.data(), 0, fixture.key.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(v, fixture.value.data(), 0, fixture.value.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_synchronize(r.backend);
        if (!ps5_ai_ggml_wait_backend("moss_precision_upload", 0)) ps5_ai_ggml_retain("moss_precision_upload_completion_unknown");
        ps5_ai_ggml_dispatch_stats before{}, after{}; ps5_ai_ggml_dispatch_snapshot(&before);
        if (before.work_pending) ps5_ai_ggml_retain("moss_precision_before_graph_pending");
        ps5_ai_ggml_report("BEFORE moss_precision case=%s CPU_fallback=0 model_loaded=0", case_name);
        const ggml_status status = ggml_backend_graph_compute(r.backend, graph);
        ggml_backend_synchronize(r.backend);
        if (status != GGML_STATUS_SUCCESS || !ps5_ai_ggml_wait_backend("moss_precision_result", 1))
            ps5_ai_ggml_retain("moss_precision_compute_completion_unknown");
        ps5_ai_ggml_dispatch_snapshot(&after);
        if (after.work_pending) ps5_ai_ggml_retain("moss_precision_result_pending");
        if (!after.hardware_verified || after.dispatches <= before.dispatches || after.submits <= before.submits || after.fenced_completions <= before.fenced_completions)
            throw std::runtime_error("MOSS precision lacks hardware dispatch/submit/fence proof");
        std::vector<float> actual(output_values);
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        ggml_backend_synchronize(r.backend);
        if (!ps5_ai_ggml_wait_backend("moss_precision_readback", 1)) ps5_ai_ggml_retain("moss_precision_readback_completion_unknown");
        const double absolute = test == 1 ? 0.002 : test >= 2 ? 0.001 : 0.01;
        const double relative = test == 1 ? 0.02 : test >= 2 ? 0.01 : 0.03;
        const Comparison result = compare_fixture(actual, fixture, absolute, relative);
        write_event({{"event", "native_moss_precision_readback"}, {"case", case_name},
            {"CPU_oracle", test == 1 ? "stable double softmax of actual F32 Q and F16 KV" : test == 2 ? "double dot of actual BF16 weights and unchanged F32 RHS" : test == 3 ? "double dot of actual BF16 weights and pinned Vulkan BF16-rounded F32 RHS" : "current GGML Q4_K reference quantizer/dequantizer then double dot"},
            {"compared_values", fixture.expected.size()}, {"finite_checked_values", actual.size()},
            {"sampled_output_indices", fixture.sampled_output_indices}, {"nonfinite", result.invalid}, {"mismatches", result.mismatches}, {"max_absolute_error", result.maximum},
            {"absolute_tolerance", absolute}, {"relative_tolerance", relative}, {"dispatch_delta", after.dispatches - before.dispatches},
            {"submit_delta", after.submits - before.submits}, {"fence_delta", after.fenced_completions - before.fenced_completions}, {"CPU_fallback", false}, {"model_inference", false}});
        if (result.invalid || result.mismatches || !ps5_ai_ggml_log_healthy())
            throw std::runtime_error("MOSS precision numerical mismatch/nonfinite or unavailable durable receipt");
        r.clear();
    }
}

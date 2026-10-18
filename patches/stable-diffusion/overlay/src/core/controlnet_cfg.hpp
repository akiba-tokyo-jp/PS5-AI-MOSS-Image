// SPDX-License-Identifier: MIT
#ifndef PS5_AI_CONTROLNET_CFG_HPP
#define PS5_AI_CONTROLNET_CFG_HPP

#include <map>
#include "core/tensor.hpp"

namespace ps5_ai_controlnet {
using Residuals = std::vector<sd::Tensor<float>>;

inline uint64_t validate_residuals(const Residuals& residuals, int64_t batch, bool sdxl) {
    if (residuals.empty() || (sdxl && residuals.size() != 10))
        throw std::runtime_error("ControlNet residual count is invalid");
    uint64_t values = 0;
    for (const auto& tensor : residuals) {
        if (tensor.empty() || tensor.dim() != 4 || tensor.shape()[3] != batch)
            throw std::runtime_error("ControlNet residual batch shape is invalid");
        for (float value : tensor.values())
            if (!std::isfinite(value))
                throw std::runtime_error("ControlNet residual contains NaN/Inf");
        values += static_cast<uint64_t>(tensor.numel());
    }
    return values;
}

// The cache belongs to one denoiser evaluation, not merely to one step number.
// A sampler can evaluate different latents/sigmas under the same step number.
template<class Condition> class EvaluationResidualCache {
    std::map<const Condition*, Residuals> single_;
    const Condition* positive_ = nullptr;
    const Condition* negative_ = nullptr;
    Residuals pair_;
    bool split_failed_ = false;

    void split_pair() {
        try {
            auto& positive = single_[positive_];
            auto& negative = single_[negative_];
            positive.reserve(pair_.size());
            negative.reserve(pair_.size());
            for (auto& tensor : pair_) {
                positive.push_back(sd::ops::slice(tensor, 3, 0, 1));
                negative.push_back(sd::ops::slice(tensor, 3, 1, 2));
                // Release each batch-two allocation immediately, rather than
                // retaining a whole pair alongside both branch copies.
                tensor = {};
            }
            Residuals().swap(pair_);
            positive_ = negative_ = nullptr;
        } catch (...) {
            split_failed_ = true;
            throw;
        }
    }

public:
    template<class Compute> const Residuals& get(const Condition& condition, Compute&& compute) {
        if (split_failed_)
            throw std::runtime_error("ControlNet residual split failed; retry forbidden");
        auto found = single_.find(&condition);
        if (found != single_.end()) return found->second;
        if (&condition == positive_ || &condition == negative_) {
            split_pair();
            return single_.at(&condition);
        }
        Residuals result;
        compute(condition, &result);
        return single_.emplace(&condition, std::move(result)).first->second;
    }

    uint64_t set_pair(const Condition& positive, const Condition& negative, Residuals pair, bool sdxl) {
        if (&positive == &negative)
            throw std::runtime_error("ControlNet CFG branches must have distinct identities");
        const uint64_t checked_values = validate_residuals(pair, 2, sdxl);
        positive_ = &positive;
        negative_ = &negative;
        pair_ = std::move(pair);
        return checked_values;
    }

    const Residuals* pair(const Condition& positive, const Condition& negative) const {
        if (split_failed_)
            throw std::runtime_error("ControlNet residual split failed; retry forbidden");
        return positive_ == &positive && negative_ == &negative ? &pair_ : nullptr;
    }
};

inline bool guided_hint_shape_compatible(const int64_t* shape, int64_t width,
                                         int64_t height, int64_t channels, int64_t batch) {
    return shape != nullptr && shape[0] == width && shape[1] == height &&
           shape[2] == channels && shape[3] == 1 && batch >= 1;
}
}
#endif

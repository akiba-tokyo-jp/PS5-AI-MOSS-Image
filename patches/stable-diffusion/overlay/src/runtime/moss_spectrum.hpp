/* SPDX-License-Identifier: GPL-3.0-only
 * Conservative MOSS-only adapter for the existing SD.cpp latent Spectrum route.
 * References: runtime/spectrum.hpp in the pinned SD.cpp backport; Spectrum method
 * arXiv:2603.01623 and Rkkss/ComfyUI-Spectrum-MOSS at
 * 3fde0674b29342565e7f42a0674af48d65aabc4a (GPL-3.0-only).
 * This predicts completed post-CFG denoised LATENTS, not Comfy's last-block
 * hidden states. It does not reproduce that extension's algorithm or benchmarks.
 */
#ifndef SD_RUNTIME_MOSS_SPECTRUM_HPP
#define SD_RUNTIME_MOSS_SPECTRUM_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace sd_moss_spectrum {
struct Shape {
    std::size_t width=0,height=0,channels=0,batch=0;
    bool operator==(const Shape& s)const {
        return width==s.width && height==s.height && channels==s.channels && batch==s.batch;
    }
    std::size_t numel()const {
        if((width!=16 && width!=64) || height!=width || channels!=128 || batch!=1)
            throw std::runtime_error("MOSS Spectrum requires admitted square256/1024 latents");
        return width*height*channels;
    }
};
struct Receipt {
    std::uint64_t initialized=0,completed_steps=0,real_steps=0,forecast_steps=0;
    std::uint64_t anchor_count=0,history_payload_bytes=0,peak_payload_bytes=0,errors=0;
};
class State {
public:
    static constexpr std::size_t total_steps=50, warmup_steps=8, tail_steps=2;
    static constexpr std::size_t history_limit=6, degree=3;
    static constexpr float blend=0.4f, ridge=1.0f;
    static constexpr std::size_t max_numel=64*64*128;
    static constexpr std::size_t max_history_bytes=history_limit*max_numel*sizeof(float);
    static constexpr std::size_t max_peak_payload_bytes=(history_limit+1)*max_numel*sizeof(float);
private:
    std::array<std::vector<float>,history_limit> anchors_;
    std::array<float,history_limit> coords_{};
    std::size_t oldest_=0,count_=0,next_=0;
    Shape shape_{};
    float last_sigma_=std::numeric_limits<float>::infinity();
    bool started_=false,failed_=false;
    Receipt receipt_{};
    [[noreturn]] void fail(const char* why) {
        failed_=true;++receipt_.errors;
        throw std::runtime_error(why);
    }
    std::size_t at(std::size_t i)const {return (oldest_+i)%history_limit;}
    static float coordinate(std::size_t step) {
        // Preserve the existing SD.cpp 50-step normalization; not Comfy /49.
        return (static_cast<float>(step)/50.0f)*2.0f-1.0f;
    }
    void check_step(int one_based,float sigma,const Shape& shape) {
        if(!started_ || failed_)fail("MOSS Spectrum state is not admitted");
        if(one_based<1 || one_based>static_cast<int>(total_steps) ||
           static_cast<std::size_t>(one_based-1)!=next_)
            fail("MOSS Spectrum requires exactly one ordered evaluation per Euler step");
        if(!std::isfinite(sigma) || sigma<0.0f || sigma>1.000001f ||
           (next_>0 && !(sigma<last_sigma_)))
            fail("MOSS Spectrum requires a finite strictly decreasing flow schedule");
        std::size_t n=0;
        try {n=shape.numel();}catch(...){fail("MOSS Spectrum latent shape is not admitted");}
        if(n==0 || n>max_numel)fail("MOSS Spectrum payload exceeds fixed admission");
        if(count_ && !(shape==shape_))fail("MOSS Spectrum shape changed within generation");
    }
    void advance(bool forecast,float sigma) {
        ++next_;last_sigma_=sigma;receipt_.completed_steps=next_;
        if(forecast)++receipt_.forecast_steps;else ++receipt_.real_steps;
    }
    static void basis(float t,std::array<float,degree+1>& p) {
        p[0]=1.0f;p[1]=t;
        for(std::size_t j=2;j<=degree;++j)p[j]=2.0f*t*p[j-1]-p[j-2];
    }
    std::array<float,history_limit> weights() {
        constexpr std::size_t P=degree+1;
        std::array<std::array<float,P>,history_limit> X{};
        for(std::size_t k=0;k<count_;++k)basis(coords_[at(k)],X[k]);
        std::array<std::array<float,P>,P> A{},L{};
        for(std::size_t i=0;i<P;++i)for(std::size_t j=0;j<P;++j) {
            float a=0.0f;
            for(std::size_t k=0;k<count_;++k)a+=X[k][i]*X[k][j];
            A[i][j]=a+(i==j?ridge:0.0f);
            if(!std::isfinite(A[i][j]))fail("MOSS Spectrum nonfinite regression matrix");
        }
        for(std::size_t i=0;i<P;++i)for(std::size_t j=0;j<=i;++j) {
            float a=A[i][j];
            for(std::size_t k=0;k<j;++k)a-=L[i][k]*L[j][k];
            if(i==j) {
                if(!(a>0.0f) || !std::isfinite(a))fail("MOSS Spectrum Cholesky rejected");
                L[i][j]=std::sqrt(a);
            }else {
                if(!(L[j][j]>0.0f) || !std::isfinite(L[j][j]))fail("MOSS Spectrum invalid Cholesky pivot");
                L[i][j]=a/L[j][j];
            }
            if(!std::isfinite(L[i][j]))fail("MOSS Spectrum nonfinite Cholesky factor");
        }
        std::array<float,P> phi{},y{},v{};
        basis(coordinate(next_),phi);
        for(std::size_t i=0;i<P;++i) {
            float a=phi[i];for(std::size_t j=0;j<i;++j)a-=L[i][j]*y[j];
            y[i]=a/L[i][i];if(!std::isfinite(y[i]))fail("MOSS Spectrum nonfinite forward solve");
        }
        for(std::size_t ii=P;ii>0;--ii) {
            const std::size_t i=ii-1;float a=y[i];
            for(std::size_t j=i+1;j<P;++j)a-=L[j][i]*v[j];
            v[i]=a/L[i][i];if(!std::isfinite(v[i]))fail("MOSS Spectrum nonfinite reverse solve");
        }
        std::array<float,history_limit> w{};
        for(std::size_t k=0;k<count_;++k) {
            for(std::size_t j=0;j<P;++j)w[k]+=X[k][j]*v[j];
            if(!std::isfinite(w[k]))fail("MOSS Spectrum nonfinite forecast weight");
        }
        return w;
    }
public:
    State()=default;
    State(const State&)=delete;
    State& operator=(const State&)=delete;
    State(State&&)=default;
    State& operator=(State&&)=default;
    void init(std::size_t steps) {
        *this=State{};
        if(steps!=total_steps)fail("MOSS Spectrum trial requires exactly50 Euler steps");
        started_=true;receipt_.initialized=1;
    }
    bool should_predict(int one_based,float sigma,const Shape& shape) {
        check_step(one_based,sigma,shape);
        // Fixed alternating manual policy: no adaptive controller, no expansion.
        return next_>=warmup_steps && next_<total_steps-tail_steps &&
               count_==history_limit && ((next_-warmup_steps)%2==0);
    }
    void observe_actual(int one_based,float sigma,const Shape& shape,const float* data,std::size_t n) {
        check_step(one_based,sigma,shape);
        if(should_predict(one_based,sigma,shape))fail("MOSS Spectrum actual/forecast scheduling receipt disagrees");
        if(data==nullptr || n!=shape.numel())fail("MOSS Spectrum actual latent buffer mismatch");
        for(std::size_t i=0;i<n;++i)if(!std::isfinite(data[i]))
            fail("MOSS Spectrum rejects nonfinite completed latent");
        const std::size_t slot=count_<history_limit?at(count_):oldest_;
        try {anchors_[slot].assign(data,data+n);}
        catch(...) {fail("MOSS Spectrum owned host history allocation failed");}
        if(count_<history_limit)++count_;else oldest_=(oldest_+1)%history_limit;
        shape_=shape;coords_[slot]=coordinate(next_);
        receipt_.anchor_count=count_;
        receipt_.history_payload_bytes=count_*n*sizeof(float);
        if(receipt_.history_payload_bytes>max_history_bytes)fail("MOSS Spectrum history budget exceeded");
        receipt_.peak_payload_bytes=std::max(receipt_.peak_payload_bytes,receipt_.history_payload_bytes);
        advance(false,sigma);
    }
    std::vector<float> predict(int one_based,float sigma,const Shape& shape) {
        check_step(one_based,sigma,shape);
        if(!should_predict(one_based,sigma,shape))fail("MOSS Spectrum refuses unadmitted forecast step");
        const std::size_t n=shape.numel();
        if(receipt_.history_payload_bytes+n*sizeof(float)>max_peak_payload_bytes)
            fail("MOSS Spectrum forecast payload budget exceeded");
        const auto w=weights();
        std::vector<float> output;
        try {output.resize(n);}catch(...) {fail("MOSS Spectrum forecast allocation failed");}
        const float* last=anchors_[at(count_-1)].data();
        const float* prev=anchors_[at(count_-2)].data();
        for(std::size_t f=0;f<n;++f) {
            float cheb=0.0f;
            for(std::size_t k=0;k<count_;++k)cheb+=w[k]*anchors_[at(k)][f];
            const float linear=last[f]+0.5f*(last[f]-prev[f]);
            const float value=(1.0f-blend)*linear+blend*cheb;
            if(!std::isfinite(value))fail("MOSS Spectrum rejects nonfinite forecast; no clamping");
            output[f]=value;
        }
        receipt_.peak_payload_bytes=std::max(receipt_.peak_payload_bytes,
                                           receipt_.history_payload_bytes+n*sizeof(float));
        advance(true,sigma);
        return output;
    }
    Receipt receipt()const {return receipt_;}
    bool completed()const {return started_ && !failed_ && next_==total_steps &&
        receipt_.real_steps==30 && receipt_.forecast_steps==20;}
};
} // namespace sd_moss_spectrum
#endif

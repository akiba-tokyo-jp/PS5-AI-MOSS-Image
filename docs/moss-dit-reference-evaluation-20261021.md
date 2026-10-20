# Additional DiT references observed 2026-10-21

2026-10-21追記：下記の評価を参考にFFN、Attention、QKV/LoRA、RMSの4項目を実装し、PS5の1024で検証しました。[実測と採用判断](moss-dit-kernels-20261021.md)。以下は調査時点の評価です。

These are source-code findings, not PS5 benchmarks or adopted math changes.

| Reference | Confirmed mechanism | Current MOSS Turbo4 implications |
| --- | --- | --- |
| [Elastic-DiT](https://github.com/wangjiangshan0725/Elastic-DiT/tree/5be46a21b298e712bff7634bc90623e505060387) | Adaptive router skips blocks and selects reduced MLP widths; block feature caching. Released infer.py loads Qwen-Image plus a separate checkpoint, BF16 CUDA,30 steps. | Not a drop-in MOSS/GGML/Vulkan patch. Would require MOSS-specific compatible routing/weights or a separately validated adaptation. Changes computed function; no claimed2x on PS5. |
| [ComfyUI-CacheDiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT/tree/1d92bbd86ec59aa6223fe2368849b7413a1acb93) | Approximate reuse of residuals across differing denoising steps. README verified example Z-Image-Turbo9 steps/warmup3; Qwen50; Klein20. | No published MOSS-specific preset found in inspected utils.py. Four steps leave little reuse after warmup. No inference precision/quality equivalence or1.4–1.6x PS5 claim. Distinct from our exact per-generation text/RoPE cache. |
| [ComfyUI MOSS source](https://github.com/Comfy-Org/ComfyUI/blob/1045d48eb4cc1afd27c419a4765555440f192eab/comfy/ldm/moss_image/model.py) | Fused rms_adaln; joint q/k RMS+RoPE; fused QKV base projection; SiTU can feed activation quantization directly; prefetch and memory compiler. | QKV base projection already packed here, independent q/k/v LoRA preserved. Our RMS reduction is still separate from weight/modulation and RoPE. Reduction+epilogue fusion is a candidate, but arithmetic grouping and aliasing need numerical tests. Prefetch benefit is conditional on actual transfer/idle gaps; resident weights do not imply new useful disk transfers. |
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen/tree/a3bac2cc284adfcae41a297932970ad65b878033) | Eager/CUDA/Triton/HIP/Ascend support matrix, including RMS and quantization kernels. | No Vulkan backend in published matrix. Port algorithms to existing PS5 backend where justified, not Python/Torch packages or CUDA binaries. INT8/FP8 changes require matching kernels, model/activation format and quality tests; PC hardware timings are not PS5 estimates. |

The existing4-step authored schedule, F32 accumulation, selectedrank144 adapter and numerical gates stay unchanged during timestamp identification. No external reference implementation is executed on the Windows client or imported into PS5 inference in this measurement pass.


## Spectrum-MOSSを再確認した結果

[Rkkss/ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/tree/3fde0674b29342565e7f42a0674af48d65aabc4a) のmainを2026-10-21に再確認しました。先の50-step調査で参照したものと同じrevisionでした。READMEの検証条件は50 steps／Euler＋sgm_uniformで、記載の約2～2.5倍は作者の条件での報告です。PS5・Turbo4の実測値としては使いません。

[nodes.py](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS/blob/3fde0674b29342565e7f42a0674af48d65aabc4a/nodes.py) は最後のTransformer blockのhidden stateを履歴として保存し、予測したhidden stateへ時刻に依存する出力headを適用します。Transformerの一部の呼出しを近似で置き換える方式です。現在優先しているFFNやAttentionのkernel単体を速くする変更ではありません。

- 既定warmup8、tail2。4-stepでは全4回がwarmupに入り、forecastは0回です。
- 既定Chebyshev次数4で、ready判定には過去の実計算5回が必要です。warmupだけ短くしても、新しい4-step生成内の履歴は予測判定時に最大3回なので不足します。次数などを変える場合は別の精度・画質比較が必要です。
- Adaptiveのdetail guardと局所token誤差p95/p99によるhorizon調整は、将来、多数stepの近似高速化を評価する際の参考です。予測の正確さや画像品質の保証ではありません。
- hidden-state履歴は現PS5 latent予測と異なります。形状をB1×4096 image tokens×3072、履歴8と仮定すると、履歴payloadだけでF32は384MiB、F16は192MiB／branchです。これは形状からの見積りで、このComfyUI版のPS5実測peakではありません。
- 予測値のNaN/Inf置換やclampを行う箇所があり、PS5版の非有限値を拒否する条件と差があります。ライセンスはGPL-3.0-onlyです。丸ごと導入したり検証条件を弱めたりしていません。

以前PS5で試した[50-step latent Spectrum](moss-sampler-spectrum-memory-20261020.md)は、1024で1023.859406秒→617.938351秒（39.65%短縮）でした。こちらはSD.cppのlatent境界で30回の実CFG step＋20回の予測を使う別実装で、ComfyUIのhidden-state版そのものの成功・同一画質を証明しません。現在のTurbo4へは有効化していません。

今回行ったのはREADME・実コード・既存実装との照合と設定条件の確認です。PS5への接続、追加生成、アプリ／driver／memory設定の変更はありません。現在の優先対象は実測に基づくFFN GEMMとFlash Attentionのままです。

# DiTの不変計算再利用とF32 Vulkan融合

2026-10-20 JST。4項目を実装し、PS5実機で256→1024を生成しました。同じcoreから最適化を無効にした対照版も測り、**1024は51.146425秒→47.760891秒（6.62%短縮）**でした。両サイズとも、対照版・以前のTurbo4基準とPNGが全バイト一致しました。各構成・各サイズ1回の観測です。

![実機256出力。今回の両構成とも同一PNG](images/moss-turbo4-upload-256-20261020.png)

![実機1024出力。今回の両構成とも同一PNG](images/moss-turbo4-upload-1024-20261020.png)

## 実装した処理

| 対象 | 変更と確認範囲 |
|---|---|
| DiT内部のtext projection・2 text-refine層・RoPE | 生成1件の最初に計算し、後の3 forwardでGPU上の結果を再利用。Qwen conditioning cacheとは別。時刻・画像で変わる42層の出力は再利用しない |
| SiTU | 元の9個のpointwise graph操作をF32 Vulkan shaderへ統合。活性化の式・演算順序を維持 |
| Norm・modulation・residual・Q/K RoPE | RMSのreduction自体は従来どおり。重み乗算＋modulation、gated residual、Q/K重み乗算＋RoPEの読み書きを融合 |
| LoRA | 倍率1のSCALEを省略し、base出力へのADD、独立q/k/vのpacked出力を融合。Q4行列積本体のepilogue融合は未実装 |

cacheは同じrunner・モデル・adapter・backendの1生成内だけです。形状とconditioning値の完全一致を検査し、新しい生成の開始時とGPU完了確認後に破棄します。跨る要求の再利用、近似denoising cache、step省略、CPUへのDiT移動は行いません。

~~~mermaid
flowchart LR
  Q[PS5 CPU Qwen] --> T[Text projectionと2 refine層]
  T --> C[1生成内のGPU cache]
  R[1生成内で1回のRoPE] --> D[4 DiT forward]
  C --> D
  D --> F[F32 SiTU・norm・residual・LoRA融合]
  F --> V[PS5 Vulkan VAE]
  V --> P[PNGとOpenAI互換API]
~~~

## 同じ条件での測定

MOSS-Image-SFT Q4_K、Qwen3.5-9B Q4_K_M、FLUX.2 small VAE、合成rank144 **F32** adapter、NSFW0.7＋Turbo1.0、runtime倍率1です。4 Euler steps／CFG1、seed20261020、固定prompt、author sigmas [1.0,0.910,0.717,0.024,0.0]、二重shiftなし。FFN分割・F16 LoRA storage・Spectrum・一般fusionはOFFです。全learned推論はPS5内で実行しています。

| 計測項目 | 同じcoreの最適化なし | 4項目有効 | 差 |
|---|---:|---:|---:|
| 256初回、モデルlazy load込み | 71.354991秒 | 69.275742秒 | −2.079249秒、−2.91% |
| 同プロセスの256後1024初回 | 51.146425秒 | 47.760891秒 | −3.385534秒、−6.62% |
| 1024 DiT graph | 44.557450秒 | 41.207966秒 | −3.349484秒、−7.52% |
| 1024 other runners graph | 5.985294秒 | 5.994667秒 | ＋0.009373秒 |
| 1024 DiTのdispatch記録呼出 | 12,556回 | 7,232回 | −5,324回 |
| 1024 DiTのfence wait呼出 | 2,916回 | 1,867回 | −1,049回 |

上2行はnative jobの生成時間です。graphとdriver呼出の時間はCPU wall計測で、包含関係があり加算できません。fence waitはGPU実行待ちを含み、その全量が削除可能な同期負担という意味ではありません。1024は単独coldでも2回目の1024でもありません。温度・clock、OS disk cacheを統制しておらず、平均・分散や各最適化単独の寄与は未測定です。

## メモリ

物理Direct予約10GiB、推論予算9.5GiB、pool外reserve1GiBを維持しました。今回の高速化で予約量は縮小していません。

| 1024の記録 | 対照版 | 4項目有効 |
|---|---:|---:|
| owned Direct data高水位 | 5,934,579,712B | 6,107,971,584B |
| 生成後owned Direct data | 4,749,066,240B | 4,749,066,240B |
| API GPU-labelled高水位 | 5,968,183,296B | 6,137,233,408B |
| API CPU/process RSS高水位 | 11,578,556,416B | 11,540,660,224B |

owned peakは173,391,872B（165.359375MiB）増えました。融合出力の非in-place allocation等を含む構成全体の差で、cache単体の量ではありません。GPU完了後にcacheを破棄し、生成後の使用量は同値へ戻りました。CPU/GPU-labelled値とowned Directは重複する別の量で、足して物理RAM／専用VRAMにはできません。長期のメモリ安定性はこの2生成だけでは証明しません。

## 数値と実行の検証

- 既存14数値ゲート、Turbo3、bounded-uploadゲートを維持。追加の融合16ケースは125,436値を元のGPU graphと比較し、非有限値0・不一致0・最大絶対誤差0でした。許容差は従来どおりです。
- Q/K RoPEの1ケースでは、別のVkBufferが同じGGML仮想アドレスを持つ配置を実際に使用しました。入出力の重なりは実VkBufferとbyte範囲で検査します。sentinelアドレスの一致だけでは重なりと判定しません。
- 各生成で4つの有限DiT出力、Q4 F32記録672／F16累積0／invalid0、LoRA504 tensors適用／skip0、GPU DiT・flash attention・VAE完了を確認しました。
- 各生成のcache miss1・hit3、RoPE構築1、SCALE省略1,008を記録しました。SiTU170、norm-modulation各168、residual336、Q/K RoPE340、LoRA ADD504、q/k/v pack168の実GPU記録があります。2枝packは合成fixtureで検証済みですが、この選定adapterの本番graphでは選ばれていません。
- native GETとOpenAI互換APIのPNGを照合し、完全decodeと目視確認を行いました。256 SHAは be9e6ebad5da8ee7a4e536a40e905307a2aa24bcced8e86b600af9fa0aaa2ad7、1024 SHAは 9b9f359d8ab249a2b6ece206a6b4a0fe1bd8df0ce8cbe000d39fe8e1d515c31a です。
- ホスト検査はpolicy39、モデル形状graph6,913、storage重なり判定200,007項目。モデルweights・learned推論・GPU実行をホスト検査に使用していません。

両構成とも各要求1POST、自動再送0です。通常API終了後に独立観測でnative/gateway不在、同boot、既存サービス・登録・モデルmetadataの維持を確認しました。firmware更新、システム領域の変更、曖昧なGPU資源の手動解放はありません。

## ビルドと残る範囲

[ビルド手順](build.md)のTurbo4＋upload＋F32 parentに、coreの --moss-dit-optimizations とnativeの --moss-dit-optimization-mask 15 を追加します。mask0は同じcoreでの対照版、1／3／7は累積した限定比較用です。既定はOFFで、production admission0・固定要求・最大4試行・実容量／有限値／GPU完了／所有権の条件は維持します。

実装箇所は [model/cache](../patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp)、[typed opsとstorage判定](../patches/stable-diffusion/overlay/src/model/diffusion/moss_dit_ops.hpp)、[Vulkan shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_fused.comp)、[GPU数値検査](../native/app/moss_dit_gate.inc)、[LoRA](../patches/stable-diffusion/overlay/src/model/adapter/lora.hpp)です。

設計参考はApache-2.0の [vLLM-Omni MOSS実装](https://github.com/vllm-project/vllm-omni/blob/942a5daa6c094615812ed89749ef07da96b8edab/vllm_omni/diffusion/models/mossimage/ops.py)と同revisionのtransformerです。CUDA/Triton・BF16コードの直接移植ではなく、現在のPS5 F32演算順序に合わせたGGML/GLSL実装です。KohakuMossTrainerとNunchakuのコードは取り込んでいません。[出典](moss-optimization-provenance.json)を参照してください。

各項目単独の速度、任意prompt・形状、長時間の連続生成、Q4 GEMM epilogue統合は未検証または未実装です。今回の画像一致や6.62%をそれらへ一般化しません。

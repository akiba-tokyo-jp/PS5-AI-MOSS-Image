# DiT kernel 4項目の実装と1024実機比較

2026-10-21 JST。4項目を実装し、PS5で256→1024→1024を各構成で生成しました。**1024平均は47.481390秒→45.127473秒、4.96%短縮**しました。採用構成は既存DiT optimization mask15に、新しいkernel mask15とAttention occupancy調整を加えた版です。通常終了・process不在の独立観測後、同じ検証済みSELFを新しいプロセスで起動し、全数値検査とジョブなしのnative／OpenAI互換gatewayを確認しました。復帰後の追加生成POSTはありません。

![採用版のPS5 1024出力](images/moss-dit-kernels-1024-20261021.png)

## 実装と採用判断

| 優先 | 変更 | 判断 |
|---|---|---|
| 1 FFN | Q4_Kを既存F16 scratchへGEMMごとに一度展開し、F32累積の行列積で再利用。元のQ4 weightsは維持 | 採用。対照平均47.481390秒→46.543395秒 |
| 2 Attention | head128・F16 K/V・F32累積の限定形状で既存shared-KV shaderを選択。さらにこの経路だけAMD向けの未使用LDS occupancy制限を除去 | 元の制限を残した案は平均51.578863秒へ悪化し不採用。調整版はRMS版45.739973秒→45.127473秒となり採用 |
| 3 QKV／LoRA | QKV/OもQ4 predecode。独立q/k/vを保ち、LoRA加算・packを4要素/threadで処理 | 採用。FFN版46.543395秒→46.126669秒 |
| 4 RMS／変換 | 512-lane F32 RMS reductionとweight／modulationまたはQK RoPEを融合。epsilon1e-6とreduction topologyを維持 | 採用。QKV版46.126669秒→45.739973秒 |

上の寄与は累積構成同士の比較です。全組合せの独立寄与や統計的な有意差ではありません。Attentionのoccupancy調整は対象shaderのworkgroup資源に限定し、GPU driver・OSメモリ枠・物理Direct予約を変更しません。

## 同条件の計測

MOSS-Image-SFT Q4_K、PS5 CPU Qwen3.5-9B Q4_K_M、Vulkan DiT／small VAE、合成rank144 **F32** adapter（NSFW0.7＋Turbo1.0、runtime1.0）。固定prompt・seed20261020、4 Euler steps／CFG1、author sigmas [1.0,0.910,0.717,0.024,0.0]、二重shiftなしです。FFN chunk、F16 LoRA storage、Spectrum、近似denoising cache、一般fusionはOFFです。

| 構成 | 256初回・lazy load込み | 256後1024初回 | 同プロセス1024の2回目 | 1024の2回平均 |
|---|---:|---:|---:|---:|
| 従来のDiT融合版（今回の対照） | 69.111579 | 47.800313 | 47.162467 | 47.481390 |
| FFN predecode | 69.561725 | 46.836615 | 46.250174 | 46.543395 |
| FFN＋Attention shared KV・元のoccupancy制限 | 69.471026 | 51.873448 | 51.284278 | 51.578863 |
| FFN＋QKV／LoRA | 69.778932 | 46.429972 | 45.823365 | 46.126669 |
| FFN＋QKV／LoRA＋RMS | 69.641481 | 46.068369 | 45.411577 | 45.739973 |
| 4項目＋Attention occupancy調整 | 69.449735 | 45.415880 | 44.839066 | 45.127473 |

すべてPS5内のgeneration_secondsです。1024単独coldではありません。各構成は新しいnative processで256から開始し、生成POSTは1回ずつ・自動再送0です。各1024の2枚は同じPNGでした。温度・clock・OS disk cacheを統制した交互A/Bではなく、順次測定の2観測平均です。256の初回は僅かに増えており、すべてのサイズ・初回loadが高速化したという主張はしません。

1024 DiT graphのCPU wall平均は40.906064秒→38.558765秒です。waitを含むCPU wallであり純GPU timestampではありません。other runnersは約6秒で、主な短縮はDiT側でした。[集約値](moss-dit-kernels-20261021.json)、[前段のGPU timestamp内訳](moss-dit-gpu-profile-20261021.md)を参照してください。

## 数値・画像・メモリ

- 元の14数値ゲート、Turbo3、upload、既存融合16ケースを維持。採用版は追加20ケースもPS5で読み戻し、不一致0・非有限値0、所有buffer解放・完了fenceを確認しました。Q4行列積は従来のabs0.001／rel0.01、融合とAttentionはabs2e-5／rel2e-5のままです。
- 追加検査はFFN4、QKV2、Attention2、LoRA vec4の連続／strided6、RMSの連続／strided6です。行列積は実際のM/Kとtokens4,202を分けたfixtureで確認し、実生成では全42層を実行しました。fixture時間はshader compilationを含み、生成の速度指標にしません。
- 生成ごとに有限DiT出力4、Q4 F32記録672／F16累積0／invalid0、LoRA504適用／skip0、GPU DiT・Attention・VAE完了を確認しました。selected pathのreceiptも検査しています。Windows側でlearned推論は行っていません。
- OpenAI互換APIとnative GETのPNGが一致し、完全decodeと目視を確認しました。採用版の1024は2回とも全byte一致しました。
- **旧版と採用版のPNGは一致しません。** Q4→F16一時展開や融合の丸め差がdenoisingで広がります。1024比較のMAEは3.5000/255、PSNRは28.91dBでした。目立つ破綻は見られませんでしたが、この値や1promptの目視で同画質・全prompt品質を保証しません。NaN置換・clamp、step省略、低精度累積で結果を隠していません。

| owned Direct data（1024） | 対照 | 採用版 |
|---|---:|---:|
| 生成高水位 | 6,107,971,584B | 6,208,634,880B |
| 生成後 | 4,749,066,240B | 4,849,729,536B |

高水位・生成後とも96MiB増えました。採用版の高水位は約5.78GiBで、2回目に累積していません。物理Direct10GiB、推論予算9.5GiB、pool外reserve1GiBは維持しています。owned pool、APIのGPU-labelled値、CPU/process RSSは重複する別の量で加算できません。長時間安定性は未確認です。

![採用版256](images/moss-dit-kernels-256-20261021.png)

対照は以前の公開PNGと一致しました：[対照256](images/moss-turbo4-upload-256-20261020.png)／[対照1024](images/moss-turbo4-upload-1024-20261020.png)。履歴の[verification](verification.json)も保持しています。

## 再現・変更範囲

[ビルド手順](build.md)のTurbo4＋upload＋F32 parentに、coreの --moss-dit-optimizations --moss-dit-kernels、nativeの --moss-dit-optimization-mask 15 --moss-dit-kernel-mask 15 --moss-fa-shmem-no-limit を明示します。古いoptimization maskと新しいkernel maskは別です。新kernel bitsはFFN=1、Attention=2、QKV/LoRA=4、RMS=8。occupancy調整にはAttention bit2が必要です。未指定はOFFです。

fresh75 core、24 initializer／6 runtime providers、source-freezeと依存closureの前後hashを確認しました。測定に使ったソースbyteとSELFを私有作業領域へ保存し、公開source inventoryは文書更新後の公開treeとして別管理します。ホストpolicy94・selector239・既存parent39・storage overlap200,007・F32 recipe93検査、SPIR-V validationを通過しました。ホスト検査は実機推論を代替しません。

実装：[Vulkan](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp)、[RMS shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_rms.comp)、[LoRA vec4 shader](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/vulkan-shaders/moss_dit_fused.comp)、[GPU検査](../native/app/moss_kernel_gate.inc)、[行列積／Attention検査](../native/app/moss_kernel_matrix_gate.inc)、[policy](../tools/moss_kernel_policy.py)、[host selector試験](../tests/moss_kernel_selection.cpp)。既存GGML Vulkan kernelsを選択・接続し、RMS/LoRAをGLSLで実装しています。CUDA/Triton・ComfyUI runtimeを実行したものではありません。[出典とshader provenance](moss-optimization-provenance.json)を保持しています。

production admission0、固定prompt・seed・256／1024、最大4試行（失敗を含む）の境界を維持します。FW、システム領域、既存登録、GPU driverの変更や、不確かなGPU資源の手動解放は行っていません。kernel epilogueの更なる統合、任意prompt・長時間運用、温度／clockを統制した比較は残る範囲です。

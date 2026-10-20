# MOSS Turbo4の実機結果（2026-10-20）

PS5上の専用trialで、MOSS-Image-SFT Q4_KとNSFW0.7＋Turbo1.0の合成rank144 F32 adapterを使い、256／1024画像を生成してOpenAI Images互換APIから取得しました。WindowsはAPIクライアントです。text encodingはPS5 CPU、DiT・Flash Attention・VAEはPS5 Vulkanで処理します。

| PS5内の生成時間 | 50-step基準キャッシュ版 | Turbo4専用trial |
|---|---:|---:|
| 256×256初回、lazy load込み | 146.016289秒 | **71.564874秒** |
| 同processの256後、1024×1024初回 | 1023.859406秒 | **51.133430秒** |

単回差は256で50.99%、1024で95.01%の短縮です。正のpromptとseedは揃えましたが、steps50→4、CFG3→1、adapterとsigmasも変わります。同画質・同一PNGの比較ではなく、同一モデルのkernelだけを20倍速くした意味でもありません。各条件1回でOS disk cache未flush、温度・clockは統制していません。1024は単独coldでも2回目の1024でもなく、256を生成した同processでの最初の1024です。

## 固定した条件と実機確認

- 4-step Euler、CFG1、seed20261020。already-shifted sigmas `[1.0, 0.910, 0.717, 0.024, 0.0]`にflow shift5を二重適用しません。CFG1はnegative-conditioning forwardを実行しません。
- rank16のNSFW係数0.7とrank128 Turbo係数1.0を、独立q/k/vを保ってrank144へ合成しました。runtime multiplierは1.0です。[入力revision・全体SHA・converter・notice](moss-turbo4-reproduction.md)を固定し、元rank16も別に保持します。
- FP16データ／FP32累積を維持します。既存14数値ゲート＋3つのrank144 GPU合成／読み戻しゲートに加え、各画像で4つの有限DiT出力、Q4のF32累積記録672／F16累積0／invalid0／nonfinite0、LoRA504 loaded/matched/used・skip0を確認しました。
- native GETとOpenAI互換APIのPNGは各画像で全バイト一致しました。256／1024とも目視で大きな破綻がないことを確認しました。各要求は1POST・自動再送0です。これは多様なpromptの画質評価や長時間連続運用の保証ではありません。

1024のDiT graph処理は4回・44.542630秒、他runnerは49回・5.987196秒でした。これはnested CPU wallで、純GPU timestampや加算可能な内訳ではありません。完全な現行ログの対応区間で、256のcache hitは0件、1024ではpositive conditioningのcache hitを1件確認しました。CFG1のためnegative forwardはありません。

![PS5実機のTurbo4専用trialで生成した1024×1024・4stepsの画像](assets/moss-turbo4-1024-20261020.png)

生成後にnativeとgatewayを通常APIで終了し、別のprocess観測で両方の不在を確認しました。さらに独立した新規observerで、同boot・既存サービス／syscore／shellcore、GPU titleなし、3タイトルの登録と選択5モデルのmetadata維持を確認しました。強制signalと手動GPU解放は0です。

## メモリについて言える範囲

物理Direct予約は基準・Turbo4とも10GiBです。APIのGPU-labelled高水位は、1024で基準4,868,833,280B（約4.53GiB）に対しTurbo4は5,968,183,296B（約5.56GiB）でした。共有Directとdriver arena由来の上界で、純粋なVRAM使用量ではありません。API CPU値もprocessの累積高水位を含み、両値を加算できません。この1024生成のowned Direct data使用量は高水位5,934,579,712B、生成後4,749,066,240Bで、物理予約10,737,418,240Bそのものとは区別します。

合成rank144の係数payloadは1,337,720,832Bで、rank16より1,189,085,184B（約1.11GiB）増えます。このstorage差は物理RAM／VRAM増分の実測ではありません。転送・temporary workspace・解像度・highwaterが重なるため、係数サイズだけから実メモリ量を算出しません。

別の[bounded-upload試験](moss-bounded-weight-upload-20261020.md)では、workerあたり最大8MiB（3workerなら可能上限24MiB）のgrow-to-fit stagingで、同じ256条件のAPI `ru_maxrss × 1024`高水位が216.6875MiB下がりました。Direct予約・Direct高水位・GPU-labelled高水位は同じで、物理RAM／VRAM削減の証明ではありません。**Turbo4ではbounded-uploadを無効にしているため、この削減をTurbo4の結果へ足し合わせません。**

BF16重みのGPU読み戻しゲートは合格しており、backendの`bf16: 0`表示だけでBF16不可とは判断しません。一方、現行BF16 matrix経路のRHS丸めと、rank144合成係数がすべてBF16-exactではない点は残ります。現在のF32契約を変更するには別の数値・画像比較が必要です。

前段の[30-step sampler／Spectrum](moss-sampler-spectrum-memory-20261020.md)、[text cache](moss-additional-optimization-20261020.md)の結果は保存しています。今回もprivateのguarded trialでproduction admission0、12GiB予約目標は未達です。

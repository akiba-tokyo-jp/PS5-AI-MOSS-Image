# Turbo4：転送・FFN分割・F16 LoRAの実測

2026-10-20 JST。Turbo4とbounded uploadを併用したF32版を速度比較の基準にし、FFN分割とF16 LoRA storageを別々に検証しました。**F16はowned Direct使用量を減らしましたが、読込済み1024の高速化は確認できませんでした。** 公開版は検証済みの10GiB限定trialを対象とします。

![Turbo4とbounded uploadの256初回出力](images/moss-turbo4-upload-256-20261020.png)

![同じプロセスでの256後1024出力](images/moss-turbo4-upload-1024-20261020.png)

両画像はF32併用版の実機PNGです。先行Turbo4単独版とも全バイト一致しました。

## 同じ要求による単回比較

いずれもPS5内で推論する4-step Euler／CFG1、seed20261020、合成rank144のNSFW0.7＋Turbo1.0、runtime multiplier1.0です。author sigmasは `[1.0, 0.910, 0.717, 0.024, 0.0]`、二重shiftなし。WindowsはAPIクライアントだけです。

| 構成 | 256初回、lazy load込み | 同じプロセスでの256後1024初回 | 1024の基準差 |
|---|---:|---:|---:|
| F32 Turbo4＋bounded upload | 71.377746秒 | **51.092988秒** | 基準 |
| 上記＋FFN token chunk2048 | 71.294669秒 | 55.195208秒 | ＋4.102220秒、＋8.03% |
| F16 LoRA＋専用FP32 shared演算、FFN分割OFF | 68.897951秒 | 51.884730秒 | ＋0.791742秒、＋1.55% |

各構成・各サイズ1回です。OS disk cacheをflushせず、温度・clockも統制していません。1024は単独cold／2回目の1024ではありません。F16の256は基準より2.479795秒低い観測ですが、反復して成立するcold速度差とは扱いません。[先行するTurbo4単独試験](moss-turbo4-results-20261020.md)の71.564874／51.133430秒も別の保存履歴です。

FFN分割は1024でworkspaceの高水位を下げる一方、Q4 F32行列記録が672→1344回となり、時間は約8%増えました。256は同一PNG、1024は異なるPNGで、目視では大きな破綻を認めませんでした。速度基準では分割をOFFにします。[実験profile](../build-profiles/moss-performance-turbo-bounded-upload-ffn2048-fp16-f32accum.json)は比較用として保持します。

## メモリの数値を分けて読む

以下は同じプロセスでの256後1024初回に対応します。全構成で**物理Direct予約は10,737,418,240B（10GiB）のまま**です。

| 記録項目 | F32併用基準 | FFN chunk2048 | F16 storage |
|---|---:|---:|---:|
| owned Direct data高水位 | 5,934,579,712B | 5,383,815,168B | 5,265,719,296B |
| 生成後owned Direct data | 4,749,066,240B | 4,749,066,240B | 4,080,205,824B |
| API GPU-labelled高水位 | 5,968,183,296B | 5,426,167,808B | 5,299,322,880B |
| API CPU／process `ru_maxrss × 1024`高水位 | 11,595,284,480B | 11,788,664,832B | 11,578,638,336B |

F16のowned Direct高水位・生成後使用量・GPU-labelled値は、基準からそれぞれ668,860,416B（637.875MiB）低下しました。係数payloadの半減量とも一致します。一方、process RSS高水位の差は15.875MiBです。これらは異なる、重複する記録であり、加算して全物理RAM／専用VRAMの削減量にはできません。予約領域をOSへ637.875MiB返した意味でもありません。

[bounded upload](moss-bounded-weight-upload-20261020.md)のworkerあたり8MiB上限は維持しています。このhost staging制限、GPU上の重みstorage、workspace、process RSS、Direct予約は別の量です。

## F16で変えたことと確認範囲

[stream変換ツール](../tools/models/convert_moss_lora_f16.py)はF32合成元を保持して別のF16ファイルを作ります。係数はlossyで、61,089,117値がF16 subnormal、31,301値が0へ丸められました。単に「FP16の全演算へ切り替えた」構成ではありません。[専用backend経路](../native/app/moss_lora_fp32_shared_backend.inc)はLoRAのF16 storageを読み、RHS・shared演算と累積をF32に維持します。Q4基底モデル、VAE、一般のF16 pipelineはこの選択で置換しません。

既存14数値ゲート、3つのrank144合成ゲート、[bounded-uploadゲート](../native/app/moss_bounded_weight_upload_gate.inc)に加え、[専用8読み戻し](../native/app/moss_lora_f16_storage_gate.inc)が合格しました。固定の非dyadic入力と独立subnormal試験でF32／F16 storageを比較し、許容差は変更していません。実際にcompileした4 pipelineのmodule・specialization・handleを記録し、両画像で同じpipelineを使ったこと、LoRA private F16行列記録2016／F32記録0、4つの有限DiT出力、Q4 F32記録672／F16累積0／invalid0、LoRA504使用・skip0を確認しました。

両サイズともnative GETとOpenAI互換APIのPNGが一致し、目視で大きな破綻を認めませんでした。F32基準とはPNGが異なり、同画質・pixel一致は主張しません。1024要求にはpositive conditioningのcache hitを1件確認しました。各要求1POST・自動再送0、終了は通常API、その後に独立観測でnative/gateway不在・既存サービス・登録とモデルmetadata維持を確認しました。

F16版は[default-OFFの専用trial](../native/app/moss_lora_f16_storage_trial.h)です。検証したcanonical rank144とN362／4202に限定し、任意prompt・形状の通常運用成功へ拡張しません。また、基準はordinary F32 LoRA経路で、F16版は専用F32 shared経路も変わるため、時間差をstorageだけの効果へ分離できません。[F16用profile](../build-profiles/moss-performance-combined-lora-f16-fp32shared.json)と[対照F32用profile](../build-profiles/moss-performance-combined-lora-f32-fp32shared.json)は明示的に分けています。

## 次の高速化と未検証項目

読込済みF32基準51.09秒のうちDiT graphは44.50秒（87.1%）、他runnerは5.99秒でした。包含関係のあるCPU wall timeで、fence waitを削除可能な同期時間とは読みません。予約量を減らすだけで計算が速くなる根拠はありません。

残る候補は、合成LoRAで残る厳密なSCALE×1の省略と、up行列後のADD／q/k/v連結を限定epilogueへまとめる処理です。現4stepsでは前者が最大1008 SCALE、後者が672 ADD＋336 CONCATのgraph-node候補ですが、個別GPU時間は未測定で、短縮率は予測できません。N4202から計算したSCALEの約166.96GiBは名目read/write pass量で、実測物理転送ではありません。SCALEの `a*1+0` はsigned zero／FTZ等の確認が必要で、一般helperのcontiguous化やalias依存も維持する必要があります。いずれも未実装の候補です。

任意prompt用の8／10GiB通常運用profileは公開対象に含めません。8GiBで起動・物理予約は確認しましたが、最初の生成要求が保存領域チェックのHTTP507でジョブ登録前に拒否されたため、推論・連続動作の成功として扱いません。公開版はproduction admission0と固定prompt／seed・最大4試行の明示trialを維持します。重み、生ログ、token、ビルド済みbinaryはGitへ含めません。

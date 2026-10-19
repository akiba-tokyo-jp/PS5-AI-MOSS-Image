# MOSS性能改善の実装と比較検証（2026-10-20）

**3点の実装と比較検証を完了しました。FP32基準・提出/fusion構成の256/1024生成は成功し、対応するPNGは完全一致しました。FP16候補は起動時の数値ゲートで拒否され、生成していません。** 提出/fusionのFP32実験profileを残しますが、単回比較の差は256で約0.87%、1024で約0.38%と小さく、大幅高速化は達成していません。安定した改善率や最速構成の証明ではありません。リポジトリはPRIVATE、production admissionは`0`を維持します。

以前の[1024 cold/warm測定](moss-1024-cold-warm-20261020.md)では、warm生成1177.725365秒のうち、DiTの`graph_call_including_internal_wait`が1167.878713秒、約99.16%を占めました。今回は、その内部の観測、既存FP16経路の比較、提出単位と限定fusionの3点を実装しました。過去の速度を新構成の結果として扱いません。

## 1. graph_call内部の計測

共通の`phase_profile.h`とengine集計をschema 2、42項目へ拡張しました。既存のgraph、weight、workspace、upload、readback、cleanupに加え、DiTとother runnersを分けて次を記録します。

| 指標 | 観測する区間 |
|---|---|
| `vulkan_submit_driver_call` | 提出APIの呼び出し前後 |
| `vulkan_fence_wait_driver_call` | フェンス待ちAPIの呼び出し前後 |
| `vulkan_dispatch_record_call` | dispatchコマンド記録APIの呼び出し前後 |
| `vulkan_bind_pipeline_record_call` | pipeline bind記録APIの呼び出し前後 |
| `vulkan_backend_graph_call` | Vulkan backend graph呼び出し全体 |

すべて`CLOCK_MONOTONIC`による**CPU wall time**です。GPUのハードウェアtimestamp、kernel実行時間、使用率ではありません。4つのdriver API指標には`vk_common_GetDeviceProcAddr`のlookupと実際の呼び出しを含み、その前後のjournal処理は含みません。フェンス待ちは未完了GPU処理の待ちを含み得ます。dispatch/bindの指標はコマンド記録全体ではなく、dispatch記録が短くてもkernelが速いとは限りません。

`graph_total`、`graph_call`、backend graph、submit、fence waitには包含関係があります。**項目を合計して生成時間と比較しないでください。** runner所属はthread-local scopeで復元し、計測無効時・ネスト・例外時の扱いと無効な時計sample数も確認します。計測scopeはengine generationで、集計ログ自身を除外します。3構成すべてに同じ計測を入れています。

実装は[phase_profile.h](../native/app/phase_profile.h)、[engine](../native/app/engine_sdcpp.cpp)、[GGML runner](../patches/stable-diffusion/overlay/src/core/ggml_runner.cpp)、[Vulkan backend](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp)にあります。

## 2. 既存FP16経路の比較

新しい重み形式やkernelを追加する変更ではありません。FP16構成だけで、PS5のstartupが`GGML_VK_DISABLE_F16`自体を解除します。`=0`を設定する方法ではありません。baselineと提出/fusion構成はFP32を維持し、構成と`--precision`が一致しないビルドを拒否します。

実機では`shaderFloat16`、16bit storage、FP16有効token（`fp16: 1`または`fp16: dot2`）の起動ログと無効化変数の不在を確認します。既存MOSS数値読み戻しゲートの絶対・相対誤差しきい値は緩めません。NaN/Inf、mismatch、GPU完了証拠不足、memory admission失敗を合格として扱いません。FP16は丸めや蓄積精度を変え得るため、同条件の実PNGを比較してから採否を決めます。高速化、画像一致、全演算のFP16化は未確認です。

## 3. 提出単位と限定elementwise fusion

試験構成2だけが次を有効にします。

```text
GGML_VK_PS5_SUBMIT_BATCH=2
GGML_VK_PS5_ELEMENTWISE_FUSION=1
```

提出単位の変更は、既存の適応FLOPしきい値とnode数の判定しきい値へ2倍の倍率を適用します。nodeの判定しきい値は通常100から200になりますが、現在の包含判定では実際に101から201nodeまで1組へ入り得ます。厳密な200dispatch上限ではなく、nodeとdispatchも常に1対1ではありません。GPU計算量、submit回数、生成時間が必ず半分になるという意味ではありません。

限定fusionは、隣接する`UNARY(SIGMOID/SILU) → MUL`だけを対象に、既存の`unary_mul` shaderを選びます。F32、同一shape、対応する連続性・stride・u32 spanに加え、従来のconsumer、dependency、alias条件を要求します。MULの左右どちらにUNARYがある場合も扱います。GELU、GEMM、一般fusion全体の有効化ではありません。

serialized submissions、async無効、general fusion無効、scalar FP32、COOPMAT/COOPMAT2無効を必須条件として維持します。不正な環境値や`GGML_VK_MAX_NODES_PER_SUBMIT`との競合は拒否します。限定経路は追加scratchを要求せず、既存の実測scratch admission、region、完了・所有権guardを維持します。baseline/FP16では両実験knobを解除します。

新しいGPU読み戻しゲートは、共有backingの左右viewを使うSIGMOID/SILUとMULの4caseです。各caseの8192×304値を独立した倍精度の式と照合し、absolute/relative toleranceをともに`0.00002`に固定します。構成2では限定fusion選択1、他構成では0を要求し、CPU fallbackを禁止します。実際のdispatch、submit、fence、finite、mismatchとdurable receiptを確認してからmodel生成へ進みます。**この節は実装条件の説明であり、全構成の実機合格報告ではありません。**

実装は[option/metadata判定](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan-ps5-profile.h)、Vulkan backend、runner admission、scratch bounds、[専用ゲート](../native/app/moss_perf_gate.inc)にあります。

## 試験契約と比較条件

| variant | profile | precision | 追加提出/fusion |
|---|---|---|---|
| 0 | `moss-performance-baseline-fp32.json` | FP32 | 無効 |
| 1 | `moss-performance-fp16.json` | FP16 | 無効 |
| 2 | `moss-performance-submit-fusion-fp32.json` | FP32 | 上記の限定経路 |

新しい`ps5-ai-moss-experimental-performance-v1`は明示的な試験用opt-inです。既存256の1claim契約、既存1024の2claim契約は維持します。新契約は各process最大4回のvalid生成試行を許可します。不正requestはclaimを消費せず、有効requestは推論前にatomic claimを消費します。失敗も消費され、返却・resetはありません。4回の上限は、4枚を必ず生成する指示ではありません。

固定条件は次のとおりです。

- MOSS-Image-SFT Q4_K、Qwen3.5-9B Q4_K_M、FLUX.2 small VAE、選定rank16 LoRA0.7。
- 過去の1024 baselineと同じprompt/negative、seed20261020、50steps、CFG3、Euler、flow shift5。
- 256×256または1024×1024、10GiB Direct pool、1GiB outside reserve、`Vulkan0=9.5`の予算。
- 同じdriver・モデル・tokenizer・LoRAを維持。requested12GiB targetの達成を意味しません。

clientが256のPNG、数値、実機完了、目視確認、running/queued=0を確認してから、同じprocessの1024へ進みます。native counter自体が目視を判定するわけではありません。必要な場合だけ1024の同process2回目を手動要求します。POSTは1回ずつで、自動再送しません。タイムアウト時もjobとhealthで完了を確認してから判断します。

**256初回の後の1024は、新しいshapeの初回であって、model未読込のcold1024ではありません。** 比較対象は同じ計測入りFP32構成の同じ256→1024順です。過去のfresh-process cold/warm pairとは分離します。`generation_seconds`、API roundtrip、Windows wallは別々に報告します。共有poolの累積high-waterを含むCPU/GPU-labelled memory peakを加算したり、pure VRAMと表示したりしません。

各試験後は正常終了とowner不在・独立idle確認を行ってから次のGPU titleへ進みます。WAI、Anima、MOSSの同時起動、system領域・firmware変更を行う試験ではありません。

## 確認済みのホスト検証とクロスビルド

- 新contract139項目、旧256契約72項目、旧1024契約42項目が成功。実atomic helperで不正request非消費、失敗後のclaim維持、並行上限4、production0と旧契約を確認しました。ホストtestでモデル推論やGPU計算は行っていません。
- 共通core全75memberをfresh compileし、旧archive member再利用は0。ビルド前後の入力hashを確認しました。
- 3native SELFのinitializer24、runtime provider6、SELF integrity、要求precision/variant/compile flags、入力前後一致を確認しました。
- 3構成のcore、GGML Vulkan provider、RADV archive、生成shader headerはそれぞれbyte同一です。variantはnative startupで選びます。この一致自体がFP16やfusionの実機適用証明になるわけではありません。
- 完成したビルドは`optimization-v2`です。途中で中断したv1を成功扱いせず、v2へ部分objectを再利用していません。元manifest/sourcefreezeは保存しています。

## 実機比較結果

| 構成 | 256初回 | 1024初回（256後・context読込済み） | 1024任意の繰り返し | 数値・画質・終了確認 |
|---|---|---|---|---|
| baseline FP32 | 153.972336秒 | 1219.096190秒 | 未測定 | GPU生成・PNG確認、正常終了・idle確認 |
| FP16 | 生成未実施 | 生成未実施 | 生成未実施 | 起動数値ゲートで拒否、正常OS終了・idle確認 |
| 提出/fusion FP32 | 152.631656秒 | 1214.444805秒 | 未測定 | 両sizeのGPU生成・PNG確認、正常終了・idle確認 |

### FP32基準の観測

| 要求 | PS5 `generation_seconds` | API roundtrip | Windows wall |
|---|---:|---:|---:|
| 256初回、model読込を含む | 153.972336秒 | 155.234秒 | 155.781秒 |
| 同processの1024初回 | 1219.096190秒 | 1221.328秒 | 1226.906秒 |

1024の50steps、100 DiT forward、Vulkan DiT/VAE完了、全151返却F32出力のfinite確認で非有限値0、LoRA504 loaded/matched/used・skip0を確認しました。PNGは目視で正常、以前の1024出力とSHA256が一致しました。

```text
484391514834f4788b5bceb8c9b5d0ba26ce023f0fa861a51b1a3d6bd2b28f27
```

今回の1024後は正常終了を確認し、独立observer009でidleを確認しました。256→1024ではconditioning cacheのsize条件が変わり、過去の1024→1024 warmとは同じ条件ではありません。旧warmとの差をそのまま計測追加の減速や純粋なmodel読込時間と解釈しません。保存TAILに明示的なcache-hit行はなく、観測されていないhit数を補いません。

1024の主なCPU wall内訳は次のとおりです。**親子項目を足さないでください。**

| 指標 | 回数 | 秒 |
|---|---:|---:|
| DiT graph call、内部待ちを含む | 100 | 1169.098366 |
| DiT Vulkan backend graph call | 102 | 1168.964269 |
| DiT driver fence wait | 64691 | 1164.567069 |
| DiT driver submit | 64691 | 1.094833 |
| DiT dispatch record call | 310300 | 0.394776 |
| DiT pipeline bind record call | 310300 | 0.320132 |

フェンス待ちの1164秒にはGPU計算を待つ時間が含まれます。この値全体を「削除できる同期オーバーヘッド」と扱いません。可視のweight prepareは0.060834秒、activation uploadは0.096877秒、readbackは0.056284秒であり、これらは計測した呼び出しのscopeです。隠れた全転送量を測ったものではありません。

APIの`peak_gpu_bytes`は8,403,091,456Bでしたが、CPU text weightsも借りる共有Direct poolの生涯peak8,369,520,640Bとdriver arenaの生涯peak33,570,816Bの合計です。GPU専用VRAM実使用量ではありません。残っていた256側Direct使用量に一時的なCPU text weightsの19claim、4,835,246,080Bが重なったことでDirect peakを正確に説明でき、19claimはすべて返却されています。1024後はDirect使用3,559,440,384B、free7,177,977,856B、CPU text Direct live0、fault retainedなしでした。単一pairの確認を、あらゆるleakや断片化が起きない証明へ一般化しません。

### FP16候補の拒否と範囲

実機のFP16/16bit storage対応、無効化変数の不在、実ログ`fp16: 1`を確認しました。しかしQ4_K行列ゲート`K3072/M17/N9`で153値中41値が固定しきい値`abs 0.001 + rel 0.01 × |expected|`を超え、最大絶対誤差は0.0947755356、NaN/Infは0でした。起動admissionがこの候補を拒否したため、生成POSTは0、画像生成時間と画質は未測定です。汎用im2colの1/7という診断値は、今回の直接の拒否条件ではありません。

Q4_KゲートのCPU正解値は、保存した量子化重みを復号した値から計算しており、元の非量子化重みとの差を混ぜていません。ソースと実リンクshaderには、FP16有効・quantized DEFAULT precision時のhalf蓄積、half stagingに伴う追加丸めの経路があります。今回の差の説明候補ですが、各要因の寄与やruntime pipeline IDは実測で分離できていません。すべてのFP16用途が不可能、driver実装の欠陥確定、という結論ではありません。

FP16演算＋F32蓄積の別候補はソース上可能ですが、まだ実装・実機検証していません。閾値を緩めて合格にしたり、fixtureだけ変更してmodelも検証済みと扱ったりしません。今回の候補は正常OS終了要求が受理され、PID不在と独立observer010のidleを確認しました。

### 提出/fusion構成の256比較

固定4caseは各2,490,368値でmismatch0、NaN/Inf0となり、GPU読み戻し・限定fusion選択のゲートに合格しました。256のmodel生成もGPU完了、50steps、DiT100/VAE1を含む全103出力finite・NaN/Inf0、LoRA504 loaded/matched/used・skip0を確認しました。

| 256初回、model読込を含む | baseline FP32 | 提出/fusion FP32 |
|---|---:|---:|
| PS5 generation | 153.972336秒 | 152.631656秒 |
| DiT graph call、内部待ちを含む | 91.475941秒 | 89.930709秒 |
| DiT driver submit / fence回数 | 64258 | 62100 |
| DiT dispatch記録回数 | 335300 | 330950 |
| 限定fusion選択数 | 0 | 4350 |

PNGはbaselineと全バイト・復号後RGB/alphaが完全一致し、目視も正常でした。SHA256は次のとおりです。

```text
6b69b1915b9afa3375db3f66bb70d033014f88d6ac1892f0a3bc9004b577e52e
```

生成全体の差は−1.340680秒、−0.8707%でした。各構成1回だけの、別processでの初回読込を含む比較です。温度・clock・OS cacheを統制した反復比較ではなく、大幅高速化や再現性の確定とは扱いません。

実際のDiT dispatchは4350回、1.2973%減り、記録されたfusion選択数と一致しました。submit/fenceは2158回、3.3583%減りました。しきい値2倍が提出回数半減にならないことを実測でも確認しています。FLOP、node、graph末尾、almost-readyなどの提出境界を維持しており、個々の提出をどの条件が発火させたかは計測していません。提出単位とfusionを組み合わせた構成のため、時間差を各変更へ独立に割り当てることもできません。

DiT graph callの差は−1.545232秒でしたが、driver submit自体のCPU wall差は約−0.044154秒です。フェンス待ちの差をすべて無駄な同期時間とみなさず、親子scopeを合算しません。

### 提出/fusion構成の1024比較

| 1024初回、同processの256後 | baseline FP32 | 提出/fusion FP32 |
|---|---:|---:|
| PS5 generation | 1219.096190秒 | 1214.444805秒 |
| API roundtrip | 1221.328秒 | 1216.266秒 |
| Windows wall | 1226.906秒 | 1222.156秒 |
| DiT graph call、内部待ちを含む | 1169.098366秒 | 1164.535786秒 |
| DiT graph total | 1171.537857秒 | 1166.958508秒 |
| DiT driver submit / fence回数 | 64691 | 62471 |
| DiT dispatch記録回数 | 310300 | 305950 |
| 限定fusion選択数 | 0 | 4350 |

生成全体は−4.651385秒、−0.3815%、DiT graph callは−4.562580秒でした。dispatchは4350回（1.4019%）、submit/fenceは2220回（3.4317%）減りました。各構成1回で、温度・clock・OS cacheを統制した反復試験ではありません。256と同様に、単回差から確実な改善率を決めません。

driver fence waitは1164.567069→1159.916232秒、submit CPU wallは1.094833→1.052372秒、other runner graph totalは46.107206→46.029770秒でした。これらは包含関係のあるCPU wall指標で、GPU純実行時間や削除可能な同期時間へ読み替えません。固定した50steps、CFG3の100 DiT forwardと主要な行列・attention計算は残ります。今回の限定経路だけでは大幅な短縮には至りませんでした。

1024のPNGは2,496,997B、先述の`484391…b28f27`と全SHA256が一致し、baselineと全バイト・復号後RGBが同一、目視も正常でした。合計4PNG、2つのsizeごとの比較で完全一致を確認しています。追加の画像コピーは作らず、既存の正常1024 assetを使用します。

両構成の1024後Direct usedは3,559,440,384B、freeは7,177,977,856B、Direct lifetime peakは8,369,520,640B、CPU text Direct live0、fault retainedなしでした。候補のAPI `peak_gpu_bytes`は8,402,403,328Bです。baselineとの差はdriver arenaの生涯peak差であり、GPU専用VRAMがその分減ったという意味ではありません。

### 残す構成と未検証範囲

インストール済みの実験profileは**variant2、FP32＋提出単位調整＋限定fusion**を維持します。FP16候補は現行しきい値を通らないため採用しません。これは別のmixed-precision候補が将来成立しないという判断ではありません。

最後のnative569/gateway572は通常API終了202を受理し、両PID不在と独立observer011のidleを確認しました。現在は停止しています。今回の許可された比較は完了し、追加の1024やwarm繰り返しは行っていません。max4/production0/privateを維持し、失敗時のclaim返却、自動POST再送、driver・モデル・poolの追加変更は行いません。

長期連続運用、平均・分散、熱・clock統制、他prompt、FP16での画像品質、最小steps、提出変更とfusion単体それぞれの寄与は未検証です。既存のWAI向け`precision.md`や`unet-profiling.md`の速度・CFG batch結果をMOSS実測として流用しません。

client結果と解析は`D:/PS5-AI/moss/optimization-20261020/`内の各`result.private.json`、`analysis-tools/baseline-large-v1`、`variant2-small-v1`、`variant2-large-v1`、`final-fp32-comparison-v1/final-evaluation.private.json`に保存しています。ビルドの根拠は`D:/PS5-AI/moss/build/optimization-v2/verified-handoff.private.json`、終了の独立根拠は`D:/PS5-AI/moss/native-observer/v1/`の`extracted-009`、`extracted-010`、`extracted-011`です。raw logs、これらのprivate diagnostic JSON、credentials、model weightsをGitへ追加しません。

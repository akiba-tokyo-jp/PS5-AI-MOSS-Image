# MOSSのFP16データ経路とFP32累積による短縮

> 過去の試験記録です。現在の採用候補・最新の秒数は[Turbo4・メモリ比較](moss-turbo-memory-options-20261020.md)と[検証状態](status.md)を参照してください。

2026-10-20 JST。PS5本体で同条件の256×256と1024×1024を生成し、WindowsからOpenAI互換APIでPNGを取得しました。FP16データ経路と明示的FP32累積を組み合わせた版では、1024のPS5内生成時間が1219.096190秒から1064.248911秒へ短縮しました。各構成1回の測定で、熱・クロック・OSキャッシュを統制した反復試験ではありません。

## 比較条件と結果

MOSS-Image-SFT Q4_K、Qwen3.5-9B Q4_K_M、FLUX.2 small VAE、指定rank16 LoRA 0.7、50steps、CFG3、Euler、flow shift5、seed20261020、同一promptを維持しました。Direct pool10GiB、推論予算9.5GiB、pool外reserve1GiB、RADVも変更していません。推論はPS5のCPUテキスト処理とVulkan DiT/VAEで行い、Windowsはクライアントです。

| PS5 `generation_seconds` | FP32基準 | 提出調整・限定fusion FP32 | FP16データ＋FP32累積 |
|---|---:|---:|---:|
| 256初回、モデル読込込み | 153.972336秒 | 152.631656秒 | **145.839851秒** |
| 同processの256生成後、1024初回 | 1219.096190秒 | 1214.444805秒 | **1064.248911秒** |

FP32基準比で256は8.132485秒（5.28%）、1024は154.847279秒（12.70%）短縮しました。1024は約20分19秒から約17分44秒です。提出調整・限定fusion版に対しても1024は150.195894秒短縮しました。

1024はモデルをまだ読んでいないprocessでの初回でも、1024を連続生成した2回目でもありません。256生成後にサイズを変更した最初の1024で、サイズ変更に伴うCPU conditioningを含みます。以前の[1024 cold/warm](moss-1024-cold-warm-20261020.md)とは条件が異なります。

## ログで特定した時間のかかる場所

| 1024の計測区間 | FP32基準 | FP16データ＋FP32累積 |
|---|---:|---:|
| DiT graph call、内部待ちを含む | 1169.098366秒 | 1014.144911秒 |
| other runners graph total | 46.107206秒 | 46.183293秒 |
| DiT dispatch記録回数 | 310300 | 310300 |
| DiT submit / fence回数 | 64691 | 64691 |

元の生成時間の約95.9%をDiT graph callが占め、今回の短縮もほぼこの区間に現れました。API通信や記録量の削減による差としては扱いません。これらは包含関係のあるCPU wall timeです。fence waitはGPUの実行待ちを含み、全時間を除去可能な同期オーバーヘッドと呼んだり、親子区間を足したりしません。純粋なGPUハードウェアtimestampや個別kernelの所要時間は未計測です。

## 実装した変更

- MOSSのLinear生成とruntime LoRAのLinearでFP32累積を明示しました。他モデルの既定precisionを変更しません。
- FP16版の起動では`GGML_VK_DISABLE_F16`を変数ごと解除し、実ログ`fp16: 1`と16bit capabilityを確認しました。
- 実際にcommandを記録したQ4行列pipelineの累積方式を集計します。各生成でF32累積16600、F16累積0、invalid0を確認しました。GPU完了は別のwitnessで確認します。
- core75オブジェクトを再構築し、nativeのcompile定義と一致することを検証しました。既存の容量・scratch・ownership・数値ゲートを維持しています。

最初のFP16候補はQ4_K N9ゲートで41/153値が許容値を超え、生成前に拒否されました。今回の版は累積方式を修正して通したもので、許容値を緩めたものではありません。Q4 N9/N304は既存のabs0.001＋rel0.01基準でmismatch0です。VAEは直接畳込みを使用し、未使用im2col診断のmismatch1/7は合格としていません。FP16有効ログは全演算がFP16になるという意味ではありません。

## 実機の画像・メモリ確認

256では103返却出力・3,964,928値、1024では151返却出力・62,554,112値を確認し、NaN/Infは0でした。各画像でLoRA504 loaded/matched/used、skipped0、DiT/VAEのGPU完了を確認しました。両PNGを開き、正常なアニメ画像であることを確認しています。native GETとOpenAI互換APIのPNGは全バイト一致しました。FP32版とのpixel一致は主張しません。

1024 PNGのSHA256は`97ff9462ebd81b772bb313a8d8bbc71212b83b0c4d34b655ba8393b81f7b9aab`、2,496,438 bytesです。APIのCPU peakは11,787,223,040B、GPU-labelled peakは8,403,091,456Bでした。後者はCPUテキスト重みも含む共有Direct poolの生涯peakとdriver arenaからなる上界です。両peakを足して物理使用量やGPU専用VRAMと扱いません。Direct peak8,369,520,640B、生成後used3,559,440,384B、CPUテキストDirect live0、fault retainedなしを確認しました。

native/gatewayは通常APIで終了し、両PID不在と独立observerのidleを確認しました。この試験ではproduction admission0と最大4試行の実験契約を維持しました。12GiB確保、長期安定性、他prompt、最小steps、再現性は今回の成功だけでは確定しません。

![今回の混合精度版でPS5が生成した1024画像](images/moss-mixed-1024-20261020.png)

## 追加候補の調査で分かったこと

FFN upgateを既存のmedium64タイルへ変更する案も実装して独立GPUゲートまで試しました。実機の数値結果はmismatch0、非有限値0でしたが、明示選択の確認は拒否され、画像POSTは0でした。

原因は実機NAVI21の非coopmat構成ではlarge候補が除外され、small/mediumの2候補になるのに、追加selectorが3候補を仮定していたことです。さらに既存selectorは今回のFFNサイズで既に同じmedium64を選びます。したがって候補数判定だけを直しても実行タイルは変わらず、高速化にはなりません。この案は採用せず、速度結果を捏造しません。nativeはGPUのquietと空のpoolを確認して通常OS終了し、独立observerでidleを確認しました。

その後、成功した上記の混合精度版と同じSELFへ戻し、全文読み戻しで一致を確認しました。再起動したnativeとgatewayがreadyとなり、必須14起動ゲート、6 MOSSゲート、4性能ゲートを元の許容値で再確認しました。この復元確認で画像POSTは追加していません。両APIを通常終了して両PID不在を確認し、独立observerでもGPUタイトル不在・idleを確認しました。現在は改善版を配置済みで停止しています。FW・システム領域・既存タイトルへの変更はありません。

## 証拠の所在

残る候補には、F32を維持したruntime LoRAのscale/add融合、MOSS固有のCFG batch化、実shapeに限定したattention調整があります。現在のログはQ4行列・attention・LoRAのGPU時間を分離していないため、その寄与や短縮率は未確定です。submit API自体は約1.09秒であり、ここだけを減らして大幅短縮とは見積もりません。大きなkernel改修前には、対応が確認できた方法で演算単位の計測と小さな数値・時間比較が必要です。

元の実測・source／artifact照合・数値・終了確認の記録は非公開の検証資料へ保存しています。生ログ、モデル、認証情報、private diagnostic JSONはGitへ含めません。

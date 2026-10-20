# 現行Turbo4のDiT処理時間をPS5で特定

2026-10-21 JST。**1024ではFFNのQ4行列積が約40.15%、Flash Attentionが約25.41%、QKVが約15.01%**を占めました。次の優先対象はFFNの行列積とAttentionです。ComfyUIのRMS融合は参考になりますが、RMS単体は約1.64%でした。

以下の割合は、現在の4-step生成内で採取した**完全なDiT forward 2回のGPU dispatch区間の合計**に対する値です。全生成47秒の排他的内訳ではありません。各サイズ1生成で、初回forwardのwitnessによる3分割viewを除き、その後の2・3回目のforwardを測定しました。全4 forwardを省略せず計算しています。

## 実機のGPU内訳

| 処理 | 256・2 forward合計 | 同GPU区間中の割合 | 1024・2 forward合計 | 同GPU区間中の割合 |
|---|---:|---:|---:|---:|
| FFN up/gate Q4行列積 | 0.476006秒 | 32.83% | 5.236539秒 | 26.41% |
| FFN down Q4行列積 | 0.264833秒 | 18.27% | 2.725899秒 | 13.75% |
| Flash Attention | 0.076056秒 | 5.25% | 5.039632秒 | 25.41% |
| QKV Q4行列積 | 0.273850秒 | 18.89% | 2.975969秒 | 15.01% |
| Attention出力OのQ4行列積 | 0.097055秒 | 6.69% | 0.995355秒 | 5.02% |
| LoRA down/up行列積と付随処理 | 0.132614秒 | 9.15% | 1.263481秒 | 6.37% |
| contiguous化 | 0.029168秒 | 2.01% | 0.379149秒 | 1.91% |
| RMS正規化 | 0.027580秒 | 1.90% | 0.324258秒 | 1.64% |
| 上記以外 | 0.072630秒 | 5.01% | 0.891265秒 | 4.49% |
| **測定GPU区間の合計** | **1.449793秒** | **100%** | **19.831548秒** | **100%** |

親graph演算への分類には、その演算が発行する変換・縮約等のdispatchも含みます。同じtimestampを実際のshader種別でも集計したところ、1024の行列積shaderは13.200663秒（66.56%）、Attentionは5.039632秒（25.41%）、融合shaderは0.844777秒（4.26%）、変換・copyは0.415179秒（2.09%）、RMSは0.324258秒（1.64%）でした。親演算とshader種別は同じ時間の別分類なので加算できません。

1024のAttention割合は256の約5%から約25%へ増えています。256の順位だけで大画像の優先度を決めず、1024の行列積とAttentionを対象にします。全値・件数は[集約JSON](moss-dit-gpu-profile-20261021.json)にあります。

## 全生成・CPU wall・数値の確認

| 記録 | 256 | 同じプロセスで256後の1024初回 |
|---|---:|---:|
| native generation_seconds | 69.118352秒、lazy load込み | 47.777175秒 |
| 前段の計測なし版 | 69.275742秒 | 47.760891秒 |
| sampled 2 forwardのCPU wall | 1.554019秒 | 20.108488秒 |
| 有効GPU timestamp区間 | 4,088 | 3,582 |
| 各完全forwardのtokens | 362 | 4,202 |
| 各完全forwardのgraph nodes | 2,390 | 2,390 |
| owned Direct data peak | 4,858,265,600B | 6,107,971,584B |
| 生成後owned Direct data | 4,725,997,568B | 4,749,066,240B |

この計測版を新しい速度改善として扱いません。温度・clock・cacheを統制した反復比較ではなく、timestampの挿入も実行スケジュールへ影響し得ます。速度基準は[計測なし版の比較](moss-dit-fusion-20261020.md)を維持します。

1024の既存CPU wall計測はDiT graph_total41.208481秒、other runners graph_total6.006801秒、DiTのsubmit driver呼出合計0.033984秒でした。fence wait40.313510秒にはGPU計算の待ち時間が含まれ、全量を削除できるCPU同期コストとは解釈できません。sampled CPU wallとGPU区間合計の差も、純粋な同期時間ではありません。

- 全learned推論はPS5。MOSS Q4_K、Qwen3.5-9B Q4_K_M、small VAE、合成rank144 **F32** adapter、NSFW0.7＋Turbo1.0、runtime倍率1を維持。4 Euler steps／CFG1／seed20261020、専用sigmas、二重shiftなしです。
- 完全forwardは42層、Q4 main nodeが各種42、LoRA down252/up252であることを実graphで確認して選別しました。各生成は4つの有限DiT出力、672 Q4 F32記録、F16累積0、LoRA504適用／skip0でした。
- 従来14数値ゲート、Turbo3、uploadゲート、融合16ケースを維持。追加のtimestamp起動ゲートもPS5実GPUで成功しました。availability・順序・範囲・overflow・完了fenceを検査し、取得失敗を有効計測として扱いません。
- 実queueのtimestampValidBits64、period10nsを取得しました。追加のCPU wait0・submission0。既存のGPU完了後だけquery結果のcacheを処理して読み戻します。
- query poolは131,072B、固定CPU計測領域293,592B。物理Direct10GiB、推論予算9.5GiB、pool外reserve1GiBを変更していません。API CPU/GPU-labelled値は重複するため加算できません。
- 両要求とも1POST・自動再送0。native GETとOpenAI互換APIのPNGを照合し、完全decode・目視確認を行いました。**両サイズのPNGは計測なし版と全バイト一致**しました。

![実機256。計測なし版と全バイト一致](images/moss-turbo4-upload-256-20261020.png)

![実機1024。計測なし版と全バイト一致](images/moss-turbo4-upload-1024-20261020.png)

測定後のソースでは、実pipeline名split_k_reduce/fa_split_k_reduceをreductionラベルへ追加しました。分類だけの修正で、host metadata1,638項目と設定拒否26件を確認しました。このラベル修正版は別のPS5計測をしていないため、表・JSONは実測時のother分類を維持しています。1024のother shaderは0.00425%、256はsplit-K縮約等を含め0.13661%でした。

## 次の改善対象

1. **FFN行列積**：up/gate＋downで約40%。現在のQ4_K→演算経路のtile・workgroup・load再利用・epilogueを調べ、F32累積とrank144 adapterの独立性を維持して比較する。
2. **Flash Attention**：1024で約25%。head128・tokens4,202の実形状に対するtile、shared memory、K/V layoutを優先。低精度Attentionへ変える場合は別の画質・数値試験が必要。
3. **QKVとLoRAのGEMM周辺**：QKV約15%、LoRA約6%。base QKVの結合は既に実装済み。独立q/k/v LoRAを崩さず、重複read/writeやdispatchを減らす余地を検討する。
4. **RMS・変換の追加融合**：ComfyUIの実装を参考にするが、今回の小さな割合を踏まえて優先度を下げる。RMS reductionの演算順変更は同一PNGを自動的に保証しない。

この調査では上記の新しい演算変更・近似cacheは採用していません。短縮秒数は未測定です。[ComfyUI、E-DiT、CacheDiTの適用評価](moss-dit-reference-evaluation-20261021.md)を併記しました。

## 再計測用ビルド

[Turbo4＋upload＋F32のビルド](build.md)にcoreの --moss-dit-optimizations、nativeの --moss-dit-optimization-mask 15 と **--moss-dit-gpu-profile** を明示します。旧50-step/rank16向け --moss-gpu-profile との同時指定は拒否します。default-OFF、最大4試行、production admission0、既存の容量・所有権・数値条件を保持します。

実装は[計測header](../native/app/moss_gpu_profile.h)、[専用policy](../native/app/moss_dit_gpu_profile_policy.h)、[Vulkan記録](../patches/stable-diffusion/overlay/ggml/src/ggml-vulkan/ggml-vulkan.cpp)、[ホスト分類試験](../tests/moss_dit_gpu_profile.py)です。ビルドpolicy／実recipe AST286項目、従来profileの59,801／59,796検査、拒否15件・source ordering17件も通過しました。これらのhost試験はGPU実行やlearned推論を代替しません。

## 測定後の状態

両生成後に通常APIでnativeとgatewayを終了し、独立した観測でprocess不在、同boot、既存サービス・タイトル登録・モデルmetadataの維持を確認しました。計測なしの検証済みmask15アプリを全byte照合して戻し、全数値ゲートを再確認し、native8192／OpenAI互換8193がジョブなし・モデル未読込の待機状態へ戻りました。合計の生成POSTは2件で、復帰後の追加生成はありません。FW・システム領域・Direct予約は変更していません。

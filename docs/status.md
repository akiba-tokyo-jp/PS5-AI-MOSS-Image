# MOSS Imageの状態

2026-10-20 JST、性能改善1〜3の実装・比較を完了しました。**現在残す実験profileはvariant2、FP32＋提出単位調整＋限定fusionです。private、production admission0、最大4試行、10GiB poolを維持します。** 最後のnative569/gateway572は通常API終了202後に両PID不在、独立observer011でidleを確認して停止しています。

| 今回の比較 | 計測入りFP32基準 | 提出調整＋限定fusion FP32 |
|---|---:|---:|
| 256初回、model読込込み | 153.972336秒 | 152.631656秒 |
| 同processの256後1024初回 | 1219.096190秒 | 1214.444805秒 |
| PNG | 各sizeで候補と全バイト・RGB一致 | 目視正常、GPU/finite/LoRA504確認 |

単回差は約0.87%／0.38%で、大幅高速化は未達、再現性や統計的な改善率は未確定です。FP16はfeature/env/logの有効化を確認しましたが、Q4_K N9数値ゲート41/153 mismatchで起動を拒否し、生成POST0でした。比較の任意warm繰り返しは実施していません。schema2のdriver計測はlookupを含むnested CPU wallで、GPU timestampや除去可能な待ち時間ではありません。[性能比較の詳細](moss-performance-optimization-20261020.md)。

## 前回の1024 cold/warm（履歴）

2026-10-20 JST。**privateの実験版です。PS5でMOSS＋指定LoRAの1024×1024・50stepsを2回生成し、WindowsからOpenAI互換APIで取得しました。両画像は正常な画質、同一PNGを確認しています。**

| 確認項目 | 現在の結果 |
|---|---|
| 実機構成 | FW12.40、PPSA99190、MOSS-Image-SFT Q4_K＋Qwen3.5-9B Q4_K_M＋FLUX.2 small VAE＋指定rank16 LoRA |
| GPU / CPU | RADV NAVI21でDiT/Flash Attention/VAE完了を観測、text encodingはPS5 CPU。VAE tile256/overlap0.5、49tile |
| 条件 | 1024×1024、50steps、Euler、CFG3、shift5、seed20261020、LoRA0.7。同prompt・同native/gateway/SELF |
| PS5生成時間 | 初回読込込み1242.706350秒／同process2回目1177.725365秒。差64.980985秒は純load時間ではない |
| API往復 / Windows全検証 | 初回1243.907秒／1249.407秒、2回目1178.875秒／1184.609秒 |
| 画像 / LoRA | RGB1024×1024 PNG2,496,997B、両回同一SHA、native GETとAPI返却bytes一致。LoRA各504/504/504、skip0 |
| 検証scope | 各phase1POST・自動再送0。coldはOS disk cache未flush、warmでもweight再転送ゼロとは主張しない |
| メモリ | trial10GiB pool、reserve1GiB。API CPU/GPU peaksは両回11,790,876,672B／4,868,833,280Bの累積上界。純VRAM値／各画像増分／加算可能値ではない |
| 終了 | native466/gateway469を通常API終了後に両PID不在を別確認。独立observer007でGPU titleなし |
| 継続する制限 | production admission0／private。要求12GiB＋reserve1GiBは未達。平均／分散、長時間連続、最小steps、他promptは未検証 |

[1024 cold/warmの詳細](moss-1024-cold-warm-20261020.md)。正常PNGは同一なので1枚だけ同梱し、生ログ・失敗画像・重み・秘密はGitに含めません。以前の[256×256・4対50steps比較](moss-native-small-trial-20261019.md)ではsampling不足による色縞を切り分けました。

<details>
<summary>初期ホスト構築時点の記録（実機試験前の履歴）</summary>

以下の未配備・未検証という記述とbuild hashは、初期host/native-v2の保存履歴です。上記のcorrected trial結果で更新された現在の実機状態とは区別してください。

2026-10-19 JST。**private開発版、指定構成はMOSS-Image-SFT＋axiomofmindのrank16 LoRAです。**
モデル取得・LoRA変換/接続・ホスト試験・fresh cross buildまで完了しました。
**MOSSのPS5配備・モデル読込・GPU推論・PNG生成は未実施です。**

| 段階 | 現在の証拠 |
|---|---|
| 正規モデル取得 | base4 component計9,277,433,929 bytesとraw LoRA74,377,208 bytesをDへ取得。全ファイルのSHA256読み戻し一致 |
| LoRA変換 | 全504 tensor・37,158,912値をexact F32化し、全値読み戻しと全体SHA256一致 |
| 実モデルとの接続 | 実GGUF headerの168 weightsに全504 shape一致、独立q/k/v順一致 |
| Source/API設定 | 固定F32パス・strength0.7・at_runtimeを接続。任意パスやSDXL漫画LoRAへ置換しない |
| ホスト試験 | 変換/数値6 tests、C++504受理/1013拒否、実upstream名前変換504、adapter構文、設定例110 checks。download制御18 testsもPASS |
| Full core | 75 sourceを全て新規compile、旧object流用0、4,135 header等と入力の前後hash一致 |
| lifecycle host試験 | 状態/ABI/first-failure境界47 checks＋HTTP mock19 checksがPASS。ready fixtureはhost mockで、実機終了成功ではない |
| Native SELF | integrity正常、strip前後loaded bytes一致、6動的依存、24 initializer slots、入力前後hash一致 |
| 実機 | 12GiB容量、Qwen CPU数値、LoRA適用、DiT/VAE GPU完了/有限値、実PNG、生成時間は未検証 |

新core archiveのSHA256は
`2184d88babd47ab8826b03c58f473de7e98adb19a79ab47798e97d576e2f56bc`、
native SELFは
`50131fb15447eec767e3f976ed19255870d2de1068512a81f86772128fdac30b`です。
coreの18 objectの警告は旧buildと対象が同じで、新規warning objectはありません。
LoRA receipt hook/snapshotは各1個のstrong定義を確認し、選定LoRAのApache-2.0をnative packageへ同梱しています。

## 実機へ進むための残件

- Title ID候補PPSA99190、native8192、gateway8193と専用namespaceを用意。WAI/Animaと同時起動しません。
- Direct予約目標12GiB、1GiB regions、推論予算11.5GiB、pool外working reserve1GiBを維持しています。
- 過去のDirect報告約11.44GiB/12GiBは12+1GiBの容量判定を通りません。物理枠が拡大できたとは扱わず、新しい実機条件での確認が必要です。
- Qwen embedding545.625MiBは旧512MiB単体上限を超えるため、sourceでは1GiB regionへ対応済みです。実割当・モデルロード・resident peakを確認した意味ではありません。
- 通常初期化はcompile-time admission0で明示的AGC/data pool初期化前に停止します。JSONで解除できません。ビルド成功だけで解除しません。
- lifecycleは初期化工程/失敗/ready/retainedを区別します。ready前・失敗時の終了要求は409で、別途確認した通常OSタイトル終了が必要です。自動killや手動GPU解放へ切り替えません。
- 受理済みPOSTはjob IDで追跡し、タイムアウトで再送しません。次の試験は前のジョブと所有者/通常終了を確認してからです。

## 証拠の所在

非公開のローカル証拠はD:/PS5-AI/moss以下の
`host-tests/requested-lora-v2/receipt.json`、`host-lifecycle-v1/run01/result.json`、`diagnostics/`、
`build/core-v2-lora-lifecycle/manifest.private.json`、
`build/native-v2-lora-lifecycle/verified-handoff.private.json`です。
core用181入力freezeの履歴と、その後のLICENSEコピーだけを分けたnative用213入力freezeを保存しました。
native builder自体も3,793入力の前後一致を確認しています。

[モデル/メモリ](moss-models.md)、[LoRA](moss-requested-lora.md)、[配置/lifecycle](deployment.md)、[ビルド](build.md)を参照してください。
旧[検証JSON](moss-verification.json)は今回のLoRA接続前のホスト検証履歴です。
継承したWAI/Animaの成功画像・ベンチマークはMOSSの実績ではありません。

</details>

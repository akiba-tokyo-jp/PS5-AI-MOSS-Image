# MOSS Imageの状態

2026-10-19 JST。**privateの実験版です。PS5で256×256・50stepsの正常な画像を生成し、WindowsからOpenAI互換APIでPNGを取得しました。** 同一SELF・同一全条件で4stepsを再現してからstepsだけ50へ変え、強い色縞が解消しました。この条件の4stepsのsampling不足が原因と判断します。

| 確認項目 | 現在の結果 |
|---|---|
| 実機構成 | FW12.40、PPSA99190、MOSS-Image-SFT Q4_K＋Qwen3.5-9B Q4_K_M＋FLUX.2 small VAE＋指定rank16 LoRA |
| GPU / CPU | RADV NAVI21でDiT/Flash Attention/VAE完了を観測、Qwen text encodingはPS5 CPU |
| 指定LoRA | strength0.7、loaded/matched/used各504、skip0、実適用確認 |
| 比較時間 | fresh processの256×256、Euler、CFG3、shift5、seed20261019。4steps 68.525607秒 → 50steps 148.133782秒（初回読込込み） |
| API / 画像 | 50stepsのHTTP200応答149.406秒、RGB256×256 PNG194,931B、native GETと応答bytes一致。各process POST1回・自動再送0 |
| 数値検証 | 6つの限定GPU読み戻しゲートがPASS。50stepsのDiT出力100・VAE出力1、textを含む3,854,336値でnonfinite0。全演算品質の普遍的証明ではない |
| メモリ | 承認済みtrial10GiB pool。native総量11.4375GiBのため要求12GiB＋reserve1GiBは未達。API GPUピークはCPU weightsも含む共有poolの上界 |
| 終了 | native432／gateway435を通常API終了後、別取得で両PID不在。独立observer005でGPU titleなし |
| 継続する制限 | production admission0／privateを維持。warm、高解像度、連続運用、最小必要stepsは未検証。通常APIの既定stepsは元から50 |

[実機結果・条件を固定した原因比較・メモリ・証拠の詳細](moss-native-small-trial-20261019.md)を参照してください。正常50stepsの小画像のみ同梱し、生ログ・失敗画像・重み・秘密はGitに含めません。

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

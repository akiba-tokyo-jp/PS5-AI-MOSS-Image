# MOSS実機の小画像試験と画質原因の比較（2026-10-19 JST）

**PS5でMOSS＋指定LoRAの256×256・50steps画像を正常な画質で生成し、WindowsがOpenAI Images互換APIから取得しました。PS5生成時間は初回モデル読込を含み148.133782秒です。**

最初の4steps画像にあった強い色縞は、同一SELF・同一設定でstepsだけ50へ増やした比較で解消しました。この条件では**4stepsのsampling不足が原因**と判断します。全prompt・解像度の品質保証や最小必要stepsの特定ではありません。privateの実験版を維持し、production runtime admissionは0のままです。

## 原因を特定した比較

両試験でnative SELF、モデル、LoRA、prompt、seed、sampler、CFG、flow shift、driver、pool、reserveを固定しました。各試験は新しいnative processで1回だけPOSTし、自動再送0。4stepsの後に通常終了と所有者不在を確認してから、同じSELFをfresh launchして50stepsを生成しました。

| 項目 | 4steps再現 | 50steps比較 |
|---|---:|---:|
| PS5 generation_seconds | 68.525607 | **148.133782** |
| 画像 | 強い細かな色縞を再現 | **色縞解消、今回の正常画質を確認** |
| PNG bytes | 196,962 | 194,931 |
| 4px周期のhighpass phase mean span | 0.11117244 | 0.00619342 |
| LoRA loaded / matched / used / skipped | 504 / 504 / 504 / 0 | 504 / 504 / 504 / 0 |

4steps再現画像のSHA256は元のcorrected4steps画像と完全に一致しました。追加したGPU数値ゲートと試験契約の変更が、元の4steps出力を変えていないことも確認しています。4px周期指標は94.43%減りました。これは目視比較を補助する記述的なpixel指標で、一般的な画質合格基準ではありません。

通常のnative／gateway／PowerShellクライアントの既定stepsは**元から50**です。4は初回smoke契約だけの値で、通常既定値を4から50へ修正したわけではありません。品質比較用profileは256×256・4または50stepsだけを明示的に許可し、1processにつき1件のatomic claimを維持します。旧4steps専用profileとproduction admission0は変更していません。

## 50stepsの実測条件と結果

| 項目 | 実測・固定条件 |
|---|---|
| 実機 | PS5 FW12.40、Title ID `PPSA99190`、RADV NAVI21 |
| エンジン | MOSS-Image-SFT Q4_K、Qwen3.5-9B Q4_K_M、FLUX.2 small VAE、外部Qwen tokenizer |
| 指定adapter | `axiomofmind/Moss-image-nsfw-uncensored-lora`、rank16をexact F32変換、runtime strength0.7 |
| 推論配置 | Qwen text encodingはPS5 CPU、DiTとVAEはPS5 Vulkan。WindowsはAPIクライアント |
| リクエスト | 256×256、50steps、Euler、CFG3、flow shift5、seed20261019、negative CFG skip layers `[22]` |
| Native job | `6ad6269b000001b00000000100000001`、`completed`、実GPU完了の観測あり |
| PS5生成時間 | **148.133782秒**。初回モデル読込を含む、このprocess最初の生成 |
| OpenAI互換応答 | HTTP200、149.406秒。Windows側の全処理150.859秒 |
| PNG | RGB、256×256、194,931 bytes、全decode成功。native GETとOpenAI応答のPNGはbytes一致 |
| PNG SHA256 | `8e14a3df612932dcef0bbdef38eb3a7cbafd964b87329e8c5a6c7bf55bc07a65` |
| LoRA適用 | loaded/matched/used各504、skipped0、`application_verified=true` |

![PS5 MOSS 256×256・50stepsの正常出力](assets/moss-256-50step-20261019.png)

warm生成、1024画像、連続生成、最小必要stepsは測定していません。148.1秒をモデル読込済みの時間として扱いません。

## GPUと有限値の証拠

実機でDiTの行列演算、Flash Attention、VAE decoder完了後の観測を取得しました。50steps逐次CFGのDiT出力100件、VAE出力1件、textを含む計3,854,336値を確認し、nonfinite0でした。限定した観測と有限値だけで、全GPU演算の数値正しさを証明するものではありません。

初期化時の6つのGPU読み戻しゲートもPASSしました。Q4_Kのvectorに加えN9 matrixとK3072/M3072/N304 matrix、Flash Attention、BF16 N7 vector／N9 matrixを検査しています。大matrixは99 edge/interior値を独立CPU double oracleで比較し、全出力の有限値を確認します。各ゲートでCPU fallbackなし、dispatch/submit/fence増分、mismatch0・nonfinite0を確認しました。BF16 N7は119値、N9は153値です。

driver表示は`fp16:0 / bf16:0`、FP16演算は無効です。BF16 storageをソフトウェア側の経路で扱えた限定試験であり、hardware BF16演算対応とは主張しません。

## メモリの意味と12GiB目標

人が許可した別の専用trialでDirect pool **10,737,418,240 bytes＝10GiB**を確保しました。pool外working reserve1GiB、推論予算`Vulkan0=9.5`、1region/claim上限1GiBです。native contextのDirect総量報告は**12,280,922,112 bytes＝11.4375GiB**で、要求されたproductionのpool12GiB＋reserve1GiBは容量判定を通りません。**12GiB目標は未達**です。

| 記録値 | bytes | 解釈 |
|---|---:|---|
| API `peak_cpu_bytes` | 11,769,872,384 | process peak RSS。独立したCPU専用メモリ量ではない |
| API `peak_gpu_bytes` | 4,867,227,648 | driver arenaとDirect data各owned poolピークの合計上界。ピーク時刻は同一とは限らない |
| Direct data peak | 4,835,262,464 | 約4.503GiB、CPU text weightsも含む共有poolのピーク |
| 生成後Direct data used | 3,536,896,000 | 約3.294GiB |
| CPU text weights peak | 4,835,246,080 | 同じDirect pool内。conditioning後のliveは0 |

APIのGPUピークは純粋なVRAM使用量ではなく、CPU/GPU値を加算できません。独立global GPU memory measurementは`unavailable`です。小画像の結果から最大解像度や長時間のメモリ余裕を推定しません。

## 修正履歴と正常終了

最初の試験は、BF16左辺の行列積17nodeをscratch admissionが未分類として拒否し、モデルGPU dispatch前で停止しました。BF16の行列左辺だけへtype/shape/stride/32-bit span検査を追加し、既存4B/elementの保守的scratch上界を維持しました。admission無効化やpool拡大はしていません。実42-layer graphのhost metadata試験と独立上界検証を経て、core75 sourceを新規compile、旧object流用0で再構築しました。

corrected初回4stepsは68.704665秒でGPU PNG/APIの技術的成功を確認しましたが、色縞のため品質は未合格でした。その後の上記4対50比較で、今回の画質原因を切り分けました。offlineの一次source照合ではMOSS本体、FLOW/shift5、FLUX.2 VAEのBN丸め、2×2 unpack、DiT軸順の明確な不一致は特定していません。これはそれらの普遍的な無欠陥証明ではありません。

品質比較SELFのSHA256は`cf61a7190e4e0f79b7aedb12038fdeb1d529dd90cf92a5a26185ce73cf3cabe5`です。core archiveは`c9809e6bbb04ef3a09044d7c5036e113004c37bc616541ca5d808d4c53acaab4`を維持しました。新しい品質契約と数値ゲートはnative側だけの変更で、全75 compiler dependency記録を監査し、coreの消費入力が変わらないことを別receiptで確認しています。旧core manifestのhashを書き換えていません。

50steps後、native PID432／gateway PID435を通常のidle API終了で閉じました。受理202だけを証拠にせず、別process取得で両PID不在、独立observer005でGPU titleなしを確認しました。追加生成や録画はしていません。

## ローカル証拠

重み・秘密・生ログ・失敗画像はGitへ含めません。上の正常50steps出力PNGだけをユーザーの許可範囲で同梱しています。以下は非公開Dドライブ上の所在です。録画やログのクラウド投稿は行っていません。

- 同一条件比較: `D:/PS5-AI/moss/diagnostics/quality-pair-20261019/comparison.private.json`
- 4steps再現: `D:/PS5-AI/moss/quality-trial-v1/baseline4-actual-001/`
- 50steps結果／PNG／応答: `D:/PS5-AI/moss/quality-trial-v1/steps50-actual-001/`
- 現在のengine/startup等ログ: `D:/PS5-AI/moss/native-logcopy/v4/execution-001/`
- 通常終了／PID不在: `D:/PS5-AI/moss/diagnostics/quality50-normal-exit-v3/`
- 独立idle確認: `D:/PS5-AI/moss/native-observer/v1/extracted-005/session.private.json`
- core: `D:/PS5-AI/moss/build/core-v4-bf16-scratch/manifest.private.json`
- 品質比較native: `D:/PS5-AI/moss/build/native-v5-quality50/verified-handoff.private.json`
- 初期corrected4steps履歴: `D:/PS5-AI/moss/smoke-corrected-v2/actual-001/`

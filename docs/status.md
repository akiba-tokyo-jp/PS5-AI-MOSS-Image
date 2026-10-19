# MOSS Imageの状態

2026-10-19 JST。private開発版。指定構成をMOSS-Image-SFT＋axiomofmindのLoRAへ更新。
LoRAは70.93MiBの追加重みで、F32重み換算141.75MiB。q/k/v→qkvの対応、BF16処理、native/API接続は残件です。[調査](moss-requested-lora.md)。既存のビルド成功はLoRA適用成功を意味しません。**PS5への転送・登録・MOSS推論は未実施**です。

- MOSS専用Title ID候補PPSA99190、native8192、OpenAI互換gateway8193を用意。
- Direct予約目標12GiB、1GiB regions、推論予算11.5GiB、pool外working reserve1GiBを設定。
- 過去のGPU稼働時Direct総量約11.44GiB、別のAGC前診断12GiBは、どちらも12+1GiBの判定を通らないため、実機で容量の再確認が必要。物理枠が拡大できたとは扱わない。
- Q4_K DiT、Qwen3.5-9B Q4_K_M、small FLUX.2 VAE、外部tokenizerの取得revision/size/期待SHA/ライセンスを固定。実モデル全体のdownload/検証は未実施。
- Qwenの実GGUF headerからembedding545.625MiB、output head795.703MiBを確認。旧512MiB制限を超えるためMOSS allocatorは1GiB単位へ準備。nativeロードとresident peakは未計測。
- SD.cppのMOSS/Qwen3.5対応をPS5向け基準版へ範囲限定backport。全75-member coreのfresh rebuild経路を追加。
- OpenAI互換API host protocol133項目、モデルdownload制御18項目はhost上でPASS。実GPU生成の証拠ではない。
- gatewayと完全75-member coreのcross compileに成功。coreは旧object流用0、75source/4131header等の前後hash一致。89件の警告を監査し、新MOSSコードの型・精度・UB警告は見つからなかった。
- engine host検査は設定57、CPU貸出44、Direct pool mock35、実GGML metadata21でPASS。core受入・拒否の5検査もPASS。いずれも実機の容量・数値・生成検証ではない。
- native SELFはビルド・リンク・署名・integrity検査PASS。strip前後のloaded bytes一致、6動的依存、24initializer slots、必要なoptimization symbol定義を確認。PS5未転送・未起動。
- 未検証の実容量・operator・数値を根拠なく飛ばさないよう、通常MOSSモデル初期化はcompile-time admission0で停止。

[検証JSON](moss-verification.json)、[モデル詳細](moss-models.md)、[ビルド](build.md)を参照してください。ここ以外のWAI/Animaの過去の成功結果はMOSSの結果ではありません。

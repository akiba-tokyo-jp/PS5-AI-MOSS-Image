# PS5 AI Anima 開発状況

2026-10-19。既存WAIから独立した新アプリとして構築中です。Animaの実機画像生成・速度・品質は未確認です。

- 出発点：PS5 AI WAI commit `86393645bb1f663c78f2cd2ac08ad67fa84e3812`。
- SD.cpp pin `8cd731e7b4667eb8f40a153315b9a7125d09a58d`はAnimaのDiT・Qwen conditionerを実装済みです。
- 対象モデル：公式Anima Turbo v1.1、Qwen3-0.6B-Base、Qwen-Image VAE。
- 公式HF revision：`f973fc41ec7545364ac9776c2440285f43ff2a30`。
- 新アプリ候補PPSA99189、native8190、OpenAI gateway8191。WAIの8188／8189と分離します。
- WAIと同時起動しません。通常終了・待機確認を伴うアプリ切替を使います。
- まず512×512・8steps・CFG1・Eulerのtext2imgを検証します。256は公式の推奨解像度範囲外です。
- SDXL用LoRA・IP-Adapter・ControlNetをAnimaに流用しません。
- 公式モデル3ファイルの全内容ハッシュを検証済みです。GPU用DiT/VAEのF16保存形式への変換・全値検証・読み戻しも完了しました。[変換記録](anima-model-derivatives.json)。
- ホストAPIの90検査、エンジン設定42検査、CPU共有Direct借用39検査が成功しています。実機モデル推論を行う検査ではありません。
- Anima固有のRMSNorm・RoPE・3D畳み込みをscratch admissionへ追加し、実際のGGML constructorでも形状を確認しています。GPU計算結果の検証は未実施です。
- native SELFとPS5用gatewayのcross buildが成功しました。SELF署名整合性、strip前後4つのPT_LOAD一致、6つのruntime依存、最終ソース／assetの一致を独立確認しています。[ビルドと検証記録](anima-verification.json)。
- 新規タイトルの登録は未実施です。保護領域に必要なメタデータ／mountの範囲を確認した上で扱います。

既存のWAI成功資料は基盤の履歴であり、Animaの実機成功証明ではありません。Animaで取得したログ・PNGだけを、今後の成功として記録します。[元WAIの状況](wai-base-status.md)。

# Animaビルド

Anima実機推論は未検証です。以下はローカルのcross compile手順です。[WAI由来の詳細な依存構築資料](wai-base-build.md)を保持していますが、その実機結果はAnimaの実測ではありません。

## 入力

- Windows用LLVM/MinGW、PS5 public native SDKとpayload SDK。
- [dependencies.lock.json](../dependencies.lock.json)に固定したSD.cpp・GGML・RADV・PS5 platformのソース。
- [build-profiles/native.json](../build-profiles/native.json)にサイズ・SHAを固定した、レビュー済み依存アーカイブと生成shader header。
- ローカルパスは[config/build.json.example](../config/build.json.example)から自分の環境用設定へ複製します。SDK・重み・秘密情報はGitへ含めません。

`tools/build.py`はアプリ、platform、変更したcore/GGML providerを再コンパイルし、依存アーカイブの指定memberだけを更新します。既存レビュー済みライブラリが必要で、クリーンなPCで依存すべてを自動取得するワンコマンドビルドではありません。元の依存構築方法は上記WAI資料を参照してください。

```powershell
python -B tools/build.py --config D:/ps5-dev/anima-build.json --out D:/ps5-dev/build/anima-native --precision fp32
python -B tools/build_gateway.py --config D:/ps5-dev/anima-build.json --out D:/ps5-dev/build/anima-gateway --native
```

native出力は`title/PPSA99189/eboot.bin`、gatewayは`ps5-ai-openai-gateway.elf`です。SELF署名・整合性検査とソースhashはbuild manifestへ記録します。ビルド処理はPS5へ接続しません。

`--precision fp32`は演算精度の指定です。Anima/VAEのF16保存重みを使うこととは別です。Qwen3は公式BF16のままCPUで扱います。Qwenの311,164,928-byte embeddingは、実際のowned Direct region上限を確認して借用し、通常の512MiB regionへ収めます。固定256MiBだったWAI版からAnima側だけ変更しています。

## モデル準備

```powershell
python -B tools/download_anima.py --output D:/ps5-dev/anima-models/official
python -B tools/models/convert_anima_bf16_f16.py --official-dir D:/ps5-dev/anima-models/official --derived-dir D:/ps5-dev/anima-models/derived --diagnostics-dir D:/ps5-dev/anima-models/conversion
```

公式revision/ハッシュは[モデル資料](anima-models.md)にあります。変換器にはNumPyが必要です。DiTとVAEのF16変換は重みデータの変換のみで、Windowsで画像推論を実行するものではありません。変換したファイルのhashは公式BF16のhashと別に記録します。全BF16符号化の丸め照合・全値の有限性と範囲・書込後の全内容読み戻しを検証します。中断したpartは期待値と照合できたバイトだけ再利用します。

## 検証の範囲

現行用のホスト検証は`tests/anima_engine_config.py`、`tests/anima_api_protocol.py`です。失敗fixtureやHTTP mockはprotocol専用であり、実モデルによる生成結果ではありません。旧WAI向けtestsは履歴として保持し、Animaの機能合格判定へ混在させません。

アプリ登録・モデル転送・実機実行はビルドとは別工程です。[配置](deployment.md)、[現在の検証結果](anima-verification.json)を確認してください。

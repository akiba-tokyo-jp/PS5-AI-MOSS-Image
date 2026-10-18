# ビルドと再現性

最新のnormal/debug・容量制限版も公開用ソースから再ビルドし、SELF整合性を確認しました。実機結果と公開別ビルドの区別は[ログ検証](logging.md)を参照。

## 今回検証した範囲

公開フォルダーのネイティブ11翻訳単位とRADVプラットフォーム1翻訳単位を再コンパイルし、
SHA256固定の既存SD/GGML/RADV/SDK依存とリンク、ELF→ネイティブELF→SELF変換・検査まで確認しました。
Windows側のAPI試験は推論を行わないfixtureです。PS5での画像生成の証拠とは区別します。

**依存ライブラリを含むクリーン環境からの一括ビルド、初回インストールの自動化は未完了です。**
このリポジトリは実機成功版のソース、対応する変更箇所、ビルドプロファイルを公開します。
依存キャッシュのない環境で、clone直後に1コマンドで実行ファイルができる状態ではありません。
モデル、SDK、生成済みシェーダー、ライブラリやlibc.prxはGitに含めません。

## ツールとソース

検証環境はWindows x64、Python3.13系、llvm-mingw20261006、MSYS2 CLANG64のLLVM23系です。
native SDKは公開SDK v0.42にPS5_PayloadSDK `b83202be…`のheaders/platformを組み合わせたもの、
CPUゲートウェイは別の公開SDK v0.44を使います。この2つのCRT・ヘッダーを混ぜません。
Vulkan-Headersは `c46850864f4661461b0f6cb9922c058ffea4915e` を使用しました。

依存のURL・revision・SDKアーカイブSHAは[lock](../dependencies.lock.json)にあります。
十分な空きのある作業ドライブを指定してください。

```powershell
python tools/prepare_sources.py --out D:/ps5-dev/sources
```

既存の変更済みcheckoutは上書きしません。2回目は別の出力先か、変更を保存したcheckoutを使ってください。

## 依存ライブラリの構築元

| 入力 | 構築元・変更 |
|---|---|
| PS5 SDK / libps5platform | 公開v0.42 SDKに固定forkのinclude/freebsd、include/khronos、include/ps5、platform headersを配置し、platform/src・regex・dlmalloc/mspaceをPS5ターゲットで静的ライブラリ化 |
| RADV | 固定PS5_Mesa、ACO、PS5 winsys、LLVM target無効。Mesa hostのmesa_clc/vtn_bindgen2はWindowsで生成し、targetライブラリと混ぜない |
| RADV変更 | `patches/mesa/overlay`の物理メモリ・import heap bookkeeping。最終winsysプラットフォームは`native/app/radv_ps5_platform_native.c`に置換 |
| stable-diffusion / GGML | 固定SD.cpp、Vulkan有効、動的backend/OpenMP/native CPU variants無効。`patches/stable-diffusion/overlay`を適用 |
| SD/GGML変更 | Direct host import、graph cut、disk weight retirement、loader poll、scratch/graph admission。core runnerとallocatorの内部ABIを揃える |
| Native CRT / SELF tools | 固定PS5_Vulkanの`tooling/native`。host C++20 + zlibでpackagerをビルド |
| 互換関数 / AGC stubs | `native/link/compat.c`と公開スタブソース。スタブはリンク用で、実GPU実装ではない |
| libc.prx | PS5_Vulkanの公開clean-room native libc作成手順。検証済みSDK/runtime構成と揃える |

targetは `x86_64-sie-ps5`、FreeBSD/x86-64、16KiBページ、emulated TLS、IEEE denormal設定を使います。
Mesaは`radv-winsys=ps5`、`vulkan-drivers=amd`、`llvm=disabled`、`amd-use-llvm=false`、`buildtype=debugoptimized`、`b_ndebug=true`です。
host shader compilerのWindowsテキスト変換を避け、生成SPIR-Vを検査してください。
現在のソース公開では、これら依存生成を統合した汎用スクリプトはまだ提供していません。

SD側はAVX/AVX2/FMA/F16C/BMI2有効、AVX512無効、C++20・公開libc++・exceptions有効です。
`GGML_MAX_NAME=160`、`sizeof(ggml_tensor)=432`を変えるとネイティブ側とABI不一致になります。
近似VAE/Flash Attention、batched CFG、model residency設定も[config](../config/native.json.example)を基準にします。

## レビュー済み依存を使うネイティブ再ビルド

1. `config/build.json.example`を非共有の`.local/build.json`などへコピーし、各パスを設定します。
2. `DEPS`に[build-profiles/native.json](../build-profiles/native.json)の名前・SHAに一致するライブラリを用意します。
3. native SDK、LLVM、Vulkan-Headers、固定revisionの上流source、host packagerを指定します。
4. 新しい出力フォルダーで実行します。

```powershell
python tools/build.py --config .local/build.json --precision fp32 --out D:/ps5-dev/build/wai-native
```

過去の個人用ログ・PID・セッションreceiptは不要です。ソースをコンパイルし直し、driver内のplatform memberを置換してリンクします。
既定は実測で速かったFP32演算版です。FP16試験版は`--precision fp16`で作ります。
FP16版は`GGML_VK_DISABLE_F16`を解除し、実際の起動ログと物理機能を検査します。
直接畳み込みは独立したCPU正解値とのGPU読み戻し照合に合格した場合だけ使います。
core runnerも公開overlayから再コンパイルし、固定の既存SD依存アーカイブ内の対応1メンバーを置き換えます。
strip前後のPT_LOADが同じこと、native runtime providerが6件であることを検査します。
完成済み出力を上書きしません。依存のSHAが違う場合は停止します。
別ツールチェーンで作ったライブラリのSHA不一致を、単に期待値を書き換えて成功扱いしないでください。
その場合は依存生成とABI・constructor・importの再検証が必要です。

出力は`title/PPSA99188/eboot.bin`とアイコン・paramです。
runtime libc、licenses、設定、モデルを含むインストール済みタイトル全体を自動生成・配備するコマンドではありません。
ファイルの存在だけではPS5起動・GPU実行を証明しません。[配置前提](deployment.md)を確認してください。

## OpenAIアダプターとホスト試験

```powershell
python tools/build_gateway.py --config .local/build.json --out D:/ps5-dev/build/gateway
# PAYLOAD_SDK と SDK_SOURCE を追加した設定で、PS5 ELFもビルド
python tools/build_gateway.py --config .local/build.json --out D:/ps5-dev/build/gateway --native
```

ゲートウェイのpayload CRTはSDK標準の自身のsyscall-range設定を保持し、credential変更のhelperを省きます。
この起動部分は完全なno-opではありません。GPUネイティブアプリのCRTと取り替えないでください。
ビルドはPS5へ接続しません。

```powershell
$env:WAI_TEST_OUT = 'D:/ps5-dev/build/gateway'
$env:WAI_TEST_LLVM = 'D:/ps5-dev/llvm-mingw/bin'
python tests/openai_gateway_protocol.py
python tests/native_log_storage.py
python tests/native_precision.py --config .local/build.json --out D:/ps5-dev/build/precision-host
```

試験はHTTP認証、未対応パラメーター、同時要求、backend failure、HMAC/base64、ログrotation/容量不足・retained状態通知を確認します。
生成画像を返さないホストfixtureを使い、GPU推論成功の代わりには扱いません。

## ログ方針のホスト検査

以下はCPUの模擬検査で、モデル推論や実GPU呼び出しを行いません。出力先をD等へ指定します。

```powershell
$env:WAI_TEST_LLVM = 'D:/ps5-dev/llvm/bin'
$env:WAI_TEST_SD = 'D:/ps5-dev/sources/stable-diffusion'
$env:WAI_TEST_LOG_OUT = 'D:/ps5-dev/tests/log-policy'
$env:WAI_TEST_STDIO_OUT = 'D:/ps5-dev/tests/log-stdio'
python tests/native_log_policy.py
python tests/native_log_stdio.py
```

実際のpolicy parserとdriver loggerを使い、normal時の整形省略、debug時のまとめ同期、
critical/checkpointの同期、fsync失敗のsticky状態を確認します。
API用ヘッダーは連続ローテーションとrename失敗時の上限維持を確認します。

## 漫画LoRA版の追加モデル

この版は7GiB Direct / Vulkan予算6.75GiBと、WAI v17専用変換LoRAを使います。元WAI・原作者imgeのGLoRAは別途取得し、作者の利用条件・クレジットを保持してください。重みはGitHubに同梱しません。

```powershell
python -m pip install numpy
python tools/prepare_manga_lora_v1.py --base D:/PS5-AI/models/waiIllustriousSDXL_v170.safetensors --directory D:/PS5-AI/models/loras
```

これはWindowsでのモデル形式変換で、画像生成ではありません。固定mirror commitと元配布SHAを確認し、WAIの完全なSHAを変換metadataへ保存します。旧GLoRAの補正をrank48/F32へ変換し、使用1008層のサンプル補正を検査します。SVDやrank切詰めは行いません。モデルごとの変換なので別checkpointに使い回しません。clip_skip=2を前提とします。

出力`manga-illustrious-wai-v17-r48-f32.safetensors`をタイトルのmodelsディレクトリへ置き、更新した`config/native.json.example`を使います。既存6GiB版のconfigだけを書き換えて動くものではなく、nativeとアプリ内RADVのVA区切りを一緒に再ビルドします。通常OS終了・原本退避・ハッシュ照合後に配備してください。

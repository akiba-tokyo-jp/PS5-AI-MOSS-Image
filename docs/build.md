# MOSS Imageのビルド

Windows x64からPS5向けにcross compileします。ビルドツールはPS5へ接続・配備・推論しません。モデル、SDK、libc.prx、生成済みshader、依存libraryはGitに含めず、固定revisionとhashを別途揃えます。clone直後に依存を含めて完成するワンコマンド構築ではありません。

## 入力とABI

[dependencies.lock.json](../dependencies.lock.json)、[native依存profile](../build-profiles/native.json)、[core profile](../build-profiles/moss-core.json)が固定入力です。[build設定例](../config/build.json.example)を自分用の非公開JSONへコピーし、SDK／LLVM／Vulkan-Headers／依存archive／upstream checkout／native packagerを指定します。新しいbuild、vendor、一時ファイルは十分な容量のある作業ドライブへ置きます。

| 入力 | 役割と境界 |
|---|---|
| native SDK／PS5_PayloadSDK | 固定headers、platform、thread／TLS／memory ABI。CPU gateway用SDKと混在させない |
| PS5_Mesa／RADV | PS5 winsys、NIR／ACO、Vulkan runtime。host shader compilerとPS5 target libraryを分離 |
| SD.cpp／GGML | 固定baseと限定MOSS backport、[公開overlay](../patches/stable-diffusion/overlay/src/model/diffusion/moss_image.hpp)。75 core objectsをfresh compile |
| Native CRT／SELF tools | 固定native tooling。loaded bytes、SELF integrity、initializer／runtime providerを確認 |
| GLSL／SPIR-V header | reviewed shader metadata／hashと一致させる |
| licenses | アプリ、SDK、driver、font、モデルの条件を分離して保持 |

PS5 targetはx86-64、16KiB page、emulated TLSです。`GGML_MAX_NAME=160`、tensor layout、exceptions／RTTI、constructor／importのABIを一致させます。設定だけの変更で物理quotaは増えません。

[source-freeze例](../config/moss-source-freeze.json.example)を基に、実際に消費する全source、header、profile、helperのsize／SHAを記録します。例はfrozen:falseの未設定テンプレートです。完全な入力一覧を生成して固定した後にfrozen:trueを設定してください。公開用source inventoryと、compile時に消費した依存closureの証拠は別です。core manifestには全75 compile、depfile、archive対応と前後hashが必要です。

## 選択した速度基準のビルド

以下はTurbo4＋bounded upload、rank144 F32の**限定performance trial**です。Direct予約10GiB、推論予算9.5GiB、pool外reserve1GiBに一致する設定を使います。新しい出力先を指定し、既存の成功版を上書きしません。

```powershell
python -B tools/build_moss_core.py --config D:/ps5-dev/moss-build.json --out D:/ps5-dev/build/moss-core-turbo-upload --source-freeze D:/ps5-dev/moss-source-freeze.json --jobs 2 --moss-f32-accumulation --moss-turbo-trial --moss-bounded-weight-upload-trial
python -B tools/build.py --config D:/ps5-dev/moss-build.json --core-manifest D:/ps5-dev/build/moss-core-turbo-upload/manifest.private.json --out D:/ps5-dev/build/moss-native-turbo-upload --experimental-small-trial --trial-10g --performance-trial-variant 1 --precision fp16 --moss-f32-accumulation --moss-turbo-trial --moss-bounded-weight-upload-trial
python -B tools/build_gateway.py --config D:/ps5-dev/moss-build.json --out D:/ps5-dev/build/moss-gateway-turbo --native --moss-turbo-trial
```

これらのCLIは、公開基準へ戻した検証済みソースの引数です。nativeにpool sizeを任意指定する引数や、無制限の通常運用flagを追加しません。`--trial-10g`が選ぶcompiled placement、VA、設定、実機の確保結果を一致させます。12GiBの旧設定例をそのまま10GiB trialへ使わないでください。

nativeは`title/PPSA99190/eboot.bin`、gatewayはCPU用ELFを作ります。モデルやruntime全体を自動インストールするコマンドではありません。gatewayはHTTP転送だけを行い、GPU uploadの推論flagを有効にしません。core／nativeのTurbo、upload、F32累積のpolicyと入力hashを一致させます。

## 試行上限と任意実験の区別

このperformance buildは1native process最大4回の有効な試行に制限します。失敗も回数を消費し、5回目を拒否します。公開する成功条件は[APIのcanonical request](api.md)の固定prompt／negative prompt、seed20261020、4 Euler steps、CFG1、shift5、256または1024の正方形です。256成功・画像確認・空queue確認後に1024へ進みます。production runtime admissionは無効です。

FFN2048分割とF16 LoRA storage／専用F32 shared演算は別の明示実験で、速度基準ではOFFです。比較結果は[最新資料](moss-turbo-memory-options-20261020.md)へ分けています。8GiBや任意prompt／seedの無制限operating版は、この公開版の成功機能として含めません。

ホストpolicy／HTTP／graph／fixture検査はモデル推論を行いません。fresh75 core、native recipes、strong provider、SELFと依存検査が通っても、PS5の数値ゲート、実pool確保、生成画像、正常終了を代替しません。[配置手順](deployment.md)、[モデル再現](moss-turbo4-reproduction.md)、[第三者通知](../THIRD_PARTY_NOTICES.md)も確認してください。

## 追加DiT最適化の限定trial

[実機比較](moss-dit-fusion-20261020.md)の4項目を有効にする場合、上のcoreコマンドに --moss-dit-optimizations、nativeコマンドに --moss-dit-optimization-mask 15 を追加します。新しいoutを使い、source-freezeにも tools/moss_dit_policy.py のfields(True)に対応するpolicy・compile flag・全sourceの実hashを固定します。coreは全75コマンドでcapability1、nativeも同じcapabilityを必要とします。mask0は同じcoreの対照版で、1／3／7は累積した限定比較用です。flags未指定はOFFです。

F32積算・Turbo4・bounded uploadを必須とし、FFN分割・private LoRA precision・Spectrum・別のsampling／GPU profile候補との混在を拒否します。一般fusion、試行上限、production admission、容量・数値・完了・所有権の条件は維持します。gatewayのAPIと設定は変更しません。生成ソースを更新するgateway build等はsource-freeze前に済ませ、core／native build中にソースを変更しないでください。

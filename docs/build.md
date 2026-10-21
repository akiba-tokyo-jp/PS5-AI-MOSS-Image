# MOSS Imageをソースからビルドする

通常API版の入口は`tools/build_service.py`です。固定prompt・seed・256先行・最大4試行を撤去したservice profileをnativeとgatewayの両方へ適用します。[実機検証](moss-service-20261021.md)。過去の資料にある低水準trial用コマンドは診断buildの記録です。

## 依存と設定

Windows x64でcross compileします。[dependencies.lock.json](../dependencies.lock.json)、[native profile](../build-profiles/native.json)、[core profile](../build-profiles/moss-core.json)の固定revisionを揃え、[build設定例](../config/build.json.example)をprivate作業ディレクトリへコピーします。SDK、LLVM、PS5 Mesa/RADV、Vulkan-Headers、SD.cpp、shader toolchain、dependency archives、native packagerの実際の場所を指定します。

SDK・library・weights・生成済みshaderはGitに含みません。clone直後にそれらまで自動取得するinstallerではありません。nativeとCPU gatewayのSDK ABIを混在させず、16KiB page、emulated TLS、`GGML_MAX_NAME=160`、constructor/importを一致させます。公開overlayと固定baseを使用し、既存core archiveの名前だけを流用しません。[モデル準備](moss-models.md)、[adapter再現](moss-turbo4-reproduction.md)、[第三者通知](../THIRD_PARTY_NOTICES.md)。

## 通常版

新しい出力先を指定します。依存configとbuild生成物をソースcheckout外の容量に余裕があるドライブへ置いてください。

```powershell
python -B tools/build_service.py --config D:/ps5-dev/build.json --out D:/ps5-dev/build/moss-service --jobs 2
```

wrapperはgatewayをbuildしてschemaを生成した後、checkout内のGit管理対象・非ignoredファイルの入力一覧をprivate freezeへ記録し、75 core objectsをfresh compileしてnativeをbuildします。低水準のprofile名にtrialという語が残っていても、`--service-mode`で生成回数制限・256条件は無効です。

選択する構成はF32 accumulation、Turbo4、bounded weight upload、DiT optimization mask15、kernel mask15、FA shared-KV occupancy調整、img2img、10GiB Direct pool／9.5GiB推論予算です。FFN chunk2048、F16 LoRA storage、GPU profiler、別sampling候補は混ぜません。モデルはrank144 F32 adapter、runtime multiplier1です。数値・容量・GPU完了・owner guardは維持します。

生成物は`native/title/PPSA99190/eboot.bin`、gateway build directory内のCPU ELF、各manifest、`moss-service.json.example`です。exact ELF pathはgateway manifestの`artifact.path`を参照します。これはビルドであり、PS5へは接続・配備・起動しません。

既存の完全な[freeze形式](../config/moss-source-freeze.json.example)を自分で管理する場合は`--source-freeze PRIVATE_JSON`を指定できます。build中にsourceを変更しないでください。public source snapshotと実際に消費したdependency closureの記録は別です。

`--prepare-only`はgatewayをbuildし、freezeと75-object core planを検査して終了します。nativeやcoreのcompile成功・実機成功を意味しません。今回の実機artifactはwrapperの下位build commandsで構築し、75 compile、native24 constructors／6 runtime providers、全量SELF readbackを確認しました。後から追加したwrapper自動freeze経路はprepare-onlyまで確認しています。

## 配備と検証

通常版設定は[config/moss-service.json.example](../config/moss-service.json.example)です。旧`moss-native.json.example`は過去の12GiB／rank16構成であり、通常版へそのまま使わないでください。設定JSONのみでcompiled pool/VAを変えることはできません。

[配置・起動手順](deployment.md)に従い、native Homebrewタイトルとして起動します。数値読み戻し後にAPI ready、最初の生成時にモデルをlazy loadします。初回1024をそのまま指定できます。host policy/HTTP試験やcompile成功はPS5 GPU実行の代わりにはなりません。[今回のPNGと計測](moss-service-20261021.md)を参照してください。

過去のdefault-OFF診断は独立して保持しています。[DiT kernel比較](moss-dit-kernels-20261021.md)、[GPU時間測定](moss-dit-gpu-profile-20261021.md)、[メモリ案の比較](moss-turbo-memory-options-20261020.md)の固定要求と試行上限を、通常版へ持ち込まないでください。

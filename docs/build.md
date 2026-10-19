# MOSS Imageのローカルビルド

Windowsでのcross compileです。PS5への接続・転送・登録・推論は行いません。依存の詳細は[WAI由来のビルド資料](wai-base-build.md)を保持していますが、過去の実機結果はMOSSの結果ではありません。

## 入力と完全core再構築

[dependencies.lock.json](../dependencies.lock.json)はPS5向けSD.cppの基準revision、GGML、MOSS追加commitとRADV等を固定します。[native profile](../build-profiles/native.json)は既存レビュー済み依存アーカイブ・shader headerを固定します。[設定例](../config/build.json.example)をDドライブ上の自分用設定へ複製してください。Cドライブの既存SDK/ツールを読み取り利用できますが、新しいbuild/vendor/tempはDへ置きます。

MOSSの型定義は広く参照されるため、旧75-member coreのうち7memberだけを差し替える経路は使用しません。`tools/build_moss_core.py`は固定baseへ範囲限定MOSS backportとPS5 overlayを適用し、C/C++の全75memberを新規コンパイルします。source/header hashの前後一致、archive member集合とSHAを記録します。`tools/build.py`はそのmanifestを必須にし、古いcore・変更後のヘッダー・不一致archiveを拒否します。

```powershell
python -B tools/build_moss_core.py --config D:/ps5-dev/moss-build.json --out D:/ps5-dev/build/moss-core --source-freeze D:/ps5-dev/moss-source-freeze.json --jobs 2
python -B tools/build.py --config D:/ps5-dev/moss-build.json --core-manifest D:/ps5-dev/build/moss-core/manifest.private.json --out D:/ps5-dev/build/moss-native --precision fp32
python -B tools/build_gateway.py --config D:/ps5-dev/moss-build.json --out D:/ps5-dev/build/moss-gateway --native
```

[source-freeze例](../config/moss-source-freeze.json.example)をD側の個人用JSONへコピーし、`repository_root`と`private_vendor_root`の2パスを実際のcloneへ合わせます。例の古い21入力を現行LoRA/lifecycleの完全freezeとして使い回しません。ソース改修後はレビュー・host確認を行い、実際の全対象入力のsize/SHAを含む新しいfreezeを作成します。初期LoRA/lifecycle buildのcore用181入力freezeは履歴です。今回の性能試験は新しい132入力freezeを作成し、source/header/overlayの前後hashも別途確認しました。core builderの引数は同toolの`--help`を参照してください。SDK・shader生成物・レビュー済み依存アーカイブが別途必要で、クリーンPCからのワンコマンド依存構築ではありません。

nativeは`title/PPSA99190/eboot.bin`、gatewayは`ps5-ai-moss-openai-gateway.elf`を出力します。SELF整合性・loaded bytes・動的依存・ソースhashを検証します。`fp32`は演算設定であり、GGUFのQ4保存形式とは別です。

## 2026-10-20の性能試験専用ビルド

新しい3profileは`moss-performance-baseline-fp32.json`、`moss-performance-fp16.json`、`moss-performance-submit-fusion-fp32.json`です。次のように明示的な試験flagを指定します。

```powershell
python -B tools/build.py --config D:/ps5-dev/moss-build.json --core-manifest D:/ps5-dev/build/moss-core/manifest.private.json --out D:/ps5-dev/build/moss-performance-baseline --experimental-small-trial --trial-10g --performance-trial-variant 0 --precision fp32
```

variant0はFP32基準、1は`--precision fp16`、2は`--precision fp32`の提出単位調整＋限定fusionです。variant/precision不一致、固定契約の欠落・追加・値/型の不一致は拒否します。最大4試行、256/1024・50stepsなどの固定条件で、production admission0の通常ビルドを解除するflagではありません。

今回の共通coreは全75memberをfresh compileし、3nativeへ同じarchiveを結合しました。各nativeのGGML providerを新規compileし、3構成のcore/GGML/RADV archiveとshader headerがbyte同一であること、initializer24/provider6、input前後一致を確認しました。性能試験を理由に既存coreのsource/header検査を緩めていません。

実機比較ではFP16が数値ゲートを通らず、variant2のFP32を残しています。ビルド成功が実機数値・画質・性能の合格を意味するものではありません。[今回の実測と制約](moss-performance-optimization-20261020.md)を参照してください。

## 実機未確認のまま起動しないための境界

`native/app/moss_runtime_admission.h`の`PS5_AI_MOSS_RUNTIME_ADMITTED`は既定0です。JSONで解除できません。12GiBの実容量、Qwenのtensor residency/数値、同一context GPU読み戻しを確認する前に、通常アプリから新しいAGC/data pool初期化へ進めません。コンパイル成功はこの解除の根拠になりません。

1GiB region、12GiB data VAと244GiB driver VAはアドレス範囲の準備です。物理quotaの拡大を実装したものではありません。既存の実容量・working reserve・所有権・ログ・GPU完了確認は維持します。

## モデルとホスト試験

モデルは[固定lock](../config/models.lock.json)と[専用ダウンローダー](../tools/download_moss.py)でDへ配置します。取得だけではPS5へ反映しません。MOSSの現行host試験は`tests/moss_*`です。HTTP fixture、pool mock、source guard試験はPS5推論成功として扱いません。継承したWAI/Anima試験は別モデルの履歴です。


## 2026-10-19の選定LoRA対応ビルド

LoRA/lifecycle接続後のfull coreは75 objectsをfresh compile、旧object流用0で完了しました。75 sourceと4,135 header等の前後hashが一致し、archive SHA256は`2184d88babd47ab8826b03c58f473de7e98adb19a79ab47798e97d576e2f56bc`です。

その完了後、native packagerへ選定LoRAのLICENSEコピー1行だけを追加しました。core用181入力freezeの履歴を書き換えず、C/C++・header・overlay・core builder/profileが変わっていないことと、native側213入力freezeを分けて記録しました。native builderはさらに3,793入力の前後一致を検査しました。SELF SHA256は`50131fb15447eec767e3f976ed19255870d2de1068512a81f86772128fdac30b`、integrity正常、動的依存6、initializer24です。これらはguard0のホスト成果物で、PS5実機起動・推論成功ではありません。

LoRAの再現可能なhost checksは`tests/moss_lora_checks.py --llvm <LLVM/bin> --sd <private-SD> --out <fresh-D-output>`です。数値/shape/実upstream name conversionを検査し、モデル推論やPS5接続は行いません。詳しい検証範囲は[status](status.md)と[LoRA](moss-requested-lora.md)を参照してください。

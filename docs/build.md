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

[source-freeze例](../config/moss-source-freeze.json.example)をD側の個人用JSONへコピーし、`repository_root`と`private_vendor_root`の2パスを実際のcloneへ合わせます。固定した21ファイルのsize/SHAが一致しなければビルドを拒否します。ソース改修後はレビュー・host確認・新しいfreezeを行います。core builderの引数は同toolの`--help`を参照してください。SDK・shader生成物・レビュー済み依存アーカイブが別途必要で、クリーンPCからのワンコマンド依存構築ではありません。

nativeは`title/PPSA99190/eboot.bin`、gatewayは`ps5-ai-moss-openai-gateway.elf`を出力します。SELF整合性・loaded bytes・動的依存・ソースhashを検証します。`fp32`は演算設定であり、GGUFのQ4保存形式とは別です。

## 実機未確認のまま起動しないための境界

`native/app/moss_runtime_admission.h`の`PS5_AI_MOSS_RUNTIME_ADMITTED`は既定0です。JSONで解除できません。12GiBの実容量、Qwenのtensor residency/数値、同一context GPU読み戻しを確認する前に、通常アプリから新しいAGC/data pool初期化へ進めません。コンパイル成功はこの解除の根拠になりません。

1GiB region、12GiB data VAと244GiB driver VAはアドレス範囲の準備です。物理quotaの拡大を実装したものではありません。既存の実容量・working reserve・所有権・ログ・GPU完了確認は維持します。

## モデルとホスト試験

モデルは[固定lock](../config/models.lock.json)と[専用ダウンローダー](../tools/download_moss.py)でDへ配置します。取得だけではPS5へ反映しません。MOSSの現行host試験は`tests/moss_*`です。HTTP fixture、pool mock、source guard試験はPS5推論成功として扱いません。継承したWAI/Anima試験は別モデルの履歴です。

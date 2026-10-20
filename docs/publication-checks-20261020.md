# 公開前の確認（2026-10-20）

公開するソースと資料の整合性を確認しました。以下はWindowsでのホスト検査で、モデル推論や新しいPS5生成テストではありません。PS5の生成時間・画像・メモリは[実測比較](moss-turbo-memory-options-20261020.md)と[検証記録](verification.json)を参照してください。

| 検査 | 結果 |
|---|---|
| native／OpenAI互換APIのloopback検査 | 134項目PASS。認証、モデル不一致の拒否、1回だけの転送、エラー伝達、埋込OpenAPI一致を確認 |
| Turbo4＋bounded uploadのbuild contract | 5テストPASS |
| FFN分割のprofile／recipe検査 | 90項目PASS |
| F16 storageのpolicy／receipt検査 | policy61項目、F32／F16各42項目PASS。GPU実行を含まない |
| dashboard／backend health | 描画領域guardと応答parser10項目PASS |
| DiT policy・形状・storage判定 | policy39、実モデル形状graph6,913、storage重なり200,007項目PASS。ホストでlearned推論・GPU実行なし |
| staged公開ファイル検査 | 機密情報・個人path・モデル／実行binaryの混入、source inventory、PNG、Markdownリンク、埋込schemaを確認 |

[公開検査ツール](../tools/check_publication.py)は作業ファイルだけでなくGitのstaged blobを読みます。[source inventory](source-snapshot.json)は公開ファイルのsize／SHA一覧です。実行ファイルのビルド証明や、消費したSDK／依存header全体のfreezeとは区別します。

公開対象は固定条件の10GiB Turbo4＋upload trialです。任意prompt向けの通常運用試作は初回要求をHTTP507で生成前に拒否したため除外しました。試作は通常APIで終了し、独立した読み取り観測でGPUアプリ不在・既存サービスと登録・モデルmetadata維持を確認しました。ファームウェア更新やシステム領域の変更は行っていません。

追加DiT最適化の実機検証は[専用資料](moss-dit-fusion-20261020.md)へ記録しました。同coreのON/OFF双方で256→1024各1枚、既存ゲートと追加16ケース、PNG一致、通常終了後の独立観測を確認しています。

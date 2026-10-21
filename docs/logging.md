# MOSSのログと保存量

`logging.level: "normal"`では日常の記録量を抑え、詳細調査時だけ明示的に`"debug"`を選びます。設定はAGC／worker初期化前に一度読み、未知の値や型を拒否します。全dispatchの詳細を常時出す説明にはしていません。

startup、engine、arenaのjournalは起動、必要な容量・ownership・GPU completion、数値ゲート、job結果、異常の追跡を残します。routine抑制とbounded rotationは別の処理です。journal書込時間は対象処理のnested CPU wallで、純GPU時間や全て除去できる待ち時間ではありません。

bounded journal／stdioはcurrentとpreviousを保持し、上限を超える前に切り替えます。limitは呼出側ごとに定義され、recordを途中まで書いて成功扱いにはしません。既存の巨大ファイルを永久にappendし続ける方式ではありません。書込・同期・metadata確認に失敗した場合も、生成成功へ置き換えません。

通常API版は生成回数のsoftware capを撤去し、各processのoutput＋job metadataを既定8GiBへ制限します。新規・未完了jobごとに32MiBを見込み、生成前に507で受理を止めます。これは実filesystem空き容量の取得ではなく、他のデータの消費までは防ぎません。成功PNGを自動削除せず、退避と停止中の管理を行います。[配置と保存枠](deployment.md)。過去の最大4試行の記録は診断buildのものです。

全文ログを検証資料へ取得する場合は、その時点のowner、boot、inode、deviceを照合し、size／mtimeの前後一致、no-gap、権限復元を確認します。rotationで先頭を失った断片を全文証拠と表示しません。rawログは非公開の検証場所へ保管し、公開repoには集約した条件、秒数、確認範囲だけを載せます。

認証token、秘密のhash、個人path、raw process記録を公開しません。Direct予約、APIのGPUラベル付きpeak、プロセスRSS、heap mapped capacityは別の量です。それらを足し合わせて全物理RAM／専用VRAMにしません。[通常版検証](moss-service-20261021.md)と[status](status.md)を参照してください。

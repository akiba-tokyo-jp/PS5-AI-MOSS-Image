# MOSSのログと保存量

`logging.level: "normal"`では日常の記録量を抑え、詳細調査時だけ明示的に`"debug"`を選びます。設定はAGC／worker初期化前に一度読み、未知の値や型を拒否します。全dispatchの詳細を常時出す説明にはしていません。

startup、engine、arenaのjournalは起動、必要な容量・ownership・GPU completion、数値ゲート、job結果、異常の追跡を残します。routine抑制とbounded rotationは別の処理です。journal書込時間は対象処理のnested CPU wallで、純GPU時間や全て除去できる待ち時間ではありません。

bounded journal／stdioはcurrentとpreviousを保持し、上限を超える前に切り替えます。limitは呼出側ごとに定義され、recordを途中まで書いて成功扱いにはしません。既存の巨大ファイルを永久にappendし続ける方式ではありません。書込・同期・metadata確認に失敗した場合も、生成成功へ置き換えません。

この公開版は10GiBのbounded performance trialです。1process最大4有効試行という制限と、ログrotationは実装されていますが、PNG／job metadata全体の長期quotaや実ディスクheadroomの毎job強制を検証済みの運用機能として含めません。保存した画像・jobを別の場所へ退避して管理します。実容量を取得できないnative互換関数の固定値を、実HDD空き容量と呼びません。

全文ログを検証資料へ取得する場合は、その時点のowner、boot、inode、deviceを照合し、size／mtimeの前後一致、no-gap、権限復元を確認します。rotationで先頭を失った断片を全文証拠と表示しません。rawログは非公開の検証場所へ保管し、公開repoには集約した条件、秒数、確認範囲だけを載せます。

認証token、秘密のhash、個人path、raw process記録を公開しません。Direct予約、APIのGPUラベル付きpeak、プロセスRSS、heap mapped capacityは別の量です。それらを足し合わせて全物理RAM／専用VRAMにしません。[最新比較](moss-turbo-memory-options-20261020.md)と[status](status.md)を参照してください。

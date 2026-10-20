# MOSSの状態画面

PS5のCPUで濃紺の背景、状態・メモリ・直近ログのカードを描画します。表示用のVulkan contextや外部GUIは追加しません。現在の表示タイトルはPS5 AI MOSS Imageです。アイコンはproject owner提供のキャラクター画像です。[asset provenance](../assets/sce_sys/icon-provenance.json)を保持します。

表示できる値はアプリ名、model、PS5 UTC時計、display heartbeat、engine状態、job進捗、アプリ自身のDirect pool使用量／予約／peak、完了済みfence、filter済み直近ログです。取得不可は項目ごと省略します。CPU／GPU使用率や温度を取得していない場合に推定値は表示しません。Direct表示はPS5全体や専用VRAMの総使用量ではありません。

表示heartbeatと実際のGPU生成完了は別です。`GET /health`の`tv_display`はVideoOutの表示完了が確認できたときに返し、一時的なmutex競合ではnullになります。`gpu_verified`は画像生成側の実GPU完了確認に基づき、画面だけが動いていても生成成功を意味しません。

font bitmapはOFL-1.1です。元copyrightと派生fontの固有名称を[font provenance](../native/ui/font-provenance.json)、[license](../licenses/WAI-Dashboard-Glyphs-OFL.txt)に保持します。license内の派生名称はアプリの対応modelを表すものではありません。写真や別modelの古いdashboard benchmarkはこのMOSS資料に含めません。

再現用の`tests/dashboard.py`はhost描画・metadataの検査で、PS5 learned-model推論を実行しません。実機の起動・表示・生成・通常終了は独立して確認します。

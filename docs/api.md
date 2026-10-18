現行版：漫画LoRA強度1を既定適用し、`manga_strength:0`で無効化できます。自動monochromeは既定falseです。OpenAI JSONでも両項目をWAI拡張として受け付けます。ControlNet Canny/OpenPoseは256/1024を実機確認済み：[結果と入力手順](controlnet.md)。

# API仕様

normal/debugとログ容量上限は[ログ設定](logging.md)を参照。OpenAIエンドポイントの仕様は維持しています。

両サービスとも認証トークンが必要です。トークンはソースに含めず、起動時に専用ファイルから読み込み、ファイルを削除してメモリ上で保持します。
同じプライベートIPv4の/24内からのみ接続できます。HTTPのため、LAN外へ転送・ポート開放しないでください。

## OpenAI Imagesの互換部分：8189

クライアントbase URL: `http://<PS5_IP>:8189/v1`、model: `wai-illustrious-sdxl-v17.0`。
別CPUプロセスが同じPS5の8188へ1回だけPOSTし、ジョブ完了を待って実PNGを返します。
Windowsやクラウドへ推論を転送する実装はありません。

| メソッド・パス | 動作 |
|---|---|
| GET `/v1/models` | モデル一覧 |
| GET `/v1/models/wai-illustrious-sdxl-v17.0` | モデル情報 |
| POST `/v1/images/generations` | 同期応答。base64 PNGまたは署名付きURL |
| GET `/health` | 自身と同じPS5のバックエンド状態 |
| GET `/openapi.json` | OpenAPI3.1文書 |
| GET `/v1/images/{id}.png?expires=…&signature=…` | 返却された画像用署名URL |

通常ルートのヘッダーは `Authorization: Bearer <token>`。URLにはAPI秘密トークンを含めません。
画像URLの署名は画像1枚・最大1時間・サーバー起動セッションに限定します。URL自体を画像へのアクセス権として扱ってください。
画像URLでもLAN制約は残り、改変・期限切れは401です。

| パラメーター | 対応 |
|---|---|
| prompt | 必須、UTF-8で4096 bytes以下 |
| model | 上記モデルのみ |
| n / output_format / stream | 1 / png / falseのみ |
| size | auto（1024×1024）、または64〜1024・64の倍数のWIDTHxHEIGHT |
| response_format | b64_json（既定） / url |
| quality / background | auto または standard / auto または opaque |
| user / safety_identifier | 256文字以下のメタデータ。モデレーション機能を追加するものではありません |
| negative_prompt | 拡張、4096 UTF-8 bytes以下 |
| steps / cfg_scale | 拡張、1〜100（既定8） / 0〜30（既定5） |
| seed | 拡張、-1または非負signed64-bit整数 |
| sampler | 拡張、euler / euler_a（既定） / dpm++2m |

任意項目のnullは既定値を使います。body上限24KiB。
未対応パラメーター・batch・透過出力・JPEG/WebP・streamingは400などで拒否します。マスク付き編集は [仕様](mask-api.md) を参照してください。
成功は `{"created":...,"data":[{"b64_json":"..."}]}` または`url`です。
`x-request-id`はアダプターのID、`X-PS5-Native-Job-ID`は実際のネイティブ生成IDです。
失敗は`error.message/type/param/code`を持つOpenAI形式です。
同期生成は1件までで、競合は429。同期応答待ちでもhealthに応答します。待機上限30分です。

## ネイティブ非同期API：8188

| メソッド・パス | 動作 |
|---|---|
| GET `/health` | model_loaded、GPU状態、backend、実行中・待機中件数 |
| POST `/api/generate` | prompt / negative_prompt / width / height / steps / cfg_scale / seed / sampler、202とjob_id |
| GET `/api/jobs/{job_id}` | queued / running / completed / failed、進捗・エラー・画像URL |
| GET `/api/images/{image_id}` | 実PNGを取得 |

HTTPタイムアウト・切断はネイティブ生成のキャンセルを意味しません。
IDを取得済みならジョブを確認し、不明なPOSTを自動再送しないでください。
致命的なretained状態はfailed/degraded、新規生成503、終了待ちできないshutdownは409です。
管理用shutdownはアプリの正常な後処理と同義ではありません。[運用](deployment.md)を参照してください。

## 参照画像とControlNet拡張

認証付きnative `POST :8188/api/inputs`へPNG/JPEGを送り、返されたimage_idを使用します。`GET /api/inputs/{id}`で寸法・SHA256・参照数を取得、`DELETE`で未参照入力を削除できます。ジョブが参照中の削除は拒否します。

OpenAI互換`POST :8189/v1/images/generations`に`controlnet:canny|openpose`、`control_image_id`、`control_strength`（有限0.01..1.5、既定1）を渡します。入力は出力と同じ寸法の完成済みedge/pose mapです。自動OpenPose検出はありません。SDKでは`extra_body`を使います。これらはOpenAI標準ではなくWAI拡張です。

`init_image_id`と`denoise_strength`はimg2img、`reference_image_id`と`ip_adapter_strength`はIP-Adapter経路です。multipart `POST /v1/images/edits`も対応しますが、これらの参照画像経路は画質・人物同一性の課題が残る実験機能です。今回のControlNet検証はその品質検証を兼ねません。マスク付きimg2img/inpaintingは実機へ反映し、白黒256と透過1024＋OpenPoseの領域保持を確認しました。指定した色や画質の合格とは分けて扱います。[マスク仕様](mask-api.md)・[実測](masked-inpainting-20261019.md)。

`scripts/Invoke-WaiControlNet.ps1`を使用できます。タイムアウト時に自動再送しないでください。ネイティブjobsの`controlnet_evals`、`controlnet_tensors`、`gpu_controlnet_verified`で実行結果を確認できます。詳細な全フィールドは[OpenAPI文書](openapi.json)を参照してください。

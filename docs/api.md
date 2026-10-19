# PS5 AI Anima API

実機未検証の開発版です。WAIのAPI・実測値は[移行前の記録](wai-base-api.md)に分けています。

| 用途 | 接続先 |
|---|---|
| OpenAI Images互換部分 | `http://<PS5_IP>:8191/v1` |
| PS5 native非同期API | `http://<PS5_IP>:8190` |
| model | `anima-turbo-v1.1` |

両APIとも専用ファイルから読み込むBearer認証が必要です。トークンは起動後にファイルから削除してメモリに保持します。公開範囲は同じprivate IPv4 /24内です。WAIとAnimaは待機を確認して通常終了し、切り替えて使います。

## OpenAIクライアント

`POST /v1/images/generations`はPS5内のnative APIへ1件だけジョブを渡して待機します。モデル推論はWindowsへ転送しません。JSON例：

```json
{
  "model": "anima-turbo-v1.1",
  "prompt": "1girl, adult woman, blue hair, glasses, smiling, anime illustration",
  "size": "512x512",
  "n": 1,
  "response_format": "b64_json",
  "steps": 8,
  "cfg_scale": 1.0,
  "seed": 20261019,
  "sampler": "euler"
}
```

`steps`・`cfg_scale`・`seed`・`sampler`はこのサーバーの拡張です。OpenAI Images API全体の実装を意味しません。応答はbase64 PNGまたは期限付き署名URLです。`GET /v1/models`、`GET /health`、`GET /openapi.json`で接続先を確認できます。[OpenAPI仕様](openapi.json)。

PowerShellクライアントは[scripts/Invoke-AnimaOpenAI.ps1](../scripts/Invoke-AnimaOpenAI.ps1)です。トークンはパラメーターで指定するファイルから読み込み、表示しません。

## native非同期API

`POST /api/generate`はジョブを受け付け、`GET /api/jobs/{job_id}`で状態、生成時間、メモリ使用量、画像URLを返します。`GET /api/images/{image_id}`でPNGを取得します。`GET /health`のモデル識別とGPU検証状態を確認してください。

初期対応はtext-to-imageのみです。WAI用漫画LoRA、SDXL ControlNet、IP-Adapter、img2img、マスク、inpainting要求を400で拒否します。WAI用ポートへの誤起動と、異なるモデルへのゲートウェイ転送も拒否します。

タイムアウトしても生成POSTを自動で再送しません。healthのrunning_jobs/queued_jobsと既存ジョブを確認してから次を送ってください。

# PS5 AI MOSS Image API

現在はホスト上で準備している開発版です。MOSSのモデル読込、PS5での推論、PNG生成、実機配備は未検証です。ホストのプロトコル検査では失敗専用の代替処理を使用し、画像を生成しません。

| 用途 | 接続先・識別子 |
|---|---|
| OpenAI Images互換部分 | `http://<PS5_IP>:8193/v1` |
| native非同期API | `http://<PS5_IP>:8192` |
| model | `moss-image-sft` |
| アプリ | `PS5 AI MOSS Image` / `PPSA99190` |
| 専用配置予定先 | `/data/ai/native-moss-image-sft/PPSA99190` |
| ゲートウェイ専用領域 | `/data/ai/moss-openai-gateway` |

WAIは8188／8189、Animaは8190／8191を維持します。WAI・Anima・MOSSは同時に実行せず、停止を確認して切り替える設計です。MOSSのサーバーは他のアプリ用ポートを指定した起動を拒否します。

両APIは専用ファイルから読み込むBearer認証を要求します。トークンは起動後にファイルから削除してメモリに保持し、ソースやURLには入れません。接続元は同じprivate IPv4 /24内に制限します。ゲートウェイは推論を行わず、同じPS5上の8192番へ1件だけ転送します。Windowsでの代替推論や自動再送はありません。

## OpenAIクライアント

`POST /v1/images/generations`の例です。50ステップ・CFG 3・Euler・`flow_shift: 5`が既定です。

```json
{
  "model": "moss-image-sft",
  "prompt": "an adult woman with blue hair and glasses, smiling, anime illustration",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "steps": 50,
  "cfg_scale": 3.0,
  "flow_shift": 5.0,
  "seed": 20261019,
  "sampler": "euler"
}
```

`steps`・`cfg_scale`・`flow_shift`・`seed`・`sampler`・`negative_prompt`はこのサーバー独自の拡張です。OpenAI Images API全体の実装を意味しません。応答形式はbase64 PNGまたは期限付きの画像専用署名URLです。`GET /v1/models`、`GET /health`、`GET /openapi.json`で識別と仕様を確認できます。[OpenAPI仕様](openapi.json)。

幅・高さは64〜1024の整数で、それぞれ16の倍数を指定します。`80x96`など、64の倍数に限らない寸法もプロトコル上は受け付けます。実機で生成可能な最大サイズやメモリ使用量を確認したという意味ではありません。

`flow_shift`はfloatへ変換しても正かつ有限である数を指定します。0・負数・NaN・無限大・floatで0に丸められる極小値を拒否します。省略またはOpenAI側の`null`では5を使用します。`quality: "auto"`／`"standard"`は受け付けますが、別の品質モデルへ切り替える指定ではありません。

PowerShellクライアントは[`Invoke-MossOpenAI.ps1`](../scripts/Invoke-MossOpenAI.ps1)です。認証ファイルを指定して使用し、トークンを表示しません。

## native非同期API

`POST /api/generate`は`prompt`、`negative_prompt`、`width`、`height`、`steps`、`cfg_scale`、`flow_shift`、`seed`、`sampler`、省略可能な`model`を受け付けます。`GET /api/jobs/{job_id}`で状態・生成時間・メモリ使用量・`flow_shift`・画像URLを返し、`GET /api/images/{image_id}`でPNGを取得する設計です。

初期対応はtext-to-imageのみです。SDXL漫画LoRA、ControlNet、IP-Adapter、img2img、マスク、inpaintingの有効な要求は400で拒否します。画像入力のキーは`null`でも拒否します。継承クライアント向けの`manga_strength: 0`と`manga_monochrome: false`だけは無効値として受け付け、プロンプトへ白黒指定を追加しません。Anima・WAIなど異なるモデルIDも拒否します。

ゲートウェイはnative healthのモデルIDと予定モデルパスの両方を検証し、不一致なら生成を転送しません。実機の完了ジョブでGPU使用を確認できた場合だけ画像を成功として返す設計です。

タイムアウトしても生成POSTを自動で再送しません。実機検証時にはhealthの`running_jobs`／`queued_jobs`と既存ジョブを確認してから、次の要求を送ります。

## モデル構成の設定例

[`config/moss-native.json.example`](../config/moss-native.json.example)はモデル本体・Qwen3.5エンコーダー・Flux2 VAE・外部Qwen3.5 tokenizerの配置予定を示します。ファイルは同梱していません。モデル名・必要な量子化形式・バックエンド互換性・メモリ設定は、別途のソース監査と実機検証が必要です。既存Anima用の重みやtokenizerへ代替する動作は設けません。

設定例のDirect予約目標は12GiB（12,884,901,888 bytes）、その内側のVulkan推論予算は11.5GiBです。`working_reserve_bytes`は1GiBを維持します。実機で12GiBを確保できた実測を示す値ではありません。現在の移植上の制限を解決し、実行許可を検証するまでは配備・起動しません。

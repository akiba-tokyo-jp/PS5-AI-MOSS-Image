# PS5 AI MOSS Image API

privateの実験版です。2026-10-19に専用10GiB trialでMOSS＋指定LoRAの256×256・50steps生成とOpenAI互換API経由のPNG取得をPS5実機で確認しました。同条件4stepsの色縞は50stepsで解消しました。production runtime admissionは0を維持し、warm・高解像度・連続運用は未検証です。[実測の範囲](moss-native-small-trial-20261019.md)を参照してください。別のホスト用プロトコル検査は失敗専用の代替処理を使用し、モデル推論や画像生成を行いません。

| 用途 | 接続先・識別子 |
|---|---|
| OpenAI Images互換部分 | `http://<PS5_IP>:8193/v1` |
| native非同期API | `http://<PS5_IP>:8192` |
| model | `moss-image-sft` |
| アプリ | `PS5 AI MOSS Image` / `PPSA99190` |
| 専用配置先 | `/data/ai/native-moss-image-sft/PPSA99190` |
| ゲートウェイ専用領域 | `/data/ai/moss-openai-gateway` |

WAIは8188／8189、Animaは8190／8191を維持します。WAI・Anima・MOSSは同時に実行せず、停止を確認して切り替える設計です。MOSSのサーバーは他のアプリ用ポートを指定した起動を拒否します。

両APIは専用ファイルから読み込むBearer認証を要求します。同じMOSSアプリのnativeとgatewayには同じトークン値を各起動ファイルへ渡します。gatewayはその値でnativeを認証するためです。WAI・Animaのトークンとは分離します。トークンは各プロセスが起動後にファイルから削除してメモリに保持し、ソースやURLには入れません。接続元は同じprivate IPv4 /24内に制限します。ゲートウェイは推論を行わず、同じPS5上の8192番へ1件だけ転送します。Windowsでの代替推論や自動再送はありません。

## OpenAIクライアント

`POST /v1/images/generations`の例です。50ステップ・CFG 3・Euler・`flow_shift: 5`が既定です。

```json
{
  "model": "moss-image-sft",
  "prompt": "an adult woman with blue hair and glasses, smiling, anime illustration",
  "size": "256x256",
  "n": 1,
  "response_format": "b64_json",
  "steps": 50,
  "cfg_scale": 3.0,
  "flow_shift": 5.0,
  "seed": 20261019,
  "sampler": "euler"
}
```

`steps`・`cfg_scale`・`flow_shift`・`seed`・`sampler`・`negative_prompt`はこのサーバー独自の拡張です。OpenAI Images API全体の実装を意味しません。応答形式はbase64 PNGまたは期限付きの画像専用署名URLです。`GET /v1/models`、`GET /health`、`GET /openapi.json`で識別と仕様を確認できます。[OpenAPI仕様](openapi.json)。JSON仕様と実機ビルドの埋込schemaは一致を維持しており、infoの`host-prepared`／実機未検証という説明文は初期ホスト構築時の履歴です。今回の結果は本資料で更新し、説明文だけのための再ビルドはしていません。

通常のnative／gateway／PowerShellクライアントの既定stepsは元から50です。初回smokeの4stepsは別の試験契約で、推奨品質設定ではありません。今回の品質比較用SELFは256×256・4または50steps、CFG3、shift5、Euler、seed20261019、LoRA0.7に限定し、各プロセスは1件だけ受理します。上の256×256・50steps例が実測した形です。

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

選定LoRAはアプリ設定の`moss_lora_path`と`moss_lora_strength`で指定し、HTTP要求から任意のモデルパスを渡す機能はありません。未知のengine設定キー、選定外のLoRAパス、0以下・1.5超・floatで有限正値にならない強度は、明示的GPU初期化前のmemory-plan検査で拒否します。構成上の対応とPS5での生成成功は分けて扱います。

設定例のDirect予約目標は12GiB（12,884,901,888 bytes）、その内側のVulkan推論予算は11.5GiBです。`working_reserve_bytes`は1GiBを維持します。native総量報告11.4375GiBのため12GiB＋reserve1GiBは未達です。人が別途許可した試験では10GiB pool・予算9.5GiB・reserve1GiBを使用しました。通常設定の12GiB確保や高解像度対応が実証された意味ではありません。

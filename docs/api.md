# PS5 AI MOSS Image API

画像生成はPS5ネイティブアプリが行います。OpenAI Images互換gatewayは同じ本体のnative APIへ1件だけ要求を転送します。WindowsはHTTPクライアントであり、モデル推論を行いません。[最新の実測比較](moss-turbo-memory-options-20261020.md)と[現在の状態](status.md)を参照してください。

| 用途 | 接続先・識別子 |
|---|---|
| OpenAI Images互換部分 | `http://<PS5_IP>:8193/v1` |
| native非同期API | `http://<PS5_IP>:8192` |
| model | `moss-image-sft` |
| アプリ | `PS5 AI MOSS Image` / `PPSA99190` |
| native配置先 | `/data/ai/native-moss-image-sft/PPSA99190` |
| gateway専用領域 | `/data/ai/moss-openai-gateway` |

両APIはBearer認証を要求し、接続元を同じprivate IPv4 /24内に制限します。nativeとgatewayには同じトークンを各process専用ファイルで渡します。起動後にそのファイルを削除し、プロセス内に保持します。トークンをソース、ログ、URLへ入れないでください。

## 公開する検証profile

現在の速度基準は **Turbo4＋bounded upload、rank144 F32 LoRA、10GiB Direct予約**です。通常の無制限運用サーバーではありません。nativeのproduction runtime admissionは無効で、明示的performance trialが1process最大4回の有効な試行だけを許可します。失敗も回数を消費し、5回目は拒否します。不正な数値要求はclaimを消費しません。

| 項目 | 検証済みの固定条件 |
|---|---|
| 入力 | 下記のcanonical text-to-image request |
| 出力 | PNG、1要求1枚、`256x256`または`1024x1024`の正方形 |
| sampling | Euler、4 steps、CFG1、`flow_shift: 5`、seed20261020 |
| sigma | `[1, 0.910, 0.717, 0.024, 0]`をアプリ内で固定。二重にflow shiftを適用しません |
| LoRA | NSFW0.7＋Turbo1.0を合成したrank144 F32。runtime multiplierは1 |
| メモリ | Direct予約10GiB、内側の推論予算9.5GiB、pool外reserve1GiB |
| 順序 | 最初の256の成功・画像確認・空queue確認後、同processで1024へ進む |

任意prompt／任意seedの通常運用や8GiB運用の成功は公開結果に含めません。別のF16 storage／FFN分割試験は[比較資料](moss-turbo-memory-options-20261020.md)へ分けて記録し、速度基準へ暗黙に混ぜません。`GET /health`で実際のモデル、process、trialの数値・policyを確認してください。readyは、モデル読込済み・画像GPU検証済みという意味ではありません。

## OpenAI Images互換クライアント

`POST /v1/images/generations`の固定requestです。最初は`size`だけを`256x256`にし、確認後に1024へ進めます。

```json
{
  "model": "moss-image-sft",
  "prompt": "A dazzling anime key visual of an adult woman with long flowing silver-white hair and violet eyes, wearing a stylish crimson jacket and a black pleated skirt, standing on a futuristic rooftop above a glowing neon city at night, a shower of shooting stars and a luminous galaxy across the sky, wind-swept hair, confident joyful smile, dramatic low-angle composition, vivid magenta and cyan rim lighting, intricate architecture, crisp clean line art, richly detailed beautiful anime illustration.",
  "negative_prompt": "low quality, blurry",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "steps": 4,
  "cfg_scale": 1,
  "flow_shift": 5,
  "seed": 20261020,
  "sampler": "euler"
}
```

`model`、`prompt`、`size`、`n`、`response_format`はOpenAI Images形式の一部に合わせています。`steps`、`cfg_scale`、`flow_shift`、`seed`、`sampler`、`negative_prompt`は独自拡張です。OpenAI Images API全体を実装した意味ではありません。[OpenAPI仕様](openapi.json)。

成功応答は`created`と1件の`data`を持ち、`data[0].b64_json`にPNGを返します。`x-request-id`はgateway、`X-PS5-Native-Job-ID`はnativeのジョブIDです。native IDを認証付き`GET /api/jobs/{job_id}`と`GET /api/images/{job_id}`で照合できます。

`GET /v1/models`、`GET /v1/models/moss-image-sft`、`GET /health`、`GET /openapi.json`を実装しています。`response_format: "url"`による最大1時間の画像専用HMAC署名URLも実装していますが、base64／native画像を照合した実測と、署名URLの別経路の検証を混同しません。署名URL自体が画像へのアクセス権を含みます。Bearer tokenをURLへ入れません。

`quality: "auto"`／`"standard"`は別モデルに切り替えません。出力はPNGのみ、streamingは非対応です。`user`／`safety_identifier`は互換メタデータとして受け取るだけで、モデレーション処理を意味しません。

PowerShellでは[`Invoke-MossOpenAI.ps1`](../scripts/Invoke-MossOpenAI.ps1)を使います。この公開検証clientはcanonical prompt／negative prompt／seedを固定し、生成POSTを1回だけ送ります。

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile <private-token-file> -Width 256 -Height 256
```

## 受理条件と非対応機能

数値条件はコンパイル済みprofileが検査します。別seed、別sampler、別steps／CFG／flow shift、非対応寸法を指定しても、任意の通常運用modeへ切り替わりません。`scheduler`と`custom_sigmas`はJSONキーではなく、アプリ内policyです。未知のパラメータは拒否します。

promptの構文上限は空でない4096 UTF-8 bytes以下、negative promptは4096 UTF-8 bytes以下です。構文としての文字列受理と、任意promptでの生成品質・通常運用の実証は別です。公開の成功例は上記canonical requestです。CFG1ではnegative側のDiT forwardを行いません。

通常buildはtext-to-imageのみです。両buildに`--moss-img2img-trial`を明示すると、下記のimg2imgが有効になります。IP-Adapter、ControlNet、マスク、inpaintingは非対応で、これらのキーは`null`でも拒否します。継承ABIの無効値として`manga_strength: 0`と`manga_monochrome: false`だけを受け付けますが、漫画LoRAや自動白黒化を有効にする機能ではありません。HTTPから任意のモデルパスやLoRAをupload・選択する機能もありません。

## 参照画像から生成するimg2img

2026-10-21にPS5実機で確認した、明示的default-OFFの限定trialです。[実測・元画像と結果](moss-img2img-20261021.md)。既存の最大4試行をtext-to-imageと共有し、無制限運用には切り替わりません。`GET /health`の`features.img2img`がtrueであることを両portで確認してください。

OpenAI形式は`POST /v1/images/edits`、Content-Typeは`multipart/form-data`です。`image`にPNG/JPEGファイルを1個、`prompt`にtext、他はgenerationsと同じ固定条件を渡します。`reference_mode: img2img`、`denoise_strength`は独自拡張です。URL、mask、reference_image、IP-Adapterモード、複数画像は受け付けません。成功時はgenerationsと同じbase64 PNGとnative job IDを返します。

入力fileは最大8MiB、**img2img推論に渡す元画像は各辺1024以下**にしてください。共通upload storeは2048までdecodeできますが、本trialのengineでは1024を超える入力を拒否します。今回検証した入力は1024×1024 RGB PNGです。PS5上の共通前処理で出力寸法へresizeし、Vulkan VAEで初期latentへ変換します。1024から256への入力変換も確認しました。

`denoise_strength`既定0.5。有限doubleとfloatの双方で0.01以上、floatで0.75未満が必要です。既存SD.cppのstep-fractionなので、requestの`steps: 4`に対し実forwardは1〜3です。floatで[0.01,0.25)は1、[0.25,0.5)は2、[0.5,0.75)は3。各区間内では同じsigma suffixです。`actual_steps`を参照し、全4forwardと誤認しないでください。0.75以上は参照latentを消すsigma1から始まるため拒否します。

PowerShell 7の例です。`$editPrompt`は[実測JSONのrequest.prompt](moss-img2img-20261021.json)と同じ文章を使います。既存の`Invoke-MossOpenAI.ps1`はtext-to-image専用です。

```powershell
$origin = 'http://<PS5_IP>:8193'
$headers = @{ Authorization = 'Bearer ' + [IO.File]::ReadAllText('<private-token-file>').Trim() }
$health = Invoke-RestMethod "$origin/health" -Headers $headers
if (-not $health.features.img2img -or $health.running_jobs -ne 0 -or $health.queued_jobs -ne 0) { throw 'Img2img gateway must be ready and idle' }
$form = @{
    image = Get-Item -LiteralPath '<reference.png>'
    model = 'moss-image-sft'; prompt = $editPrompt
    negative_prompt = 'low quality, blurry'
    size = '256x256'; n = '1'; response_format = 'b64_json'
    steps = '4'; cfg_scale = '1'; flow_shift = '5'
    seed = '20261020'; sampler = 'euler'
    reference_mode = 'img2img'; denoise_strength = '0.25'
}
# One POST. On timeout inspect native health/jobs; do not automatically retry.
$result = Invoke-RestMethod "$origin/v1/images/edits" -Method Post -Headers $headers -Form $form -TimeoutSec 1800 -MaximumRedirection 0
[IO.File]::WriteAllBytes('<new-output.png>', [Convert]::FromBase64String($result.data[0].b64_json))
```

最初は256で出力を確認し、空queue確認後に1024へ進めます。今回の1024はstrength0.5／実3forwardで40.833333秒でした。参照元の構図・色を反映しましたが、実写化promptでもアニメ調を強く残しました。人物同一性の固定や高度なスタイル変換を保証しません。

native非同期経路では、認証付き`POST /api/inputs`へPNG/JPEGのraw bodyを送ります（Content-Typeは`image/png`／`image/jpeg`）。201の`image_id`を`POST /api/generate`の`init_image_id`に指定し、`denoise_strength`を加えます。指定なしは0.5。`GET /api/jobs/{id}`の`mode: img2img`、`init_image_sha256`、`init_encode_seconds`、`actual_steps`で照合できます。HTTPにはfile pathを渡しません。手動uploadの入力はjob完了・reference count0確認後に`DELETE /api/inputs/{image_id}`で削除できます。gatewayは自分が作成した一時inputを既存の完了後cleanupで削除します。

## native非同期APIと再送

`POST /api/generate`では、同じ固定条件を`width`／`height`と他の数値・textフィールドで渡します。`GET /api/jobs/{job_id}`は状態、`generation_seconds`、`actual_steps`、`gpu_used`、ピーク指標、画像URLなどを返します。生成時間はPS5内の計測です。

タイムアウト後も本体で処理が続く場合があります。POSTを自動再送せず、`/health`の`running_jobs`／`queued_jobs`と既存ジョブを確認してください。gatewayは同じ本体8192のモデルID・モデルパスを検証し、native GPU完了を確認してから画像を返します。

CPU/GPUは物理メモリを共有します。Direct予約、プロセスRSS、APIのGPUラベル付きピークを足し合わせて全物理RAMにしません。モデル・precision・メモリ設定の変更はHTTP requestではなく、明示したprofileと検証手順で行います。

## ソースの仕様説明と実行中プロセス

OpenAPI JSONと[`native/gateway/openapi_schema.inc`](../native/gateway/openapi_schema.inc)は同じ内容を保持します。img2imgではnativeとgatewayを再ビルドし、実機のedits応答を検証しています。実際の受理条件は`/health`のコンパイル済みtrial policyと照合してください。画像編集の成功はこの限定img2imgの範囲であり、OpenAI Images API全機能への準拠を意味しません。

モデル重み、認証情報、raw診断ログ、ビルド済みバイナリはリポジトリに含みません。

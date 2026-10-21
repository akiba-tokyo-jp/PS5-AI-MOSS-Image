# MOSS Image API

通常版は`--service-mode`でビルドします。任意prompt・seed、長方形、img2imgを受け付け、生成回数上限と初回256の条件はありません。[実機結果](moss-service-20261021.md)。推論は同じPS5のnative engineで実行します。

| 項目 | 現在の値 |
|---|---|
| OpenAI Images互換部分 | `http://<PS5_IP>:8193/v1` |
| native非同期API | `http://<PS5_IP>:8192` |
| model | `moss-image-sft` |
| 出力 | PNG、1要求1枚。各辺64～1024、16の倍数。長方形可 |
| prompt／negative_prompt | 各4096 UTF-8 bytes以下。prompt必須、negative省略時空文字 |
| seed | -1または0～9223372036854775807。省略／-1はnativeが新規選択 |
| Turbo4 profile | Euler、4 steps、CFG1、flow_shift5 |
| 専用sigmas | `[1,0.910,0.717,0.024,0]`、二重のflow shiftなし |
| adapter | NSFW0.7＋Turbo1.0合成rank144 F32、runtime multiplier1 |

文章・被写体・negativeを自動追加しません。CFG1ではnegative側のDiT forwardを行いません。Turbo4 adapterに合わせたsampling条件は明示的なモデルprofileとして固定し、別steps／CFG／samplerへの自動変更はしません。HTTPでモデルファイルや任意LoRAを差し替える機能はありません。

通常版では、文章長・tensor形状によって一部の最適化kernelを選ばず既存GPU経路へ戻ることを許可します。全kernelの使用実績を要求するのはdiagnostic buildだけです。起動時GPU読み戻し、実forward数、Q4 F32累積、有限値・GPU完了・容量の検査は通常版でも必須で、実際の選択数と未使用maskをログへ記録します。

## 認証・状態

全APIはBearer認証、private IPv4の同じ/24内に限定します。tokenをURL、ログ、ソースへ入れないでください。署名画像URLはそのURL自体が画像専用アクセス権を持ちます。両processへ同じ値を別の一時ファイルで渡す[起動手順](deployment.md)を使用します。

`GET /health`で`service_mode: true`、`requires_256_first: false`、`max_generation_attempts: null`、`model_profile: turbo4`を確認できます。`status: ready`はモデル読込済みという意味ではなく、`model_loaded`を別に確認します。`features.img2img`、`running_jobs`、`queued_jobs`も返します。

## text-to-image

`POST /v1/images/generations`へJSONを送ります。初回から1024を指定できます。

```json
{
  "model": "moss-image-sft",
  "prompt": "An anime illustration of a peaceful garden.",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "seed": -1,
  "steps": 4,
  "cfg_scale": 1,
  "flow_shift": 5,
  "sampler": "euler"
}
```

成功時は`created`、`data[0].b64_json`にPNGを返します。`response_format: url`は画像専用HMAC署名URL（最大1時間）です。`X-PS5-Native-Job-ID`と`X-PS5-Seed`ヘッダーからnative jobと実際のseedを取得できます。seedは64bit整数なので、JavaScript等では文字列として保持し、精度を落とさないでください。

`GET /v1/models`、`GET /v1/models/moss-image-sft`、`GET /openapi.json`も利用できます。OpenAI Images互換はこの部分実装を指します。`seed`、`steps`、`cfg_scale`、`negative_prompt`、`flow_shift`、`sampler`は独自拡張です。streamingや全OpenAIモデルへの互換性を意味しません。`quality: auto/standard`でモデルを切り替えません。

[Windows client](../scripts/Invoke-MossOpenAI.ps1)の例です。PowerShellから実際に5件の生成とPNG取得を確認しました。

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An anime illustration of an adult woman reading a book in a sunny library.' -Width 512 -Height 768 -Seed 17 -OutputDirectory D:/moss-output
```

## img2img

`POST /v1/images/edits`、`multipart/form-data`にPNG/JPEGの`image`1個と`prompt`、上記のsamplingフィールド、`reference_mode: img2img`、`denoise_strength`を送ります。入力は8MiB以下、各辺2048以下。PS5で出力寸法へresizeし、PS5 Vulkan VAEでencodeします。外部URL入力、複数画像、マスク、inpainting、IP-Adapter、ControlNetは未対応です。

`denoise_strength`は0.01～1、既定0.5。4stepのsuffix選択で、連続的なnoise強度の調整ではありません。floatで[0.01,0.25)は実1 forward、[0.25,0.5)は2、[0.5,0.75)は3、[0.75,1]は4です。**0.75以上は初期sigma1で参照latentの寄与が0になり、実質的な再サンプリングになります。** 元画像の保持には0.75未満を使ってください。各区間内では同じsigma suffixです。

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An ink drawing.' -ReferenceImage D:/images/reference.png -DenoiseStrength 0.5 -Width 512 -Height 512 -Seed 12345 -OutputDirectory D:/moss-output
```

結果の人物同一性や実写変換を保証する機能ではありません。[今回の出力](moss-service-20261021.md)、[前段のimg2img結果](moss-img2img-20261021.md)。

## native非同期API

`POST /api/generate`では`prompt`、`negative_prompt`、`width`、`height`、`steps`、`cfg_scale`、`seed`、`sampler`、`flow_shift`を受け付けます。202応答の`job_id`と解決済みseedを保存してください。`GET /api/jobs/{job_id}`はqueued/running/completed/failed、エラー、`generation_seconds`、実際の`actual_steps`、seed、寸法、メモリ指標、画像URL等を返します。`GET /api/images/{job_id}`でPNGを取得できます。

img2imgは認証付き`POST /api/inputs`へPNG/JPEGのraw bodyを送り、201の`image_id`を`init_image_id`に指定します。`denoise_strength`も渡します。jobに`init_image_sha256`、`init_encode_seconds`を記録します。手動入力は参照数0・job完了後に`DELETE /api/inputs/{image_id}`で削除できます。gatewayは自分が作った入力を処理後にcleanupします。HTTPでローカルfile pathを受け付けません。

## 直列処理・保存量・エラー

GPU生成は直列です。job tableは32 slotsで完了slotを再利用するため、32回の累積上限ではありません。clientは空queueを確認してから1件だけPOSTし、タイムアウト時は既存jobとhealthを確認します。自動で再送しないでください。未知・非対応パラメータは拒否します。[OpenAPI](openapi.json)を参照してください。

通常版は各processの`output`＋`jobs`に既定8GiBの保存枠を設けます。新規jobおよび未完了jobごとに32MiBの余白を見込み、収まらなければ生成前に507を返します。`--storage-limit-mib`で変更可。これは実ディスク空き容量の測定ではなく、既存ファイルや他アプリの消費までは防ぎません。生成物を自動削除せず、退避後の管理は[配置資料](deployment.md)に従います。入力storeは別に128MiB／32件／24時間cleanup、[ログ](logging.md)はbounded rotationです。

物理Direct、RSS、GPUラベル付きピークは重複し、合算できません。容量・有限値・GPU完了の検査を満たさない生成は失敗として返します。回数制限の撤去は、任意の要求が必ずメモリ内に収まる保証ではありません。

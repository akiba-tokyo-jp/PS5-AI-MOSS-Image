# MOSS Imageの配置計画（未実施）

MOSSのモデル取得・変換・native SELFのfresh cross buildはWindows/Dドライブ上で完了しています。MOSS本体・モデルは未配備で、実機推論は未確認です。PS5上の現boot・既存サービス・GPUアプリ所有者を確認してから扱います。既存WAI/Anima、FW、システム領域へ変更を加えません。

候補rootは`/data/ai/native-moss-image-sft/PPSA99190`です。native8192、gateway8193（root `/data/ai/moss-openai-gateway`）を使い、WAI8188/8189・Anima8190/8191から分離します。GPUアプリは同時起動しません。

```text
PPSA99190/
  eboot.bin
  sce_sys/param.json
  sce_sys/icon0.png
  sce_module/libc.prx
  models/
    MOSS-Image-SFT-Q4_K.gguf
    Qwen3.5-9B-Q4_K_M.gguf
    flux2-vae_small.safetensors
    tokenizer_qwen3.5.json
    moss-image-nsfw-uncensored-r16-f32.safetensors
  licenses/
  ai-runtime/config.json
  ai-runtime/config/api-token
  ai-runtime/logs/
  ai-runtime/jobs/
  ai-runtime/output/
```

DownloaderはD側の`diffusion/`、`text_encoders/`、`vae/`、`tokenizers/`とraw adapterの`loras/`へ保存します。選定一式の取得と全体SHA256読み戻しは完了しました。[stage plan tool](../tools/models/prepare_moss_model_stage.py)は基本4個＋F32 derivativeの5ファイルを上記のflatな`models/`へ対応付けます。作成するのは転送計画だけで、PS5接続・コピー・現在の巨大ファイル再hashを行いません。実転送では全量をhashし、転送先でも全体size/SHAを確認してから公開します。

選定LoRAの元ファイルは`loras/moss-image-nsfw-uncensored-r16-bf16.safetensors`です。[専用converter](../tools/models/convert_moss_lora_bf16_f32.py)で全504 tensor・元SHA・名前・形状・有限値を検査し、値を正確にF32へ展開してq/k/vの結合先を正規化します。変換済みの`derived/moss-image-nsfw-uncensored-r16-f32.safetensors`だけを上記の固定モデルパスへ配置します。元BF16ファイルを名前だけ変えて配置しません。PS5でのLoRA適用・GPU数値・生成成功は別の検証です。今はstage/転送していません。VAEのF32変換やモデルの暗黙置換は行いません。

新タイトルのホーム登録はファイル転送と別です。既存のmount/登録条件と空いているTitle IDを実機で確認し、システム領域を変更せず扱える経路を確認する必要があります。今回の準備に自動登録・再起動・jailbreak改変は含めません。

実機再開時はまず読み取りで利用可能Direct総量・空き区間・CPU側余裕を測定します。12GiB+1GiB reserveを満たせない条件では容量チェックを維持して停止します。モデルなしの段階的割当/解放とCompute読み戻し、Qwen3.5 CPU数値比較、MOSS DiT/VAE数値・PNGの順で証拠を残します。実機未検証の通常アプリはcompile-time admission0で止めます。

MOSSのnativeとgatewayは、同じ認証tokenの値を各プロセス専用の起動ファイルへ渡します。gatewayはこの値をそのままnativeへのBearer認証に使うため、異なる値にはできません。WAI・Animaとは別のMOSS専用tokenを生成し、ソースへ直書きしません。runtime/configはアプリownerだけに許可し、tokenは0600、LAN bindとBearer認証を維持します。

nativeは起動時に`/app0/ai-runtime/config/api-token`が存在しないことを確認し、当該起動のowner witnessを確認した後の発行を待ちます。起動前にtokenを残さず、publisherで当該起動へ発行します。両プロセスはそれぞれの起動ファイルを読み取り後に削除し、値はメモリに保持します。token値をログ・URL・資料へ出しません。API生成POSTは1回だけです。タイムアウト時はhealth/job照会で結果を追跡し、自動再送しません。

[`config/native.json.example`](../config/native.json.example)と[`config/moss-native.json.example`](../config/moss-native.json.example)は同じMOSS設定例です。engineの未知キーはmemory-planの読み取り時に拒否し、明示的AGC/Direct初期化へ進めません。選定LoRAは`moss_lora_path`の固定canonical F32パスと`moss_lora_strength`（既定0.7、有限かつ0より大きく1.5以下）で指定します。空パス・無効強度によるbase-only運用は、この選定構成では受け付けません。汎用の`lora_path`は未知キーです。


## 起動から通常終了までの境界

新しいboot、title owner/PID、session/launchの組合せを受理してから、当該起動にruntime/configとtokenを発行します。古いbootやPID番号の再利用だけを根拠にmount、起動、config受理、モデル稼働を判定しません。登録と`/app0`のmount、配備ファイル、起動ownerの一致も別々に確認します。

現sourceはconfiguration、明示AGC、driver、Direct、backend、GPU読み戻し、readyと、failed/retainedを分けて観測します。healthの`lifecycle`は観測値であり、OS closeの許可証ではありません。guard0の準備版はconfigurationの前提検査で拒否されます。SCE constructorの効果は不明であり、明示AGC要求0だけで安全な終了やGPU副作用なしとは判定しません。

APIが応答できる状態でready前・初期化失敗・retainedの場合、`POST /api/admin/shutdown`は409を返し、別途fresh ownerを確認した通常OSタイトル終了を求めます。readyかつidleの経路でも、worker join、GPU quiescence、checked model cleanup、所有権/完了ログの証拠がそろってから終了へ進みます。初期化失敗をreadyとして扱ったり、自動kill・`exit`・手動GPU解放で代用しません。このlifecycle実装とhost検査は完了していますが、MOSS実機での終了/再起動成功は未確認です。

LoRAの実機受入は設定パスの存在だけでは足りません。各生成で504 loaded/504 matched/504 used/0 skipped、GPU完了と有限値、実PNGとWindows取得を確認します。POSTは1回、job IDで追跡し、応答が曖昧なら次のPOSTを送りません。通常終了とowner消失を確認してから次のアプリを起動します。

# MOSS Imageの配置計画（未実施）

現在はWindows/Dドライブ上の準備だけです。PS5はAnimaの停止・再起動と既存サービスの回復確認が先です。既存WAI/Anima、FW、システム領域へ変更を加えません。

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
  licenses/
  ai-runtime/config.json
  ai-runtime/config/api-token
  ai-runtime/logs/
  ai-runtime/jobs/
  ai-runtime/output/
```

DownloaderはD側の`diffusion/`、`text_encoders/`、`vae/`、`tokenizers/`へ保存します。将来のstage工程で、[lock](../config/models.lock.json)の4個の`relative_install_path`を上記`models/`へbasenameを変えずにコピーし、転送前後のsize/SHAを照合します。今はstage/転送していません。VAEのF32変換やモデルの暗黙置換は行いません。

新タイトルのホーム登録はファイル転送と別です。既存のmount/登録条件と空いているTitle IDを実機で確認し、システム領域を変更せず扱える経路を確認する必要があります。今回の準備に自動登録・再起動・jailbreak改変は含めません。

実機再開時はまず読み取りで利用可能Direct総量・空き区間・CPU側余裕を測定します。12GiB+1GiB reserveを満たせない条件では容量チェックを維持して停止します。モデルなしの段階的割当/解放とCompute読み戻し、Qwen3.5 CPU数値比較、MOSS DiT/VAE数値・PNGの順で証拠を残します。実機未検証の通常アプリはcompile-time admission0で止めます。

nativeとgatewayには別の認証tokenを生成し、ソースへ直書きしません。runtime/configはアプリownerだけに許可し、tokenは0600、LAN bindとBearer認証を維持します。API生成POSTは1回だけです。タイムアウト時はhealth/job照会で結果を追跡し、自動再送しません。

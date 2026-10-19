# Animaの配置・切替

新タイトルの実機登録は未実施です。既存WAIの登録を上書きする更新ではありません。[WAIの旧配置資料](wai-base-deployment.md)は履歴として保持しています。

```text
/data/ai/native-anima-turbo-v11/PPSA99189/
  eboot.bin
  sce_sys/param.json
  sce_sys/icon0.png
  sce_module/libc.prx
  models/anima-turbo-v1.1-f16.safetensors
  models/qwen_3_06b_base.safetensors
  models/qwen_image_vae-f16.safetensors
  licenses/
  ai-runtime/config.json
  ai-runtime/config/api-token
  ai-runtime/logs/
  ai-runtime/jobs/
  ai-runtime/output/
```

native APIは8190、別CPUプロセスのOpenAI gatewayは8191です。gatewayのrootは`/data/ai/anima-openai-gateway`。WAIの8188/8189と別ですが、GPUアプリは同時起動しません。

nativeのbind IPv4は`native/app/server.c`に固定されています。自分のprivate LANアドレスに合わせてビルドしてください。`0.0.0.0`やインターネットへ公開しません。

設定は[Anima設定例](../config/anima-native.json.example)を使用します。WAIの設定や一度限りtokenを流用しません。runtime/configはアプリUIDの0700、tokenは0600です。新規登録された本人のPID・起動時刻・TitleID・bootを確認してから設定を渡し、native bootstrapを進めます。tokenは読込後削除されます。

ホーム画面への新規登録には、専用メタデータとmount準備が必要です。FTPでebootを置くだけでは登録できません。システム領域への操作を含む初期登録は、対象と復旧境界を確認した別の管理工程です。一般向けの自動登録installerは本リポジトリに含めていません。

切替は、元のnative/gatewayのrunning_jobsとqueued_jobsが0であること、正しいownerであることを確認し、通常のOS終了を使います。元アプリのPID消失とBigApp状態を確認してから次を起動します。強制終了やGPU reset、共有Directの手動解放で切り替えません。

Anima起動後はAGC/RADVの初期化、64MiB Compute読み戻し、health、512×512の実生成、PNG取得の順に確認します。初期Computeだけではモデル推論のGPU成功と扱いません。DiT/VAEの実計算完了・有限値証拠と出力PNGを別に確認します。

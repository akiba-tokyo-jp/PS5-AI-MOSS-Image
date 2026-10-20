# MOSS Imageの配置と終了

PPSA99190、native8192、OpenAI Images互換gateway8193でPS5画像生成を検証しました。公開する速度基準はTurbo4＋bounded upload、rank144 F32、Direct10GiBの限定performance buildです。無制限の通常運用版ではありません。firmwareやsystem領域を書き換える一般インストーラー、無人jailbreak／復旧ツールは含めません。

## 配置内容

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
    moss-nsfw07-turbo4-r144-f32.safetensors
  licenses/
  ai-runtime/config.json
  ai-runtime/config/api-token
  ai-runtime/logs/
  ai-runtime/jobs/
  ai-runtime/output/
```

モデルは[lock](../config/models.lock.json)と[Turbo合成手順](moss-turbo4-reproduction.md)を使い、元weightsを保持した別ファイルとして作成します。転送前後に全量のsize／SHAを照合します。headerだけの一致は全ファイルidentityの代わりではありません。weightsやprivate receiptをGitへ置きません。

F16 storageは別ファイル・別profileの限定試験です。通常のF32速度基準へ自動置換しません。旧rank16／50-step比較も別設定です。

## 設定と所有権

native SELF、compiled Direct pool／VA、推論予算、対応する設定を一致させます。選択した10GiB版の関連値は`gpu_pool_bytes: 10737418240`、`max_vram_gib: "Vulkan0=9.5"`、`working_reserve_bytes: 1073741824`です。合成F32 LoRAのruntime multiplierは1です。これらは完全なconfigではなく、[モデル構成](moss-models.md)とビルドprofileの他の項目も必要です。JSONだけで予約容量を変えません。

runtime parentはアプリowner専用、config／tokenはownerだけが読める権限にします。既存設定と実行ファイルをprivate backupへ保存し、完成ファイルの全量readback後にatomic renameします。元モデルや他タイトルを上書きしません。native側で所有者フィールドが取得できない環境を、UID0なら安全という緩和で通しません。既存bootstrapの外部owner証拠とFD identity guardを保持します。

nativeとgatewayは同じ認証token値を各process専用の起動ファイルで受け取ります。tokenはGit、ログ、URL、コマンド表示へ出さず、起動後はファイルから削除してmemoryに保持します。APIはBearer認証と同じprivate IPv4 /24を維持します。

## 1回ずつ検証する手順

1. 他のGPUタイトルが動いていないことと、既存boot／service identityを確認します。
2. nativeを1回だけ起動し、元の数値ゲート、rank144 readback、bounded upload gate、物理pool、backend、ownershipを確認します。readyとモデル読込・画像検証は別です。
3. 同じnativeを指すCPU gatewayを起動します。[APIのcanonical request](api.md)をまず256で1回だけ送り、native GPU完了、PNG decode、native／gateway画像一致と目視を確認します。
4. 空queueと同ownerを確認してから同processで1024を1回送ります。seed20261020、4 Euler steps、CFG1、shift5を維持します。有効な生成はprocess全体で最大4試行、失敗も消費します。
5. queueが空でGPU処理が完了してから、認証付き`POST /api/admin/shutdown`をnative／gatewayへそれぞれ1回要求します。両PID不在を別観測で確認し、独立したidle観測で同boot、既存service、タイトル登録、モデルmetadataを確認します。

タイムアウト・結果不明でPOSTを自動再送しません。GPU completionやownerが不明な資源をforce freeしません。kill成功を正常なGPU終了と同義に扱いません。画面heartbeatは表示loopの生存確認であり、推論の進捗そのものではありません。

## 保存量

ログのbounded rotationは[ログ資料](logging.md)の範囲です。成功PNGやjob metadataを恒久的に無制限で保存できる保証、実ディスク空き容量を毎job強制するquota機能は、この公開版の実証済み機能ではありません。保存物を別の場所へ退避し、owner、認証、symlink制約を保って管理してください。長期連続・任意prompt運用の成功を本資料から推定しないでください。

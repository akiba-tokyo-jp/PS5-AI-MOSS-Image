# MOSS Imageの配置・起動・終了

通常版はnativeタイトル`PPSA99190`、native API8192、CPU gateway8193です。native Homebrewを起動できる環境と、このタイトルの登録が必要です。FW12.40で検証しています。一般的なタイトル登録・jailbreak installerは含めません。SELFをWebSrvのELF loaderへ送ってもnativeタイトルの起動にはなりません。

## ファイル配置

タイトルの実体は`/data/ai/native-moss-image-sft/PPSA99190`に配置し、既存native launcherのmount/登録を使います。system領域や他タイトルを上書きしません。

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
  ai-runtime/
    config.json
    config/
    logs/
    jobs/
    output/
```

[通常版config](../config/moss-service.json.example)を`ai-runtime/config.json`へ配置します。[モデルlock](../config/models.lock.json)と[adapter合成](moss-turbo4-reproduction.md)に従い、転送前後の全量size/SHAを確認します。runtime libraryの配布条件は別途確認してください。

物理Direct10GiB、内側予算9.5GiB、pool外reserve1GiB、F32 rank144 adapter/runtime1をcompiled配置と一致させます。旧12GiB/rank16設定を使い回さないでください。実行ファイルと設定をprivate backupへ保存し、転送済み完成ファイルを全量照合してから切り替えます。

## 所有者確認とtokenの受け渡し

nativeはUID1のタイトルとして起動します。`ai-runtime`とそのconfig/logs/jobs/output等のdirectoryを実際のUID1所有・0700、config.jsonを0600にします。symlinkを使わず、既存launcher/管理shellで所有者と実体を確認します。native互換statのowner欄が取得できない場合、UID0という報告を実際のownerだと解釈しません。外部の信頼できる観測でUID1を確かめるbootstrapを保持します。

1. Windows側のprivateファイルに十分な乱数tokenを作り、画面・ログ・URL・引数に出さず保持します。clientへ渡すtokenもこのファイルです。native/gatewayは同じ値を使用します。
2. 他のGPUタイトルが終了済み、同じbootのownerと既存serviceが正常であることを確認します。**nativeの`ai-runtime/config/api-token`が存在しない状態で**登録済みnative launcherからタイトルを1回起動します。
3. 新しく作られたPID/時刻付きbootstrap journalとowner witnessが今回のUID1/PPSA99190 processに属すること、runtime directoryのowner/mode、witnessの全byte identityを外部から照合します。token待機は最大5分で、まだGPU/APIを始めません。古いjournalやPIDだけで確認を省略しません。
4. 確認できたprocess用に、同じtokenの一時ファイルを`ai-runtime/config/api-token`へUID1/0600で渡します。nativeが消費して削除し、memoryに保持します。起動前にtokenを置くとbootstrap条件に合いません。timeoutしたprocessへ繰り返し投入しないでください。
5. 起動時のGPU readback、実pool確保、engine readyを確認します。初回のモデルlazy loadは画像要求時です。

この手順には既存Homebrew環境のprocess/UID/起動・mount管理機能が必要です。個人環境用のPIDやnonce、root controllerを公開installerとして使い回すことはしません。所有者を検証できる管理機能がない場合は、owner guardを外すのでなく、その起動環境に合うprovisionerが必要です。

## CPU gateway

`/data/ai/moss-openai-gateway`にowner専用のconfig/logs/jobs/output等を用意し、`config/api-token`へnativeと同じ値をgateway owner/0600で配置します。gatewayは起動時に読み取って削除します。gatewayはGPUタイトルではなくCPU ELFなので、既存WebSrvの`/elfldr`からこのELFを1回だけ起動できます。

```text
<gateway.elf> --root /data/ai/moss-openai-gateway --bind auto --port 8193
```

bindの既定はautoです。private IPv4を検出できなければ終了するので、`--bind <PS5_PRIVATE_IP>`を明示します。native/gatewayが同じLAN addressを使用する必要があります。gatewayはそのaddressの同じPS5 native8192へだけ転送し、生成POSTを再送しません。既存WebSrv8080は維持します。

両`GET /health`をBearer認証で確認し、`service_mode:true`、`requires_256_first:false`、`max_generation_attempts:null`、ready・空queueを確認します。[client](../scripts/Invoke-MossOpenAI.ps1)から初回1024を直接生成できます。固定promptや先行256の操作はありません。

## 保存量と運用

通常版は各processのoutput＋jobsに既定8GiBの保存枠を設けます。`--storage-limit-mib`で起動時変更可。未完了jobと次jobごとに32MiBを予約計上し、収まらなければ生成前507です。物理空き容量の取得・保証ではありません。gateway側にもPNG copyとmetadataがあるため、両rootを管理してください。input storeと[bounded logs](logging.md)は別枠です。

PNGを自動削除しません。保存枠に近づいたらclientから新規要求を止め、job完了・空queueを確認し、必要なPNG/metadataをPC等へ全量取得してから、停止中に対象rootの古いoutput/job組だけを管理します。rootそのもの、現用token/config、他タイトルのデータを再帰削除しないでください。上限を引き上げる前にストレージ空きを確認します。

## 終了・復旧

処理完了・空queue・GPU完了を確認し、認証付き`POST /api/admin/shutdown`をgateway、nativeへそれぞれ1回送ります。202は終了要求の受理であり、終了済みの証明ではありません。別のprocess観測で両owner不在と同bootのidle、既存serviceを確認します。次回は新しいowner検証と一時token受け渡しを行います。

タイムアウトしても自動再送せず、healthと既存jobを確認します。GPU状態不明・fault retention・終了拒否のときは、force freeやkillを正常終了の代用にしません。既存の信頼できるOS close/復旧経路を使い、別GPUタイトルを重ねません。今回の5件後は正常終了、両owner不在の独立観測、同artifactの再起動と全GPUゲートを確認しました。

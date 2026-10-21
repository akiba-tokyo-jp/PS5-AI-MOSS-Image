# PS5 AI MOSS Image

**MOSS-Image-SFTをPS5本体のGPUで動かすネイティブ画像生成アプリ**です。WindowsからOpenAI Images互換APIで画像生成とimg2imgを要求し、PNGを取得できます。WindowsはビルドとHTTPクライアントを担当し、モデル推論はすべてPS5内で実行します。

公開版を通常のAPI利用向けに更新しました。**初回256の前処理、固定prompt・seed、1起動4回の制限はありません。** 初回から1024を生成でき、長方形、任意の文章・seed、ランダムseed、参照画像を指定できます。[今回の実機検証](docs/moss-service-20261021.md)。

![PS5で512×768生成した出力](docs/images/moss-service-aspect-512x768-20261021.png)

## APIで生成する

登録・起動済みのPS5アプリとgateway、秘密のtokenファイルを用意します。[導入と起動](docs/deployment.md)、[ソースからのビルド](docs/build.md)。PS5のprivate IPv4とtokenのファイル名を自分の環境に置き換えてください。

```powershell
./scripts/Invoke-MossOpenAI.ps1 -BaseUrl http://<PS5_IP>:8193/v1 -TokenFile D:/ps5-private/api-token -Prompt 'An anime illustration of a peaceful garden.' -Width 1024 -Height 1024 -OutputDirectory D:/moss-output
```

`-Seed`省略時はランダムです。`-NegativePrompt`は既定空文字で、人物や画風の指定を自動で付け足しません。参照画像は同じclientに`-ReferenceImage D:/images/reference.png -DenoiseStrength 0.5`を追加します。256を先に送る必要はありません。

出力は各辺64～1024、16の倍数。現在のモデル構成は**Turbo4：Euler 4 steps／CFG1／専用sigma列**です。これは検証回数制限ではなく、同梱する設定のTurbo adapter用samplingです。任意stepsや別samplerを選ぶ機能は含めません。[API仕様と例](docs/api.md)、[OpenAPI](docs/openapi.json)。

## 今回の実機結果

FW12.40／RADV NAVI21。以下は同じnative起動の5件で、公開PowerShell clientを使い、PS5内の生成時間と取得PNGを照合しました。

| 順序 | 要求 | PS5内の生成時間 |
|---|---|---:|
| 1 | 起動後最初の1024×1024・モデル読込込み | 110.46秒 |
| 2 | 別prompt／seed17、512×768 | 42.18秒 |
| 3 | img2img 512×512、strength0.5・実3 forward | 35.47秒 |
| 4 | img2img 256×256、strength1・ランダムseed | 31.13秒 |
| 5 | text-to-image 256×256、ランダムseed | 3.51秒 |

5件目も再起動なしで成功しました。各要求のprompt・解像度・encode・conditioning cacheの状態が異なり、相互の速度比較ではありません。5件の成功は長期連続運転や全解像度・全promptのメモリ適合を保証しません。

従来の同条件1024比較では、DiT kernel改善で平均47.481390秒→45.127473秒（4.96%短縮）でした。この値は256を先に生成した測定で、上記のcold1024とは区別します。[速度比較と画像差](docs/moss-dit-kernels-20261021.md)。

## PS5内の構成

![処理構成](docs/images/architecture.svg)

- OpenAI Images互換gateway：8193、model `moss-image-sft`。同じPS5のnative非同期API 8192に1件渡します。
- Qwen3.5-9B Q4_K_M text encoding：PS5 CPU。MOSS Q4_K DiTとFLUX.2 small VAE：PS5 Vulkan／RADV。img2imgのVAE encodeもPS5 GPUです。
- Direct予約10GiB、推論予算9.5GiB。Q4 weights、FP16データ経路と明示的F32累積、合成rank144 F32 adapterを使用します。
- Bearer認証、private LANの同じ/24内、GPU生成の直列処理を維持します。初期の数値読み戻し、有限値、実容量、所有権、GPU完了の検査を残します。
- output＋job metadataに既定8GiBの保存枠を設け、上限前に507を返します。生成物は自動削除しません。ログは既存のbounded rotationです。

マスク、inpainting、IP-Adapter、ControlNet、任意LoRAのuploadは未対応です。img2imgは人物固定を保証しません。強い実写化も未達です。共有メモリではRSS、Direct、API GPUラベル付きピークが重複するため、足し算できません。

## 導入・資料

[ビルド](docs/build.md)／[配置・起動・終了](docs/deployment.md)／[API](docs/api.md)／[現在の状態](docs/status.md)／[検証データ](docs/verification.json)。[モデルと利用条件](docs/moss-models.md)、[Turbo adapter再現](docs/moss-turbo4-reproduction.md)、[第三者通知](THIRD_PARTY_NOTICES.md)も確認してください。

PS5のnative Homebrew起動環境、登録済みタイトル、SDK・依存library・モデルを別途用意するソース公開です。一般向けの自動jailbreak／タイトル登録installerやビルド済みbinaryは含めません。通常版の入口は`tools/build_service.py`と`config/moss-service.json.example`です。過去のtrial資料の回数制限は、当時の診断buildについての記録です。

[OpenMOSS](https://github.com/OpenMOSS/MOSS-Image)、[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)、PS5向けVulkan／Mesa／SDK等を接続しています。各ファイルのlicense・copyrightと、モデル独自の利用条件を保持してください。

# PS5 AI MOSS Image

MOSS-Image-SFTとユーザー指定LoRAをPS5本体で推論するネイティブアプリと、LAN用のOpenAI Images互換APIを開発しています。

**2026-10-20、MOSS-Image-SFT＋[選択LoRA](docs/moss-requested-lora.md)で、PS5のCPU text encoding・Vulkan DiT/VAE・WindowsからのOpenAI互換API経由PNG取得まで、1024×1024・50stepsを確認しました。PS5内の生成時間は初回読込込み1242.706350秒（20分42.7秒）、同一process2回目1177.725365秒（19分37.7秒）です。両画像は正常な画質で、PNGも完全一致しました。** [実測条件と時間の範囲](docs/moss-1024-cold-warm-20261020.md)を参照してください。

**Privateの実験版です。** 同一prompt・seed20261020・LoRA0.7・CFG3・flow shift5・Eulerを固定し、各phase1POST、自動再送0で測定しました。初回のOS disk cacheはflushしておらず、warmでも必要なweight転送ゼロとは扱いません。10GiB poolの専用trialで、12GiB目標は未達、production admission0を維持します。平均／分散・長時間連続運用・最小必要stepsは未検証です。両APIの通常終了、両PID不在、独立idle確認も完了しました。

![PS5で生成した1024×1024・50stepsのMOSS画像](docs/assets/moss-1024-20261020.png)

以前の256×256比較では4stepsの色縞が50stepsで解消しました。通常のAPI既定値は元から50で、4は初回smoke限定値です。[256×256の原因比較](docs/moss-native-small-trial-20261019.md)を履歴として保持します。

```mermaid
flowchart LR
  PC[Windows API client] -->|LAN / Bearer / 8193| GW[PS5 MOSS gateway]
  GW -->|one job / 8192| API[PS5 native engine]
  API --> TE[Qwen3.5-9B / PS5 CPU]
  TE --> DIT[MOSS-Image-SFT DiT / Vulkan]
  LORA[Selected rank16 LoRA / exact F32] -->|runtime scale 0.7| DIT
  DIT --> VAE[FLUX.2 small VAE / Vulkan]
  VAE --> PNG[PS5 PNG output]
  PNG --> PC
```

Windowsは開発・APIクライアントに使います。モデル推論をPCやクラウドへ移しません。WAI・Anima・MOSSは同時起動しません。

| 項目 | 現在の構成 |
|---|---|
| アプリ / Title ID | PS5 AI MOSS Image / PPSA99190（登録・実機起動確認） |
| 配置 | `/data/ai/native-moss-image-sft/PPSA99190` |
| Native / OpenAI gateway | 8192 / 8193、model `moss-image-sft` |
| モデル | SFT Q4_K + Qwen3.5-9B Q4_K_M + FLUX.2 small VAE + 外部tokenizer + 選定rank16 LoRA |
| Direct予約目標 / 実測trial | **12GiBは未達** / 承認済み実験10GiB、1GiB regions |
| 推論予算 | production例11.5GiB / 実験9.5GiB、pool外working reserve 1GiB |
| 最新の実測 | 1024×1024・50steps、初回1242.706350秒／同process2回目1177.725365秒。Euler、CFG3、shift5、LoRA0.7 |

12GiBはGPU専用の独立VRAMではなく、CPU/GPU共有のDirect領域の目標です。今回のnative contextの総量報告は11.4375GiBで、pool12GiB + reserve1GiBの容量判定を通りません。仮想アドレスの区切り変更では物理メモリ枠は増えません。今回の1024画像2枚の成功から、12GiB確保や他条件・長時間運用の安全性は推定しません。APIの`peak_gpu_bytes`もCPU text weightsを含むowned poolピークの合計で、純粋なVRAM実使用量ではありません。

Qwenのembeddingは実GGUFヘッダーで545.625MiBと確認しました。旧512MiB単体割当上限を超えるため、MOSS側のregionを1GiBへ拡張しています。今回の実機ではCPU text encoding、BF16を含む限定GPU読み戻しとDiT/VAE完了を確認しましたが、全演算の数値正しさと画質を証明するものではありません。SDXL用LoRA、ControlNet、IP-Adapter、img2img/inpaintingはMOSSへ流用せず、APIで拒否します。

- [現在の検証範囲](docs/status.md)
- [固定モデル・ハッシュ・メモリ・ライセンス](docs/moss-models.md)
- [指定LoRAの実ヘッダー・メモリ・互換対応](docs/moss-requested-lora.md)
- [APIとOpenAPI](docs/api.md)
- [設定例](config/moss-native.json.example)
- [ビルド](docs/build.md) / [配置計画](docs/deployment.md)

[OpenMOSS公式](https://github.com/OpenMOSS/MOSS-Image)と[SD.cppのMOSS対応](https://github.com/leejet/stable-diffusion.cpp/blob/b13a48219d5cdf408edcb93da6cd69b612ed03d3/docs/moss_image.md)を参照しています。実装は既存PS5向けSD.cppへMOSS対応commitを範囲限定で移植し、選定LoRAを含むcore全75オブジェクトを新規再構築しました。最新upstream全体への置換ではありません。

このリポジトリは[PS5 AI WAI](https://github.com/akiba-tokyo-jp/PS5-AI-WAI)とPS5 AI Animaの履歴を継承します。`docs/`に残るWAI/Animaの成功画像・ベンチマーク・検証結果はMOSSの結果ではありません。[Anima元README](ANIMA-BASE-README.md)、[WAI元README](WAI-BASE-README.md)、著作権・[第三者通知](THIRD_PARTY_NOTICES.md)を保持します。ホームアイコンは継承した仮アセットです。

重み・token・個人ログ・ビルド済みバイナリはGitへ含めません。正常な検証出力PNGだけを同梱し、今回の同一画像は1枚だけ保存しています。アプリのGPL等と各モデルのApache-2.0条件は別に扱います。公開の別指示まではprivateを維持します。

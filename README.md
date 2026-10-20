# PS5 AI MOSS Image

**MOSS-Image-SFTをPS5本体で動かすネイティブ画像生成アプリ**です。WindowsからOpenAI Images互換APIで要求し、同じPS5のGPUで生成したPNGを取得できます。Windowsは開発・APIクライアントとして使い、モデル推論はPS5内で行います。

最新の検証済み速度基準は **Turbo4＋bounded upload＋DiT不変計算再利用・F32融合**です。256×256の初回モデル読込込みは **69.275742秒**、同じプロセスでの256後1024×1024初回は **47.760891秒**でした。同じcoreの最適化なし51.146425秒から約6.62%短縮し、両サイズでPNGが完全一致しました。4 Euler steps／CFG1、作者の専用sigma列、NSFW0.7＋Turbo1.0の合成rank144 adapterを使います。[最新のDiT比較・メモリ・確認範囲](docs/moss-dit-fusion-20261020.md)、[固定weightsの再現手順](docs/moss-turbo4-reproduction.md)。

![PS5のTurbo4＋bounded uploadで生成した1024×1024画像](docs/images/moss-turbo4-upload-1024-20261020.png)

## 生成時間

2026-10-20、各構成・各サイズ1回のPS5内計測です。右列は同promptの256生成後の1024で、1024単独cold／2回目の1024ではありません。

| 構成 | 256初回、読込込み | 同じプロセスでの256後1024初回 |
|---|---:|---:|
| 50-step Euler／CFG3、F32累積＋text cache | 146.016289秒 | 1023.859406秒 |
| 30-step res_multistep／CFG4 | 113.804763秒 | 617.365825秒 |
| 50-step Euler／CFG3＋Spectrum latent予測 | 114.148733秒 | 617.938351秒 |
| 従来Turbo4＋bounded upload、F32 LoRA | 71.377746秒 | 51.092988秒 |
| 今回の同core対照版、DiT最適化OFF | 71.354991秒 | 51.146425秒 |
| **上記＋DiT不変計算再利用・F32融合** | **69.275742秒** | **47.760891秒** |
| 従来Turbo4＋upload＋FFN chunk2048 | 71.294669秒 | 55.195208秒 |
| F16 LoRA storage＋専用F32 shared演算、FFN分割OFF | 68.897951秒 | 51.884730秒 |

今回のDiT最適化ON/OFFは同じsampling・adapterで、256／1024ともPNGが全バイト一致しました。他のsamplingやadapterが異なる行は同画質比較ではありません。温度・clockは統制せず、平均・分散・長時間安定性は未確定です。FFN分割は読込済み1024が約8%遅くなったため通常の速度基準ではOFF。F16 storageはowned Direct使用量を637.875MiB下げましたが、読込済み1024の高速化は確認できず、任意prompt向け標準経路にしていません。[前段のsampling比較](docs/moss-sampler-spectrum-memory-20261020.md)。

**現在公開しているのは検証用アプリです。** 採用候補のTurbo4プロファイルは固定prompt・seed・256／1024、1プロセス最大4試行に制限しています。任意promptの連続運用版は公開対象に含めません。[確認済み機能と制約](docs/status.md)を先に確認してください。

## PS5内の処理

![MOSS専用の処理構成](docs/images/architecture.svg)

- OpenAI互換gateway：PS5の8193、model `moss-image-sft`。
- 非同期native engine：同じPS5の8192。gatewayはジョブを1件だけ渡します。
- Qwen3.5-9B Q4_K_Mのtext encoding：PS5 CPU。
- MOSS-Image-SFT Q4_KのDiTとFLUX.2 small VAE：PS5 Vulkan、RADV NAVI21。
- FP16データ経路と明示的FP32累積、選定LoRAの数値・有限値・GPU完了確認を維持します。
- text-to-image対応。ControlNet、IP-Adapter、img2img、マスク編集、inpaintingはこのMOSS APIでは未対応です。

## 動作・メモリの境界

アプリはPPSA99190として登録・実機起動を確認しました。APIはBearer認証と同じprivate IPv4 /24内の接続に制限し、生成要求を直列処理します。タイムアウト時は既存jobとhealthを確認し、自動再送しません。GPUアプリは同時起動しません。

検証済み速度基準のDirect予約は10GiB、推論予算9.5GiB、pool外reserve1GiBです。CPU/GPUは物理メモリを共有し、APIの`peak_gpu_bytes`は専用VRAMの実使用量ではありません。12GiB予約目標は現native quotaでは未達です。workspaceやstorageの使用量低下と、予約領域をOSへ返すことを区別します。

bounded uploadは同形式のGPU重み転送用host stagingをworkerあたり最大8MiBへ制限します。これは全推論RAMを8MiBにする機能やMoE expert cacheではありません。モデルはdenseです。既存14数値ゲート、Turbo用3ゲート、upload用ゲートを実機で確認し、画像ごとに4つの有限DiT出力、672 Q4 F32記録、LoRA504 tensorsの適用とPNGを確認しています。確認済み条件以外の安全性・同画質を保証するものではありません。

## 利用・ビルド

- [APIとOpenAPI](docs/api.md) / [PowerShellクライアント](scripts/Invoke-MossOpenAI.ps1)
- [モデル・ハッシュ・ライセンス](docs/moss-models.md) / [Turbo4合成adapterの再現](docs/moss-turbo4-reproduction.md)
- [ビルド](docs/build.md) / [配置と終了](docs/deployment.md)
- [現在の検証範囲](docs/status.md) / [最新のDiT比較](docs/moss-dit-fusion-20261020.md) / [前段のメモリ比較](docs/moss-turbo-memory-options-20261020.md)
- [公開前の確認](docs/publication-checks-20261020.md) / [実機検証データ](docs/verification.json)

既定のproduction admissionはOFFです。各実験は明示的profileで区別し、対応するソース・設定・compiled poolを一致させます。SDK・生成済みshader・reviewed依存が別途必要で、clone直後のワンコマンド完全構築ではありません。モデルweights、token、個人ログ、ビルド済みbinaryはGitに含めません。掲載PNGは検証済みの正常出力です。

[OpenMOSS](https://github.com/OpenMOSS/MOSS-Image)、[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)、PS5向けVulkan／Mesa／SDK等を接続しています。アプリのライセンスと各モデルの利用条件は別です。[第三者通知](THIRD_PARTY_NOTICES.md)とファイルごとのlicense・copyrightを保持してください。

# ControlNet Canny / OpenPose 実機検証（2026-10-18）

現行の9422MiB予約・CPU CLIP分割・ControlNetバッチ化と1024の結果は[2026-10-19の検証](optimization-20261019.md)を参照してください。以下の9GiBや時間は当時の実測です。

FW12.40のPS5で、WAI v17.0にCanny・OpenPoseを接続し、両方の256×256 /1024×1024生成とWindowsからのPNG取得を確認しました。各条件1枚の機能検証です。長期安定性や画質最適化の完了を意味しません。

| 条件 | 漫画LoRA強度 | PS5内生成 | Windows API応答 |
|---|---:|---:|---:|
| canny256 | 0 | 74.449秒 | 75.672秒 |
| canny1024 | 1 | 162.557秒 | 164.922秒 |
| openpose256 | 0 | 43.146秒 | 43.531秒 |
| openpose1024 | 1 | 329.682秒 | 333.125秒 |

全条件20steps、CFG5、Euler A、seed20261018、control_strength1。UNetとControlNetとTAESDXLはVulkan、CLIPはCPU。FP32演算とbatched CFGを維持しました。1024はLoRA強度1、小型検証は0です。異なるサイズ・LoRA・初回モデル/シェーダー状態が混在するため、この表から高速化率は算出しません。

各画像で20回の有限UNet出力、40回のGPU ControlNet評価、10残差テンソル、NaN/Inf0を確認しました。ControlNet残差は実際のUNetへ渡しています。公式OpenAI SDK（max_retries0）から8189へ1回だけPOSTし、同一PS5の8188が生成。返却PNGはnative取得分と全バイト一致し、実画像も目視しました。Windowsではモデル推論を行っていません。

## メモリと修正

Directは総量9GiB、18領域×512MiB、推論予算8.75GiBです。1024のUNetが必要とした464,230,404Bの作業バッファに合わせ、1領域256MiBの上限を拡大しました。総量とGPUドライバーは変更していません。起動時には64MiBのGPU計算を読み戻し、16,777,216個の値の不一致0を確認しました。9GiB全域のGPU検査ではありません。

CPU CLIPは逐次ロードし、一時重みを既存Directプールから借用するCPU_Mappedバッファを使います。LoRAを含むCPUパラメーター群も対象とし、CPU実行のままです。新規の物理Direct確保やVulkanへのCPU重みimportは行いません。全条件で借用の確保・返却件数とバイト数が一致し、生成後の借用量0を確認しました。

`peak_gpu_bytes`は所有プールの報告ピークの合計で、CPU借用分も含み、プロセス開始後のピークを引き継ぎます。GPU専用VRAM量や各ジョブ単独の必要量ではありません。`peak_cpu_bytes`もプロセスpeak RSSで共有GPUマッピングを含み得るため、両者を合算しません。Directのreported sizeとavailable spanも、現在の総空き容量ではありません。GPU使用率・温度は取得していません。

起動時の保存済み入力検査は、実機で問題があったstdio経路をbounded raw FD読み取りへ変更しました。同じ保存済み入力を再起動後に回収し、全SHA256一致を確認しています。stdio内部のどの処理が原因だったかまでは確定していません。

## 入力と品質の範囲

Cannyには完成済みedge map、OpenPoseには完成済みpose mapを指定します。写真からOpenPoseを自動検出する機能はありません。入力と出力の寸法は一致させます。256のOpenPoseは構図確認用で、顔の細部が粗く、上端の切れがあります。1024も参照ポーズの厳密な関節一致や最終作品品質を保証するものではありません。TAESDXL復号であり、フルVAEと同じ品質という意味ではありません。

img2img / IP-Adapterの既存経路は維持していますが、以前の小型画像で画質の課題があり、今回のControlNet検証で人物同一性や品質が確認されたとは扱いません。マスク付きinpainting候補は現行アプリへ未反映です。

## 使用例

1. 認証付き`POST http://<PS5_IP>:8188/api/inputs`へPNG/JPEGの完成済みmapを送る。
2. 返された`image_id`をOpenAI互換`POST :8189/v1/images/generations`の`control_image_id`へ渡す。
3. `controlnet`は`canny`または`openpose`、`control_strength`は0.01〜1.5（既定1）。これらはWAI独自拡張です。

`scripts/Invoke-WaiControlNet.ps1`はアップロード、1回の生成、PNG保存を行います。タイムアウト時は自動再送せず、healthとnative job状態を確認してください。トークンはファイルで渡します。

## モデルとライセンス

重みはリポジトリに同梱しません。両配布元のmodel cardはApache-2.0表記です。出典カードと全文ライセンスを`licenses/controlnet/`に保存しました。

- [MIC-Lab Illustrious Canny](https://huggingface.co/MIC-Lab/illustriousXLv1.1_controlnet/tree/4fe9d46fadfad2ab46150749b89561b8cb998d38)：`illustriousXLv1.1_canny_fp16.safetensors`を`controlnet-illustrious-canny-f16.safetensors`へ配置。SHA256 `3255c963a219a20040d2bef913216dd03d6da958068f5e69548dfe6e66528bcd`。
- [xinsir SDXL OpenPose](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0/tree/23f966cd5cfdd3f7729c903e243d87152162d2b7)：`diffusion_pytorch_model.safetensors`を`controlnet-xinsir-openpose-sdxl-f16.safetensors`へ配置。SHA256 `b8524e557a7df60d081f5d4a0eb109967d107df217943bf88c2d99b9ebcc06c5`。

各2,502,139,104B。PS5へ転送後も全ファイルのSHA256を確認しました。WAI/漫画LoRA/IP-Adapter等はそれぞれの条件も確認してください。

## 工程別の実測（2026-10-18の記録）

2026-10-19には[UNet・転送・同期の計測とログ修正](unet-profiling.md)を実施し、同条件OpenPose初回194.54→184.03秒、PNG完全一致を確認しました。下記は変更前の記録です。

ControlNet時間はsampling時間に含まれます。二重加算しないでください。

Canny1024のsampling132.95秒に対し、OpenPose1024は296.24秒でした。CLIPは24.52/24.51秒、ControlNet評価合計は34.47/40.81秒、ログに記録されたtensorロード6件の合計は16.95/16.93秒です。遅延差の大部分はsampling内のControlNet以外にあります。UNet計算・転送・同期・メモリ管理の時間を個別分離できておらず、熱やログが原因とは確定していません。速度改善完了とは扱いません。

### canny256

- get_learned_condition completed, taking 21.73s --- image.cpp:547
- sampling completed, taking 51.26s --- image.cpp:918
- decode_first_stage completed, taking 0.10s --- image.cpp:642
- generate_image completed in 73.69s --- image.cpp:1069

### canny1024

- get_learned_condition completed, taking 24.52s --- image.cpp:547
- sampling completed, taking 132.95s --- image.cpp:918
- decode_first_stage completed, taking 1.56s --- image.cpp:642
- generate_image completed in 162.12s --- image.cpp:1069

### openpose256

- get_learned_condition completed, taking 21.77s --- image.cpp:547
- sampling completed, taking 20.59s --- image.cpp:918
- decode_first_stage completed, taking 0.05s --- image.cpp:642
- generate_image completed in 42.56s --- image.cpp:1069

### openpose1024

- get_learned_condition completed, taking 24.51s --- image.cpp:547
- sampling completed, taking 296.24s --- image.cpp:918
- decode_first_stage completed, taking 4.82s --- image.cpp:642
- generate_image completed in 329.56s --- image.cpp:1069


公開ソースの再ビルドとSELF整合性検査、24初期化スロット一致、APIの40項目とPowerShellクライアント試験も成功しました。公開用に再ビルドした別SELFそのものはPS5へ再配備していません。機械可読値は`controlnet-verification.json`、実機ログは私有Dドライブに保持しています。失敗動画や秘密トークンは公開しません。

# Anima Turbo v1.1モデル

正規配布元：[CircleStone Labs / Anima](https://huggingface.co/circlestone-labs/Anima)。取得revision：`f973fc41ec7545364ac9776c2440285f43ff2a30`。

| 公式ファイル | bytes | SHA256 |
|---|---:|---|
| anima-turbo-v1.1.safetensors | 4182230656 | fba11953276b57edf59d1dc4f1857ac05aa079c56f982b4d7c20298d57d3f7eb |
| qwen_3_06b_base.safetensors | 1192135096 | cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba |
| qwen_image_vae.safetensors | 253806246 | a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f |

2026-10-19に全3ファイルのサイズと全内容SHA256を照合しました。全tensorはBF16です。GPU向けDiT/VAEはF16保存形式に変換して使用する候補で、変換後のファイル名・SHAは公式BF16のものと区別します。CPU用Qwenは元ファイルを使います。重み形式F16とGPU演算精度FP16は別であり、初期アプリの演算精度は既存のFP32設定を引き継ぎます。

AnimaはCosmos系DiTとQwen3-0.6B、Qwen-Image VAEを使用します。SDXL checkpointやTAESDXLに置き換えることはできません。SD.cppの既存Anima実装と内蔵Qwen/T5 tokenizerを利用し、T5 encoder重みは読み込みません。[上流Anima手順](https://github.com/leejet/stable-diffusion.cpp/blob/8cd731e7b4667eb8f40a153315b9a7125d09a58d/docs/anima.md)。

Turbo公式目安はCFG1、8–12ステップです。最初の実機検証では512×512、8steps、Euler、CFG1を指定します。実測前の速度やメモリ量は保証しません。

利用条件は[取得したモデルLICENSE](../licenses/Anima-model-LICENSE.md)を確認してください。CircleStone Labs Non-Commercial License v1.2と由来モデルの条項が適用されます。アプリのGPLライセンスをモデル重みに適用しません。重みはGitHubリポジトリに同梱しません。

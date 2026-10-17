# モデル

重みは配布しません。コードのGPL/MITとモデルの利用条件は別です。
WAIの配布元・該当バージョン・利用条件を確認して取得してください。

| 対象 | 検証したファイル | bytes |
|---|---|---:|
| WAI-Illustrious-SDXL v17.0 | waiIllustriousSDXL_v170.safetensors | 6,938,040,682 |
| 近似VAE | taesdxl-official-f32.safetensors | 9,793,292 |

WAI SHA256:

```text
f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04
```

TAESDXL SHA256:

```text
ff4824aca94dd6111e0340fa749347fb74101060d9712cb5ef1ca8f1cf17502f
```

[WAI0731の正規ページ](https://civitai.com/models/827184?modelVersionId=2883731)と
[該当バージョンAPI](https://civitai.com/api/v1/model-versions/2883731)のファイルSHAを照合しています。
CheckpointにはF16のUNet、VAE、CLIP-L/G重みが含まれます。確認したSafetensorsヘッダーは2,515 tensorsでした。
Tokenizer処理・定数はこの版のstable-diffusion.cpp側を使います。追加の別Text Encoderファイルは今回の構成では使っていません。

今回の出力はフルVAEを使用せず、[madebyollin/taesd](https://github.com/madebyollin/taesd)のTAESDXLを使います。
検証ファイルはF32版で、公式配布のF16ファイルを単に改名したものではありません。
重みの取得・変換とハッシュ照合はビルドと別工程です。

Illustrious系列には[Fair AI Public License 1.0-SD](https://freedevproject.org/faipl-1.0-sd/)が関連します。
作者の該当モデルページにある個別の許可・条件も確認してください。
ミラーの`cdla-permissive-2.0`というラベルをWAIの利用条件として採用していません。
コードの公開はモデル本体の再配布や無条件の商用利用を許可するものではありません。

元モデルは `/data/ai/models/`、ネイティブから参照する検証済みコピーはタイトル内の `/app0/models/` です。
`config/native.json.example`に実際のエンジン設定を示します。

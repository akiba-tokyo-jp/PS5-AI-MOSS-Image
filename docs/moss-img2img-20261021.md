# MOSS img2imgのPS5実機検証

2026-10-21。**参照画像をVAEで初期latentへ変換するimg2imgを実装し、256と1024でPS5生成・WindowsでのPNG取得を確認しました。** native非同期APIとOpenAI Images形式のmultipart編集APIを接続しています。追加の学習済みモデルは不要で、既存small VAEのencoderを使用します。

![参照画像を使った1024出力](images/moss-img2img-1024-3step-20261021.png)

[参照した元のアニメ画像](images/moss-dit-kernels-1024-20261021.png)。銀髪・赤いジャケット・星空・構図を反映しました。photorealisticを指定しても今回の1024はアニメ調が強く残っており、実写化の成功や人物同一性の保証は主張しません。IP-Adapterの人物固定とは別の初期画像conditioningです。

## 実測

元画像は1024×1024 RGB PNG。新しいprocessで通常256を生成してから、下表を上から実行しました。PS5内のgeneration_secondsで、各1観測、モデル読込済み、入力のVAE encodeと出力decodeを含みます。request.stepsは4ですが、img2imgは作者scheduleの末尾だけを実行します。

| 出力 | denoise_strength | 実DiT forward | 生成時間 | うち入力encode | 経路 |
|---|---:|---:|---:|---:|---|
| 256×256 | 0.25 | 2 | 3.454672秒 | 0.862秒 | native非同期 |
| 256×256 | 0.10 | 1 | 1.816666秒 | 0.093秒 | native非同期 |
| 1024×1024 | 0.50 | 3 | 40.833333秒 | 4.425秒 | OpenAI形式edits |

1024のHTTP全体は43.875秒でした。入力encode初回と再利用後では準備状態が異なり、256の2例はstep数も異なります。text-to-imageの4 forwardに対する同一処理の高速化比較にはしません。先行した通常256はモデル読込込み70.140223秒で、変更前の同条件PNGと全byte一致しました。

![256・2forward](images/moss-img2img-256-2step-20261021.png)
![256・1forward](images/moss-img2img-256-1step-20261021.png)

## 強度の意味と現在の境界

両buildで明示的に`--moss-img2img-trial`を指定します。未指定では画像入力を拒否し、既存text-to-imageの4 forwardを維持します。現在の検証は固定prompt・seed20261020、256／1024正方形、1process最大4試行（失敗を含む）、production admission0です。

`denoise_strength`は連続したnoise量ではなく、既存SD.cppの`floor(4 * float(strength)) + 1`によるstep選択です。doubleとfloatの両方で下限0.01以上、floatで0.75未満を受け付け、境界近くはfloat丸めに従います。

| float(strength) | 実forward | 保持するsigma列 | 初期latentの係数 |
|---|---:|---|---:|
| [0.01,0.25) | 1 | [0.024,0] | 0.976 |
| [0.25,0.50) | 2 | [0.717,0.024,0] | 0.283 |
| [0.50,0.75) | 3 | [0.910,0.717,0.024,0] | 0.090 |

係数はflow noise_scalingの`(1-sigma)*latent + sigma*noise`です。0.75以上はsigma1から始まり参照が消えるため、このtrialは拒否します。強度0.5と0.7は同じsuffixになります。sample plannerが報告する実step数、実有限DiT出力、168×実forwardのQ4 F32記録を独立に照合しました。initial_sigmaのログ値は固定scheduleからの予測値で、GPUでsigmaを直接観測した値ではありません。

今回の成功範囲は通常のimg2imgです。マスク、inpainting、IP-Adapter、ControlNet、複数参照、任意promptでの長期運用は未対応です。1〜3 forwardと256／1024をそれぞれ上表の組合せで検証したもので、全強度・全サイズの直積を検証した意味ではありません。

## 処理・数値・容量の確認

- source uploadをPS5上でRGB正規化し、job所有のleaseで固定。実推論直前にheader＋寸法＋RGBのSHA256を再計算して照合しました。クライアントから任意file pathは渡せません。
- PS5 Vulkanのencoder重みを使う完了witness、有限VAE encode/decode、実DiT、DiT内Attention、decoderの完了を確認しました。witnessはsampled operationで、全GPU命令数ではありません。
- 通常4／画像2／画像1／画像3の有限DiT出力、Q4 F32記録672／336／168／504、F16累積0、invalid0、NaN等0、適用LoRA504／skip0を確認しました。元の14＋Turbo3＋upload＋融合16＋kernel20の数値ゲートを維持しました。
- native uploadの入力はjob完了後にreference count0を確認して削除。gatewayはedits用一時inputを完了後に削除する既存経路を使用します。1024のbase64 PNGとnative GETのPNGは全byte一致、完全decode・目視も確認しました。
- 1024 img2imgのowned Direct高水位は6,277,365,760B（約5.85GiB）、生成後4,920,754,176B。物理予約10GiB、推論予算9.5GiB、pool外reserve1GiBは変更していません。API GPU-labelled peak6,308,823,040B、process RSS高水位11,629,019,136Bとは重複があり、足し合わせられません。
- host強度契約7412、actual native/gateway parserのflag OFF/ON148、既定API回帰134、multipart拒否4。PS5では未対応mask/IPや不正画像・強度・重複input等10ケースをモデル読込・生成前に拒否しました。host検査はPS5推論を代替しません。

rawログ、token、個人path/PID、weights、実行binaryは非公開です。[集約JSON](moss-img2img-20261021.json)、[API例](api.md)、[ビルド](build.md)、[全検証一覧](verification.json)。通常終了・独立したprocess不在確認後、同じ検証済みnativeとgatewayを再起動し、空queue・4試行の未使用状態で待機しています。再起動後はモデル未読込のため、次の要求にはlazy loadが加わります。FW・システム領域・GPU driver・既存登録は変更していません。

## 前段の画像と速度検証

img2imgは入力encodeとstep数が異なるため、[従来のkernel速度比較](moss-dit-kernels-20261021.md)を置き換えません。履歴の正常出力も保持しています：[moss-turbo4-upload-256-20261020](images/moss-turbo4-upload-256-20261020.png) / [moss-turbo4-upload-1024-20261020](images/moss-turbo4-upload-1024-20261020.png) / [moss-dit-kernels-256-20261021](images/moss-dit-kernels-256-20261021.png) / [moss-dit-kernels-1024-20261021](images/moss-dit-kernels-1024-20261021.png)。

測定後、gatewayのOpenAPI説明にnegative_promptと入力寸法の記載を補完しました。推論・転送処理のソースとcompile flagsは測定版と同一で、差分はschema JSONと埋込schemaだけです。gatewayを再ビルド・再起動し、実機GETのschema一致、既定API回帰134、実機拒否10ケースを確認しました。説明修正後に画像生成を追加測定したという主張ではありません。

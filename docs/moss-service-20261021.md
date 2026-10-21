# 通常API版のPS5実機検証 — 2026-10-21

ユーザーの指示により、検証専用の「256を先に1回」「固定prompt/negative/seed」「native1起動最大4試行」を通常版から撤去しました。任意の文章とseed、ランダムseed、長方形、初回1024、img2imgを受理します。各辺64～1024・16刻みの受理範囲を設け、実容量や数値の検査は維持します。

Turbo4のEuler4／CFG1／flow5／sigmas[1,.910,.717,.024,0]、NSFW0.7＋Turbo1 rank144 F32/runtime1はモデル構成として明示しています。別samplingや任意LoRAの機能ではありません。通常版では`service_mode:true`、`requires_256_first:false`、`max_generation_attempts:null`です。継承した低水準のproduction admission macro0を、通常APIの受理状態と混同しません。

## 実際に送った5件

FW12.40、RADV NAVI21、Direct10GiB／推論予算9.5GiB。同じnative owner・gatewayで公開PowerShell clientを使用し、生成POSTを各1回だけ送信しました。モデルは最初の要求でlazy loadしています。数値はPS5内部の`generation_seconds`です。

| 順序 | 要求 | seed指定 | 実forward | 秒 |
|---|---|---|---:|---:|
| 1 | text-to-image 1024×1024、先行256なし・読込込み | 20261020 | 4 | 110.459715 |
| 2 | 別文章・negative空、512×768 | 17 | 4 | 42.180063 |
| 3 | img2img 512×512、strength0.5 | 12345 | 3 | 35.473797 |
| 4 | img2img 256×256、strength1 | -1→新規選択 | 4 | 31.132021 |
| 5 | text-to-image 256×256、4と同じ庭の文章 | -1→別seed | 4 | 3.513742 |

5件目も再起動せずに成功し、4回capの撤去を実機で確認しました。最初の1024 PNGは以前の採用kernel版の同条件PNGと全byte一致しています。任意seedの例、ランダム2件の解決済みseed、実際のprompt/negative、PNG size/SHA、各指標は[JSON](moss-service-20261021.json)へ記録しました。入力のimg2imgは既存の公開benchmark PNGを用い、個人の添付画像は含めません。

![任意文章・別seed・長方形](images/moss-service-aspect-512x768-20261021.png)

![512 img2img・strength0.5](images/moss-service-edit-512-20261021.png)

![256 img2img・strength1](images/moss-service-edit-full-256-20261021.png)

![同一起動5件目・ランダムseed](images/moss-service-fifth-256-20261021.png)

img2imgの`An ink drawing.`要求でも強い墨画化は示せず、アニメ調が残りました。img2img経路の機械的な動作確認と、画風転換の達成は別です。strength1はsigma1により参照latentの寄与が0になるため、庭の再サンプリングになっています。人物固定や実写化の成功を主張しません。

## GPU・メモリ・終了の確認

元14＋Turbo3＋bounded upload＋fusion16＋kernel20の起動時GPU読み戻しを保持し、全て合格しました。各5件でGPU DiT/attention/decoder完了、有限DiT出力4/4/3/4/4、Q4 F32 records672/672/504/672/672、F16累積records0、nonfinite0、adapter504 tensors適用、graph cache miss1/hit実forward-1を照合しました。img2img2件はGPU VAE encode完了と入力hash、sigma suffixを確認しています。

nativeから取得したPNGとgateway/clientのPNGは全byte一致し、全画像decode・寸法・画素変化・目視を確認しました。rawログとtoken、個人のprocess識別子は公開しません。

owned Directの累積高水位は最終9,629,646,848 bytes（約8.97GiB）、4/5件目の終了時は4,897,161,216 bytesで同値でした。これはprocess全体のhighwaterで、各job単独のピークではありません。RSSやGPUラベル付き値と足し合わせません。5件だけでは長期の断片化・安定性を証明できません。

正常shutdown後に独立observerで両owner不在と同boot/servicesを確認し、全量readback済み同SELFを再起動しました。初期GPUゲートを再び確認し、nativeとgatewayはmodel未読込・空queueのready状態に戻しています。system領域、firmware、GPU driverやメモリ設定を変更していません。

## 公開buildとhost検査

fresh75 core objects、native24 constructors／6 runtime providers、SELF転送後全byte readbackを確認しました。通常policyのhost contract60,015検査、service HTTP55検査（40件連続mock、画像推論なし）、legacy HTTP134検査が通過しています。quota拒否はhostで確認し、実機では保存枠内の5件を確認しました。8GiBまでディスクを埋める試験は行いません。

通常build入口は`tools/build_service.py`、設定は`config/moss-service.json.example`です。gateway build後の共通header変更は説明commentのみで、記録hashを再構成し実行token列が同一であることを確認しました。後から加えた自動freeze wrapperはprepare-only（gateway build＋75-object plan）まで確認し、実測nativeは同じ下位build commandsで作成しています。cloneだけで依存SDKやnativeタイトル登録まで自動導入するものではありません。[build](build.md)、[owner/token・起動手順](deployment.md)、[API](api.md)。

## 計測範囲

今回の5件はprompt、寸法、encode、conditioning cache状態が異なり、相互の速度比較ではありません。3.51秒を1024の速度や通常の新prompt生成時間として扱わないでください。旧同条件1024平均45.127473秒は[以前の速度比較](moss-dit-kernels-20261021.md)です。全寸法・全promptのメモリ適合、2048入力や全PNG/JPEG形式の実生成、長期安定性は未検証です。

通常版はprivate LAN自動bind（明示IP override可）、Bearer認証、直列GPU生成、既定8GiB output/jobs保存枠を使います。保存枠は実filesystem空きの測定ではありません。32MiB/jobの予約を見込み507で受理を止め、生成物を自動削除しません。初回256や回数制限の代わりに、これらの実容量・セキュリティ・完了検査を維持します。

## 公開検証画像の一覧

以下は[verification.json](verification.json)の新旧画像です。過去のtrial結果は当時のprofileとして保持します。

- [moss-dit-kernels-1024-20261021.png](images/moss-dit-kernels-1024-20261021.png)
- [moss-dit-kernels-256-20261021.png](images/moss-dit-kernels-256-20261021.png)
- [moss-img2img-1024-3step-20261021.png](images/moss-img2img-1024-3step-20261021.png)
- [moss-img2img-256-1step-20261021.png](images/moss-img2img-256-1step-20261021.png)
- [moss-img2img-256-2step-20261021.png](images/moss-img2img-256-2step-20261021.png)
- [moss-service-aspect-512x768-20261021.png](images/moss-service-aspect-512x768-20261021.png)
- [moss-service-edit-512-20261021.png](images/moss-service-edit-512-20261021.png)
- [moss-service-edit-full-256-20261021.png](images/moss-service-edit-full-256-20261021.png)
- [moss-service-fifth-256-20261021.png](images/moss-service-fifth-256-20261021.png)
- [moss-turbo4-upload-1024-20261020.png](images/moss-turbo4-upload-1024-20261020.png)
- [moss-turbo4-upload-256-20261020.png](images/moss-turbo4-upload-256-20261020.png)

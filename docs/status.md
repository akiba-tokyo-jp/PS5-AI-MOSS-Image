# 現在の状態

2026-10-21、MOSS通常API版をPS5 FW12.40／RADV NAVI21で検証しました。**任意prompt・seed、長方形、初回1024、5件目の生成を実際に確認済み**です。初回256、固定prompt/seed、4試行の制限を外しました。[今回の詳細とPNG](moss-service-20261021.md)、[機械可読データ](moss-service-20261021.json)。

| 機能 | 状態 |
|---|---|
| OpenAI形式の画像生成・base64 PNG取得 | PS5と公開PowerShell clientで確認 |
| native非同期job・画像取得との照合 | 全5件で確認 |
| 起動後最初の1024 | モデルlazy load込み110.459715秒、先行256なし |
| 任意文章・negative省略・別seed・512×768 | 実画像で確認 |
| ランダムseed | 2要求で別seed、結果header/native jobを照合 |
| img2img | 512 strength0.5/実3、256 strength1/実4を確認。従来256/1024も検証済み |
| 生成回数 | software試行capなし。同じ起動で5件成功。長期連続は未検証 |
| 入力PNG/JPEG各辺2048まで、出力16刻み64～1024 | 受理policy。全寸法・全入力形式の実生成は未検証 |
| 通常保存枠8GiB・507 admission | 実機5件で正常許可、hostで上限拒否を確認。実機で8GiBまで埋める試験は未実施 |
| 正常終了・再起動 | 独立owner不在、同boot/services、再起動全readbackを確認 |
| マスク/inpainting/IP-Adapter/ControlNet/任意LoRA | 未対応 |
| 人物同一性固定・強い実写変換 | 保証なし。実写化は未達 |

追加の任意文章1024試験で、GPU計算が完了しても最適化kernelの全使用を要求するdiagnostic判定が返却を拒否する例を修正しました。通常版はshapeに応じた既存GPU fallbackを許可し、数値・容量・GPU完了検査は保持します。修正版をfresh75 coreと通常build wrapperから作り直し、PS5で新しいキャラクターの1024 PNG取得を確認しました（モデル読込込み124.476684秒）。新旧はpromptが異なるため速度比較ではありません。

Turbo4 profileはEuler4／CFG1、作者sigmas、NSFW0.7＋Turbo1 F32 rank144/runtime1。これは通常利用でも維持するモデル構成です。固定sample文章や試行capとは別です。Direct予約10GiB／推論予算9.5GiB、元14＋Turbo3＋upload＋fusion16＋kernel20の読み戻し・数値・容量・所有権検査を保持します。全5件のDiT有限値とGPU完了、168×実forwardのQ4 F32 records、LoRA504適用を照合しました。

旧`production_runtime_admitted:false`は継承した低水準ABIの状態です。通常利用の受理は独立した`service_mode:true`で行います。旧diagnostic buildの試行制限は残し、過去の比較結果を改変していません。[API](api.md)と[通常版build入口](build.md)を使用してください。

DiT kernel速度基準は[同条件1024平均45.127473秒](moss-dit-kernels-20261021.md)。今回のcold1024は110.459715秒です。異なるpromptやcache状態の5件を速度改善の比較に使いません。長期安定性・全要求のメモリ適合や新しい高速化を示す試験ではありません。追加最適化余地は残ります。

モデルweights、token、private controller、rawログ、ビルド済みbinaryは公開repoに含めません。registered native Homebrew環境・SDK/依存が必要なソース公開です。[導入・owner/token手順](deployment.md)、[第三者通知](../THIRD_PARTY_NOTICES.md)。

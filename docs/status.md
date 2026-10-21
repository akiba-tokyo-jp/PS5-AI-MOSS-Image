# MOSS Imageの検証状態

2026-10-21。最新速度基準はTurbo4＋bounded upload＋DiT融合＋kernel mask15・Attention occupancy調整です。1024は45.415880秒／44.839066秒、平均45.127473秒。今回の対照平均47.481390秒から4.96%短縮しました。初回256は69.449735秒（モデルlazy load込み）。旧版とはPNG画素差があり、同画質・全prompt・長時間安定性は未確認です。[最新の実機比較](moss-dit-kernels-20261021.md)。

前段のGPU timestamp測定で、1024はFFN40.15%、Attention25.41%、QKV15.01%を確認しました。その結果を今回のkernel改善に反映しています。[計測範囲](moss-dit-gpu-profile-20261021.md)。

img2imgを明示的default-OFF trialとして追加しました。native非同期とOpenAI形式editsでPS5出力を確認し、1024は40.833333秒（入力encode4.425秒・実3 forward）でした。256の1／2 forwardも検証済みです。元絵のアニメ調が残り、実写化や人物固定を保証しません。[詳細](moss-img2img-20261021.md)。

## 採用判断と任意実験

| 経路 | 現在の扱い | 実機で確認した範囲 |
|---|---|---|
| DiT kernel4項目＋FA occupancy調整 | 最新の限定trial速度基準、optimization15＋kernel15 | 1024平均45.127473秒、追加20 GPUゲート、4 DiT／672 Q4 F32／LoRA504。owned Direct高水位＋96MiB |
| DiT text/RoPE再利用＋SiTU/norm/residual/QK/LoRA融合 | 前段の基準、明示mask15 | 69.275742／47.760891秒、対照版とPNG完全一致。既存ゲート＋融合16ケース、4 DiT／672 Q4 F32／LoRA504 |
| F32 Turbo4＋bounded upload | 従来の速度比較基準 | 256→1024各1枚、4 DiT／672 Q4 F32／LoRA504、既存14＋Turbo3＋uploadゲート |
| FFN chunk2048 | 速度基準ではOFF | 1024が55.195208秒、前段基準より約8%増。workspace高水位は低下 |
| F16 LoRA storage＋専用F32 shared演算 | default-OFFの任意試験 | 68.897951／51.884730秒、専用8ゲート、LoRA private記録2016。owned Direct使用量637.875MiB低下、warm高速化は未確認 |
| 任意promptの通常運用版 | 公開対象から除外 | 8GiB起動時の物理確保は確認したが、初回要求が保存領域チェックのHTTP507でジョブ登録前に拒否。画像生成は未実施 |
| 30-step res_multistep／Spectrum | 別のsampling試験 | 256→1024各1枚。sampling／forecast変更を含み、Turbo4や基準と同画質比較ではない |

最新のDiT比較・メモリ・画像差は[kernel改善](moss-dit-kernels-20261021.md)、前段は[DiT最適化](moss-dit-fusion-20261020.md)、前段の構成比較は[Turbo4メモリ比較](moss-turbo-memory-options-20261020.md)にまとめています。公開版は確認済みの限定trialを対象とし、通常運用の完成や長時間安定性は主張しません。通常運用試作はファイル所有者情報・ディスク空き容量・相対path処理のnative互換性に不整合が見つかったため、検証済みの構成から分離しました。既存画像は削除せず、試作アプリとgatewayは通常APIで終了し、別の確認でprocess不在と既存サービス維持を確かめました。

## 実装上の境界

- MOSS-Image-SFT Q4_K、Qwen3.5-9B Q4_K_M、FLUX.2 small VAE、外部tokenizerと選定adapterを使います。Windows側のlearned-model推論はありません。
- Turbo4はNSFW0.7＋Turbo1.0をrank144 F32へ合成し、runtime倍率1.0、4 Euler steps／CFG1、author sigmas `[1.0,0.910,0.717,0.024,0.0]` を使用します。flow shiftを二重適用しません。
- production admissionは既定0。明示的なTurbo4 trialは固定prompt・seed20261020、256／1024、1プロセス最大4試行です。失敗した要求も試行数に含め、自動再送しません。任意promptの連続運用は未対応です。
- 数値ゲート、全出力有限値、実scratch／容量、GPU完了、所有権、通常終了を弱めません。不確かなGPU資源を手動で解放しません。
- APIは認証・private /24・n=1・PNG・非streaming。明示img2img trial以外のmask/IP/ControlNet等は拒否します。モデルpathや秘密をクライアント指定で置換しません。

## メモリと容量

速度基準の物理Direct予約は10GiB、予算9.5GiB、pool外reserve1GiBです。API GPU-labelled値、process RSS高水位、heap mapping、owned Direct使用量は別の重複する量で、足して全物理RAM／専用VRAMにはできません。

物理8GiB予約の起動確認は、8GiBでのモデル読込・推論成功を意味しません。予約を小さくすると新prompt時のGPU重みの退避・再読込が必要になり得るため、速度向上の根拠にはしません。

bounded uploadはworkerあたり8MiBの一時host stagingを制限します。F16 storageやFFN分割のDirect使用量低下とは異なり、物理予約の縮小を証明しません。

## 終了と記録

前段のTurbo4、bounded upload、FFN、F16試験では通常API終了後のprocess不在と独立したidle観測を確認しました。同boot・既存サービス・タイトル登録・モデルmetadataを保持しています。通常運用試作も未生成のまま通常終了し、process不在を確認しました。

公開するのは集約した条件・秒数・正常PNG・ソースです。token、個人path／PID、元ログ全文、モデルweights、実行binaryは含めません。

## 保存したMOSS検証資料

- [最新DiT kernel4項目の実機比較](moss-dit-kernels-20261021.md)

- [現行Turbo4のGPU時間特定](moss-dit-gpu-profile-20261021.md) / [外部実装の適用評価](moss-dit-reference-evaluation-20261021.md)
- [最新DiT融合・不変計算再利用](moss-dit-fusion-20261020.md)
- [前段Turbo4／メモリ比較](moss-turbo-memory-options-20261020.md)
- [先行Turbo4単独試験](moss-turbo4-results-20261020.md)
- [sampling／Spectrum](moss-sampler-spectrum-memory-20261020.md)
- [bounded upload](moss-bounded-weight-upload-20261020.md)
- [混合精度](moss-mixed-precision-20261020.md) / [追加GPU候補とtext cache](moss-additional-optimization-20261020.md)
- [50-step単独cold／warm](moss-1024-cold-warm-20261020.md)

各履歴ページに記載した当時の未確認事項や既定値は、その試験時点の範囲です。現在の採用状態はこのページと最新比較を優先します。

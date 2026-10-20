# GPU重み転送のhost staging：実機のメモリ観測

2026-10-20 JST。MOSSのraw・同形式GPU重み転送を、必要量まで増やすhost Scratchで分割しました。**上限8MiBはworkerあたり**です。既存のVulkan tensor転送・同期経路を使い、CPUからGPUアドレスを直接書きません。モデル係数、推論精度、sampling条件は変えていません。

## 同じ要求での初回256比較

| 条件・結果 | 基準キャッシュ版 | 転送制限版 |
|---|---:|---:|
| PS5内の初回生成時間、lazy model読込込み | 146.016289秒 | 145.680067秒 |
| サイズ／steps／sampler／CFG／flow shift | 256×256／50／Euler／3／5 | 同じ |
| seed／選定LoRA | 20261020／rank16 F32、0.7 | 同じ |
| 完了した有限GPU DiT出力 | 100 | 100 |
| Q4行列のF32累積選択／F16累積／invalid | 16600／0／0 | 16600／0／0 |
| 選定LoRA tensor使用／skip | 504／0 | 504／0 |
| API返却・native GETのPNG | 同じ要求の比較基準 | 基準と全バイト一致 |

差は−0.336222秒（−0.2303%）で、この各1回の観測では生成時間はほぼ同等です。反復した高速化の証明にはしません。既存14数値ゲートと、8MiB境界をまたぐ9MiB+68Bの転送・GPU SCALE・CPU読み戻しゲートも合格しました。実機ではPS5内CPU text encoding、Vulkan DiT/VAE、OpenAI互換API経由のPNG取得を維持しました。

## 実際に通った転送経路

| 完了loadの対象 | tensor upload／chunk | 最大の元tensor | 実測最大worker Scratch |
|---|---:|---:|---:|
| DiT本体 | 185／462 | 113,246,208B（108MiB） | 8,388,608B（8MiB） |
| 選定rank16 LoRA | 504／504 | 1,048,576B | 1,048,576B |
| VAEの対象部分 | 104／104 | 589,824B | 589,824B |

計3回の完了loadで793tensor upload／1070chunk、3,477,715,724Bが対象経路を通りました。この合計はload呼出しの集計で、uniqueな常駐tensor量ではありません。LoRAやVAE用Scratchは最初から8MiBを確保せず、各workerが処理するtensorの必要量まで増え、再利用します。拡張時は旧領域を先に解放し、二重確保による一時ピークを避けます。

各loadで3workerが観測され、fileあたりの最大worker数も3でした。可能な同時staging上限は3×8MiB＝24MiBです。3loadのworker instance合計9を同時worker数とは扱いません。24MiBも実測同時使用量、loader全体、process全体、Direct予約の値ではありません。CPU Qwenのhost重み、形式変換、mmap、zip等の対象外経路へこの上限を広げません。

## Heap mappingとRSSの観測

| 計測点・会計 | 基準キャッシュ版 | 転送制限版 | 差 |
|---|---:|---:|---:|
| 生成前heap mapped／peak-mapped | 64／64MiB | 64／64MiB | 0／0MiB |
| model context構築後heap mapped／peak-mapped | 240／240MiB | 240／240MiB | 0／0MiB |
| 生成後heap mapped capacity | 480MiB | 400MiB | −80MiB |
| process lifetime heap peak-mapped capacity | 777.3125MiB | 545.125MiB | −232.1875MiB |
| API `ru_maxrss × 1024`高水位 | 11,787,255,808B | 11,560,042,496B | −227,213,312B（−216.6875MiB） |

Heap値はallocatorのsegment容量で、live malloc payloadやresident page数ではありません。model context構築後という計測点も、lazyな大規模GPU重み転送をすべて終えた時点ではありません。生成後のsegment数は28→25、arena数は8のまま、libc fallbackは0でした。

heap peakと`ru_maxrss`は起動ゲート・CPU text encoding・全生成工程を含むprocess高水位です。今回の低下を個々のtensorやloaderだけの物理削減量へ帰属させません。heap容量差とRSS差は加算できず、後段の節約が前段の高水位に隠れる場合もあります。基準版には同等のworker容量テレメトリがないため、旧vectorからの正確なtensor別削減byte数は算出しません。

## DirectとGPUラベル値は変わらない

| 会計 | 両版の値 |
|---|---:|
| 物理Direct予約 | 10,737,418,240B（10GiB） |
| 所有Direct内の割当高水位 | 4,835,262,464B |
| 生成後の所有Direct live claims | 3,536,896,000B |
| CPU text共有Direct高水位 | 4,835,246,080B、最終live0 |
| API GPUラベルのpool＋driver高水位 | 4,867,227,648B |

host stagingを減らしても、起動時に確保・mapした10GiBの物理Direct予約をsystemへ返したわけではありません。共有Directとdriver会計は独立GPUの専用VRAM residencyではなく、CPU側RSSへ加算もしません。全processの物理RAM、resident VRAM、UMA全体の削減量は未証明です。

## 時間と検証範囲

0.01秒精度のloader総wallは、CPU textが23.16→22.89秒、DiTが12.96→12.90秒、VAEが0.38→0.38秒でした。CPU textの差はこのGPU raw転送変更の対象外で、因果的な改善とは扱いません。read・memcpy・convert・backend-copyの内訳はworker合計をthread数で割った値で、加算できるwall工程や純GPU時間ではありません。

この実験は50-step・Spectrumなしの初回256を各1回比較したものです。温度・clock・OS disk cacheを統制した速度分布、1024の転送制限版、SpectrumやTurboとの併用、長時間運用はこの結果に含めません。より小さいhost stagingと同一出力を確認したことと、モデル全体のメモリ適合性・画質を全条件で保証することは分けて扱います。

bounded stagingの設計参照とライセンスは[既存の出典資料](moss-sampler-spectrum-memory-20261020.md#source-and-attribution)および[第三者通知](../THIRD_PARTY_NOTICES.md)を参照してください。重み、認証情報、生ログ、個人のファイルパスはこの公開資料に含めません。

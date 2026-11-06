# core — Knowledge-informed LSTM for debris-flow early warning

Code and data accompanying the manuscript submitted to *Computers & Geosciences*
("Knowledge-informed LSTM … Gaizi Valley, Chinese Tian Shan").

The repository contains the full training / cross-validation pipeline, the
hourly ensemble inference used for the 2026 early-warning experiment, and the
input data needed to reproduce both.

## Layout

```
core/
├── code/
│   ├── config.py         # path resolution (data are found relative to the repo)
│   ├── model.py          # single-layer-stack LSTM (2 → 32 → 1 logit)
│   ├── data_process.py   # event inventory, 24-h windows, stratified split, scaler
│   ├── train.py          # 5-fold CV, warmup + cosine schedule, early stopping,
│   │                     #   knowledge prior and gated composite loss
│   ├── main.py           # CLI: train/evaluate all modes and splits
│   ├── evaluate.py       # held-out metrics, per-trigger-type breakdown
│   └── forecast.py       # hourly ensemble inference (2026 season)
├── data/
│   ├── ds_plus.csv       # hourly ERA5-Land meteorology, 2022-2025 (training)
│   ├── inventory.csv     # 42 documented debris-flow events (UTF-8)
│   ├── rain.csv          # rainfall-triggered subset
│   ├── heat.csv          # melt-triggered subset
│   └── ds_one_2026.csv   # hourly ERA5-Land meteorology, 2026 (validation)
├── results/
│   ├── model_evaluation_summary.csv   # mean ± SD per split and model (200 runs)
│   └── statistical_tests.csv          # paired t-tests against the baseline
├── requirements.txt
├── LICENSE               # MIT
└── NOTES.md              # known code/paper inconsistencies and caveats
```

## Data

All meteorological fields are hourly ERA5-Land values for the single grid
cell centred on 38.8° N, 75.2° E (3,641 m; the melt calculation uses this
grid elevation with a fixed snowline, see NOTES.md).

| File | Rows | Content |
|---|---|---|
| `ds_plus.csv` | 5,856 | `valid_time, tp, sro, ssro, t2m, swvl1, swvl2, rainfall_mm, temprature_Cel`; contiguous hourly blocks covering the debris-flow seasons of 2022 (Jun–Sep), 2024 and 2025, i.e. the hours from which the 24-h windows are drawn |
| `inventory.csv` | 42 | road-inspection DF records: `事件, 发布时间, 影响路段, 受损方向, 道路类型, 通行状态, 开始时间, 实际恢复时间, 详细描述, 数据来源` |
| `rain.csv` | 42 | `event_time, rain_start, duration_hours, rain_mm, rain_in_debris` (rainfall-triggered subset) |
| `heat.csv` | 42 | `event_time, heat_start, duration_hours, heat_cel, temp_in_debris, meltwater` (melt-triggered subset) |
| `ds_one_2026.csv` | 2,208 | `valid_time, tp, t2m, rainfall_mm, temprature_Cel`, 2026-06-01 08:00 – 2026-09-01 07:00 (the early-warning season) |

`inventory.csv` was converted from GBK to UTF-8; the code reads it with
`encoding='utf-8-sig'`.

## Environment

Tested with Python 3.13.5, PyTorch 2.6.0 (CUDA 12.6), NumPy 2.4, pandas 3.0,
scikit-learn 1.8, SciPy 1.17. CPU-only execution works but is slow for the full
200-run sweep.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running

Reproduce the training comparison (8:2, 7:3 and 6:4 splits, 200 runs each):

```bash
cd code
python main.py --modes baseline per_run_ed global_ed --splits 0.2 0.3 0.4 --runs 200
```

Quick smoke test (a few seconds, checks the whole pipeline):

```bash
python main.py --modes baseline --splits 0.2 --runs 1 --epochs 2 --k-folds 2
```

Outputs are written next to `main.py` (`mnt_Test_0.2/`, `mnt_Test_1.0/`, …),
including per-run CSV logs and training curves. The aggregate tables are
written to the repository root as `model_evaluation_all_runs.csv`,
`model_evaluation_summary.csv` and `statistical_tests.csv`.

Hourly ensemble inference for the 2026 season (requires trained weights in
`models/`, see below):

```bash
python forecast.py --models ../models --start 2026-06-01 --end 2026-08-31
```

The 2026 series starts at 08:00 on 1 June, and the first 24 h are consumed as
window lookback, so the first hourly score is issued at 2026-06-02 08:00.
The output CSV holds one row per hour: ensemble voting share (0–100), mean
probability, wet-spell duration and cumulated water, the E–D reference score,
and the binary alarm flags.

## Training protocol (as implemented)

| Item | Value |
|---|---|
| Input | 24 h × 2 features (`rainfall_mm`, `temprature_Cel`) |
| Hidden units / layers / dropout | 32 / 2 / 0.2, unidirectional |
| Output | 1 logit; sigmoid applied in the loss and at evaluation |
| Data loss | `BCEWithLogitsLoss` |
| Knowledge loss | `(1−λ)·BCE + λ·MSE(σ(z), p)`, λ = clip(R², 0, 1), gated on prior–label agreement |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | 5-epoch linear warmup, then cosine annealing (T_max = 100) |
| Batch size | 8 |
| Epochs / patience | 100 / 10 |
| Cross-validation | `StratifiedKFold(5, shuffle, random_state=3407+run)` inside the training partition; the fold with the lowest validation loss is kept |
| Repetitions | 200 per split–model combination (results/), CLI default 100 |
| Ensemble | 100 KI-LSTM + 100 baseline models, equal weights, alarm when the voting share ≥ 50% |

The knowledge prior is `p = clip(E / (a·D^b), 0, 1)`, where `E` is the
cumulated water equivalent of the wet spell in the 24 h before the window end
and `D` its duration. The global threshold is `a = 0.238`, `b = 1.265`,
`R² = 0.841`; the per-run variant re-fits `a`, `b` on the training events with
`scipy.optimize.curve_fit`.

## Model weights

The 200 trained weights (~11 MB) are not committed. Regenerate them with:

```bash
python main.py --modes baseline global_ed --splits 1 --runs 100
```

which writes `mnt_Test_1.0/lstm{i}.pth` and `mnt_Test_1.0/PI_lstm_global{i}.pth`;
copy or symlink that folder to `models/` before running `forecast.py`.

## Naming

The code keeps the historical `PI_lstm` file prefix. In the manuscript these
models are referred to as **KI-LSTM** (knowledge-informed); `PI_lstm_global`
is the single-threshold variant, `PI_lstm` the per-run threshold variant.

## Citation and licence

Please cite the manuscript when it is published. Code released under the MIT
licence (see `LICENSE`).

Known limitations and the inconsistencies between this code and earlier
versions of the manuscript are documented in `NOTES.md`; please read it before
interpreting the reported numbers.

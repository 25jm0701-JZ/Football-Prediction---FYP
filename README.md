# Football Prediction FYP — Atta Mills PL Walk-forward Branch

This branch tests whether the feature framework from Atta Mills et al. (2024)
improves a clean pre-match Premier League result prediction task.

The core question is deliberately narrow:

> If we use Premier League data from football-data.co.uk, exclude odds from
> training, exclude half-time/in-play variables, and evaluate by walk-forward,
> how well does an Atta Mills-style feature set perform?

The answer from this branch is: useful as a reproduction baseline, but not yet
better than the Bet365 closing market.

## Branch Context

| Item | Value |
|---|---|
| Baseline branch | `master` |
| Experiment branch | `codex/atta-mills-pl-walkforward` |
| Baseline backup tag | `backup/current-experiment-2026-09-03` |
| Baseline commit | `208dd90` |
| Main experiment record | `docs/experiment_log.md` |

## Data And Target

- Data source: football-data.co.uk Premier League CSV files.
- Seasons: 2019/20 through 2025/26.
- Matches: 2,660.
- Target: full-time match result, encoded as Away / Draw / Home.
- Validation: season-by-season walk-forward.
- Market benchmark: Bet365 closing odds, de-vigged into implied probabilities.

## Clean Prediction Rules

The following are excluded from model training:

- Bet365 odds and all other bookmaker odds.
- Half-time result and half-time goals.
- Over/Under 2.5 target and odds.
- FootyStats PPG/xG/possession data.
- Current-match raw statistics that would leak the result.

This keeps the experiment as a pre-match prediction task. The high accuracy
reported in Atta Mills et al. is likely helped by half-time/in-play variables,
which are not comparable to a pre-match model.

## Phase 1: Atta Mills-only Features

Script:

```powershell
python scripts/atta_mills_pl_phase1.py
```

Feature set: 44 pre-match features aligned to Atta Mills-style categories:

- Team state / rolling points
- Attack strength
- Defense strength
- Goals for
- Goals against
- Goal differential
- Win/draw/loss history
- Win margin goals
- Loss margin goals

Aggregate walk-forward result:

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **49.69%** | 0.3810 | 0.0443 | 1.0339 | 0.2065 |
| FNN/MLP | 49.56% | 0.3750 | 0.0356 | 1.0328 | 0.2063 |
| Random Forest | 49.52% | 0.3709 | 0.0247 | 1.0353 | 0.2066 |
| XGBoost | 46.58% | 0.3883 | 0.1427 | 1.1041 | 0.2186 |
| LR balanced | 44.25% | 0.4172 | 0.2447 | 1.0595 | 0.2129 |
| Bet365 closing | **54.87%** | 0.4087 | 0.0000 | **0.9626** | **0.1904** |

Conclusion: the strict Atta Mills-style pre-match feature set underperforms the
Bet365 closing market.

## Phase 2: Add Original Clean Features

Script:

```powershell
python scripts/atta_mills_pl_phase2.py
```

Phase 2 keeps the 44 Phase 1 features and adds 21 original clean features:

- 16 rolling shot-efficiency features
- 5 head-to-head features

Aggregate walk-forward result:

| Model | Accuracy | Macro F1 | Draw F1 | LogLoss | Brier |
|---|---:|---:|---:|---:|---:|
| Random Forest | **51.01%** | 0.3773 | 0.0073 | 1.0151 | 0.2020 |
| SVM | 49.96% | 0.3638 | 0.0000 | 1.0277 | 0.2053 |
| FNN/MLP | 49.21% | 0.3595 | 0.0110 | 1.0356 | 0.2059 |
| LR | 48.60% | 0.4071 | 0.1376 | 1.0708 | 0.2120 |
| LR balanced | 44.96% | 0.4304 | **0.2786** | 1.0980 | 0.2187 |
| Bet365 closing | **54.87%** | 0.4087 | 0.0000 | **0.9626** | **0.1904** |

Conclusion: adding the original clean features improves the best model from
49.69% to 51.01%, but still does not beat the closing market.

## Phase 3: Half-time Paper-style Backup

Script:

```powershell
python scripts/atta_mills_pl_phase3_halftime.py
python scripts/atta_mills_english4_phase3_halftime.py
```

This branch keeps a paper-style half-time reproduction as a backup experiment
only. It is **not recommended** as the final project direction because
half-time goals/result are in-play information, not pre-match information. The
result is useful for explaining why the reference paper's reported accuracy can
look much higher than a clean pre-match forecast, but it should not be presented
as the deployable prediction model.

Aggregate walk-forward result:

| Data | Config | Model | Accuracy | Draw F1 | Market Accuracy |
|---|---|---|---:|---:|---:|
| PL only | Half-time, no odds | RF balanced | 60.88% | 0.3879 | 54.87% |
| PL only | Half-time + B365 opening | RF | 62.15% | 0.2806 | 54.87% |
| E0-E3 | Half-time, no odds | LR | 59.16% | 0.3688 | 49.38% |
| E0-E3 | Half-time + B365 opening | LR | 60.31% | 0.3518 | 49.38% |

Conclusion: half-time variables lift accuracy sharply, but they change the
research problem into an in-play task. Keep this branch as evidence and backup,
not as the recommended modeling path.

## Kaggle GPU Notebook

The local machine can also run CUDA PyTorch through the isolated `.venv_cuda`
environment. The local GPU check detected:

```text
PyTorch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

Local GPU MLP run:

```powershell
.\.venv_cuda\Scripts\python.exe scripts\atta_mills_pl_gpu_mlp.py
```

Result: `torch_mlp_deep` reached 49.96% accuracy and `torch_mlp_wide` reached
48.68%, both below Bet365 closing odds and the Phase 2 Random Forest.
No further GPU tuning is planned for this branch; the CPU walk-forward scripts
are the default path for routine experiments.

A Kaggle-ready GPU notebook is also included:

```text
notebooks/atta_mills_phase2_kaggle_gpu.ipynb
```

Upload the project or the `PL*.csv` files to Kaggle, enable GPU in notebook
settings, and run the notebook to train PyTorch MLP variants on cloud GPU.

## Outputs

Generated outputs are ignored by Git and saved locally under:

```text
outputs/atta_mills_pl_walkforward/
```

Important generated files:

- `phase1_atta_mills_only/model_comparison_walkforward.csv`
- `phase1_atta_mills_only/market_comparison_bet365_closing.csv`
- `phase2_with_original_features/model_comparison_walkforward.csv`
- `phase2_with_original_features/market_comparison_bet365_closing.csv`
- `phase2_with_original_features/feature_set.csv`

## Interpretation

This branch is evidence that the reference paper's headline performance should
not be copied directly into a pre-match setting. Once half-time variables, odds
training features, O/U 2.5, and external FootyStats features are removed, the
problem becomes much harder.

The next useful direction is either:

- Restore richer pre-match data such as FootyStats PPG/xG/possession.
- Improve draw handling without sacrificing too much overall accuracy.
- Use Kaggle GPU to test stronger neural architectures on the Phase 2 feature
  set.

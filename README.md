# Arb Future Edge Model

Minimal modeling workspace for:

- extracting IDENTICAL-like cross-venue opportunities from Replay Lab
- generating supervised labels from execution outcomes
- running a first baseline evaluation before any live policy changes

## Setup

1. Copy `.env.example` to `.env`
2. Fill in:
   - `REPLAY_LAB_API_URL`
   - `REPLAY_LAB_API_KEY`
   - `DATABASE_URL` (read-only is enough)
3. Install dependencies:

```bash
pnpm install
```

## Commands

```bash
pnpm check:api
pnpm extract:dataset
pnpm labels:generate
pnpm baseline:eval
pnpm train:baseline
pnpm backtest:baseline
pnpm train:robust
pnpm backtest:robust
pnpm backtest:walkforward
```

## Output

- `data/training_dataset.jsonl`: one row per sizing decision with market metadata (titles/categories/tags)
- `data/labeled_training_dataset.jsonl`: capacity-aware labels with:
  - short horizons (`15m`, `1h`, `3h`)
  - resolution-anchored labels (max 7d)
  - policy-window targets (`deltaNetPnlPolicyWindowAtNowSize`, `buyNowBeatsWaitWindow`)
  - taxonomy labels (`domain`, `subdomain`, `topic`)
- `data/label_summary.json`: horizon stats + resolution phase stats + taxonomy-domain stats
- `data/baseline_report.json`: baseline metrics for policy-window regression/classification targets
- `data/model_baseline_report.json`: trained baseline model accuracy (regression + classification)
- `data/model_baseline_artifacts.json`: feature schema/scaler and learned baseline weights
- `data/model_backtest_report.json`: held-out decision backtest with threshold sweep
- `data/model_robust_report.json`: robust model metrics (winsorized + segmented + calibrated)
- `data/model_robust_artifacts.json`: robust model artifacts and tuned threshold
- `data/model_robust_backtest_report.json`: validation-tuned robust backtest on test split
- `data/walkforward_backtest_report.json`: rolling walk-forward backtest window-by-window

## Latest Live Refresh + Backtest Snapshot

Run date: `2026-02-17`

### 1) Live opportunity refresh (Artemis run once)

- Artemis backend was started locally and `/api/scan/category` was triggered once with:
  - `keywords=bitcoin`
  - `marketsPerKeyword=1`
  - `matchesPerMarket=1`
  - `patternTypes=IDENTICAL`
- Scan response:
  - `arbsFound=0`, `arbsStored=0`, `identityRejected=1`
  - warning indicates table mismatch in the current DB target:
    - `relation "market_opportunities" does not exist`

### 2) Data refresh after live run

- Extracted rows: `2823`
- Labeled rows used for training: `2681`
- Dedupe series: `142`

### 3) Robust model accuracy (single chronological split)

- Classification target: `buyNowBeatsWaitWindow`
  - test `AUC=0.568`
  - test `F1=0.679`
  - test `accuracy=0.543`
- Regression target: `deltaNetPnlPolicyWindowAtNowSize`
  - test `MAE=11.20`
  - test `RMSE=42.20`
  - test `R²=-0.141`

### 4) Backtest performance

- Single-split backtest (`model_robust_backtest_report.json`)
  - tuned threshold: `0.65` (tuned on validation only)
  - validation relative PnL vs always-buy-now: `+1146.64`
  - test relative PnL vs always-buy-now: `-152.76`

- Walk-forward backtest (`walkforward_backtest_report.json`)
  - windows: `5`
  - profitable windows: `4/5` (`80%`)
  - mean total relative PnL per window: `+1543.13`
  - median total relative PnL per window: `+1908.84`

### 5) "T-7d, T-24h, T-6h" phase accuracy for resolved-style labels

- Current label coverage is concentrated in `T_7d_3d` only.
- `T_3d_1d`, `T_24h_6h`, `T_6h_1h`, and `T_1h_close` currently have `0` rows in this sample.
- Practical implication:
  - We can evaluate meaningful predictive performance for early phase (`T-7d..T-3d`).
  - We cannot yet claim accuracy for late phases (`T-24h`, `T-6h`, `T-1h`) from this dataset slice.

### 6) What accuracy depends on most right now

- **Market domain shift** is material:
  - `sports` and some `other` slices show stronger classification behavior.
  - `politics` is weaker and drives generalization instability.
- **Phase coverage** is the main blocker for late-resolution claims.
- **Liquidity/capacity noise** and sparse high-quality late snapshots dominate regression error.

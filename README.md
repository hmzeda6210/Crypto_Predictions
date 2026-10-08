# Bitcoin Price Prediction — BSc Final Year Project (2024)

> **Archived final year project, kept public deliberately.** This was my
> dissertation project for BSc Computer Science at the University of
> Westminster. The code runs and is reasonably clean, but both models contain
> methodology errors I did not catch at the time. I have left them unfixed
> and documented them in full below, because the post-mortem is more useful
> than the results were.
>
> **Do not read the reported metrics as working results.** See
> [What I'd do differently](#what-id-do-differently).

---

## Projects

### 1. Price Movement Classification
**Notebook:** `BTC_Classification.ipynb`

Random Forest classifier over 5-minute BTCUSD bars.

- **Data:** `BTCUSD_m5.csv` — Jan 2023 to May 2024, ~110,000 rows, sourced
  from [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)
- **Task:** Three-class trend label — no trend (0), downtrend (1), uptrend (2)
- **Features:** ATR, RSI, midprice, SMA 40/80/160, and rolling
  linear-regression slopes of each over a 6-bar window
- **Split:** Chronological 70 / 15 / 15
- **Reported result:** 75.3% validation accuracy, 71.5% test accuracy

![Random Forest pipeline](RandomForest_Pipeline_Diagram.png)

### 2. LSTM Price Forecasting
**Notebook:** `LSTM_BTC_Final_Model.ipynb`

Two-layer LSTM regression on the next close price.

- **Data:** 1-minute BTC-USD bars pulled live via `yfinance` (rolling 7-day window)
- **Features:** OHLC, RSI(15), EMA 13/50/200
- **Architecture:** LSTM(32, return_sequences) → Dropout(0.2) → LSTM(32) →
  Dropout(0.2) → Dense(1); Adam, MSE loss, early stopping on validation loss
- **Sequence length:** 4
- **Split:** Chronological 80 / 10 / 10
- **Reported result:** validation R² 0.987, test R² 0.013

![LSTM pipeline](LSTM_Pipeline_Diagram.png)

---

## What I'd do differently

Reviewed in 2026. Four substantive problems, in order of severity.

### 1. The classification target looks backwards

`check_candles` labels row *i* by comparing `Close[i-6:i]` against
`MA40[i-6:i]` — entirely past bars. The model classifies a trend that has
already happened; it does not forecast anything. Grouping the labels by
subsequent return makes this plain:

| Label | Mean past 6-bar return | Mean future 6-bar return |
|---|---|---|
| Uptrend | +0.0916% | +0.0134% |
| Downtrend | −0.0740% | +0.0054% |
| No trend | −0.0021% | +0.0032% |

The past separates cleanly. The future is near-identical across all three
classes — the label carries almost no information about what happens next.
The 71.5% is real accuracy at recognising a label derivable from data the
model can already see.

### 2. No baseline, so the headline number is meaningless

`Close > MA40` — one comparison, no model, no training — scores **67.5%** on
the same test split against the Random Forest's 71.5%. Fifteen features and
an ensemble bought four points. The majority-class floor is 37.2%, which is
the comparison the notebook implicitly invites and which flatters the result
badly.

### 3. Data leakage in the LSTM

The MinMax scaler is fitted on the full dataset before splitting:

```python
df_scaled = scaler.fit_transform(df[features + ['TargetNextClose']])  # all rows
```

The symptom is visible in the committed output — `Test MAPE: inf`. That only
occurs when `y_test` contains exactly `0.0`, which in MinMax space means the
global minimum of the entire series fell inside the test window and the
scaler had already seen it.

### 4. The LSTM loses to doing nothing

Test R² of **0.013**, against a persistence baseline (next close = current
close, zero parameters) scoring **R² 0.997, MAE ~$84**. The model is
dramatically worse than no model.

The validation R² 0.987 → test R² 0.013 collapse was the warning sign, and I
did not act on it. Metrics were also never inverse-transformed, so the
reported RMSE of 0.0748 is in scaled 0–1 units and means nothing in dollars.

### Also wrong

- **Non-stationary features into a tree.** Raw OHLC and moving-average
  *levels* fed to a Random Forest trained on $16–30k prices and tested on
  $57k+. Trees split on absolute thresholds and cannot extrapolate beyond the
  training range. Ratios (`Close/MA40 - 1`) would have been correct.
- **Unexamined data quality.** 3.1% of bars have zero price change, with a
  longest frozen run of **252 consecutive bars** — about 21 hours of flat
  synthetic price from Dukascopy gap-filling. Never cleaned or acknowledged,
  and it inflates the "no trend" class.
- **Irreproducible by construction.** The LSTM pulls a rolling 7-day
  `yfinance` window, so it returns different data on every run. Combined with
  `random_state=None` on the classifier and no TensorFlow seeds, none of the
  reported numbers can be regenerated — including by me.
- **`StandardScaler` before a Random Forest** is a no-op. Harmless, but it
  shows the pipeline was assembled by pattern rather than by reasoning.
- **`Volume` dropped** from the LSTM features with no stated justification.

### The general lesson

Every one of these is a variant of the same mistake: not printing a trivial
baseline next to the model. A backward-looking target and a leaking scaler
both look like success right up until something with zero parameters beats
you. I now treat a baseline as a precondition for reporting any model result,
not a nice-to-have. That it took a final year project's worth of work to
learn it is the point of leaving this up.

---

## Dissertation

The submitted report is included as `dissertation.pdf`. It was written in
2024 and predates the audit above — where the two disagree, the audit is
correct.

---

## Running these notebooks

The notebooks are archived as-is and are **not** reproducible — see point 4
above. If you want to run them anyway:

```bash
git clone https://github.com/hmzeda6210/Crypto_Predictions.git
cd Crypto_Predictions
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

`TA-Lib` requires a system-level C library that `pip` will not install for
you. On macOS: `brew install ta-lib`. On Debian/Ubuntu you need to build it
from source. On Windows, use a prebuilt wheel. The LSTM notebook imports it
but barely uses it — if installation is painful, you can drop the `import
talib` line and the notebook still runs.

`BTC_Classification.ipynb` runs offline against the committed CSV.
`LSTM_BTC_Final_Model.ipynb` requires an internet connection and will fetch
whatever the last 7 days happen to be.

---

## Repository contents

| File | Description |
|---|---|
| `BTC_Classification.ipynb` | Random Forest trend classification |
| `LSTM_BTC_Final_Model.ipynb` | LSTM next-close regression |
| `BTCUSD_m5.csv` | 5-minute BTCUSD OHLC, Jan 2023 – May 2024 (Dukascopy) |
| `dissertation.pdf` | Submitted final year project report (2024) |
| `RandomForest_Pipeline_Diagram.png` | Classification pipeline diagram |
| `LSTM_Pipeline_Diagram.png` | LSTM pipeline diagram |

---

## Licence

MIT — see `LICENSE`.

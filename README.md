# Hourly Electricity Load Forecasting (PJM East)

A small, honest comparison of naive baselines, linear and tree models, an LSTM and a
Kalman filter for **one-hour-ahead** electricity load forecasting, including an
uncertainty analysis of the Kalman filter.

## Problem
Forecast hourly electricity load (MW) for the PJM East region (PJME).
**Horizon: 1 step ahead (1 hour).** At each hour, the model sees the true past values
and predicts the next hour. Predictions are not fed back into the inputs.

## Data
[PJM Hourly Energy Consumption (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption),
file `PJME_hourly.csv`, hourly data from 2002 to August 2018 (not included in this repo).

Cleaning: removed 4 duplicate timestamps and filled 30 missing hours by linear
interpolation, giving 145,392 regular hourly observations.

## Method
- **Chronological split:** first 80% for training (2002 to about 2015), last 20% for testing
  (about 2015 to 2018). No random split, to avoid leakage.
- **Linear Regression / Random Forest:** features are hour, day of week, month, weekend flag,
  lags (1 h, 24 h, 168 h) and a 24 h rolling mean (shifted by one step to avoid leakage).
- **LSTM:** 48-hour input window plus sin/cos-encoded calendar features, trained in PyTorch
  on Google Colab (GPU) for 15 epochs. The best checkpoint on a validation slice of the
  training period was restored before testing. Same test set as the other models.
- **Kalman filter:** local linear trend model (level + slope). Noise parameters were tuned on
  the first 20,000 training hours using Gaussian negative log-likelihood; the test set was
  not used for tuning.
- **Metrics:** MAE, RMSE, MAPE.

## Results (test set)

| Model | MAE (MW) | RMSE (MW) | MAPE % |
|---|---|---|---|
| Naive (same hour last week) | 3391.9 | 4670.3 | 10.56 |
| Naive (same hour yesterday) | 2184.3 | 3009.0 | 6.93 |
| Naive (1 hour ago) | 1070.0 | 1373.8 | 3.48 |
| Kalman (local linear trend) | 1070.0 | 1373.9 | 3.48 |
| Linear Regression | 977.0 | 1254.2 | 3.15 |
| Random Forest | 336.3 | 462.7 | 1.07 |
| **LSTM** | **187.4** | **259.3** | **0.61** |

All plots (seasonality, actual vs. predicted, error analysis, Kalman intervals) are
shown directly in the notebooks `analysis.ipynb` and `lstm_colab.ipynb`.

## Findings
- The strongest simple reference is the 1-hour-ago naive forecast. Linear Regression beats it
  only slightly; Random Forest and the LSTM beat it clearly.
- The LSTM was the best model. It also saw a richer history (48 h) than Random Forest
  (a few hand-picked lags), so part of the gap may come from the inputs rather than the model
  type. Each model was trained once (single seed).
- The LSTM's validation loss fell from 0.0078 to about 0.0017 and plateaued, with the best
  checkpoint (epoch 13) restored. No sign of overfitting; validation error (about 250 MW RMSE)
  was consistent with test error (259 MW).
- "Same hour yesterday" beat "same hour last week", the opposite of what I expected.
- The largest Random Forest errors cluster on public holidays (Thanksgiving, Labor Day,
  Christmas Eve) and probably on extreme-weather days. Neither holidays nor weather are
  features in this version.

### Kalman filter and uncertainty
- Tuning pushed the filter to trust the measurements fully, so its point forecast is
  essentially the 1-hour-ago naive forecast. It adds no accuracy.
- Its value is the uncertainty estimate. The 95% interval covered **94.9%** of test points,
  which looks well calibrated, but the mean interval width was **5,544 MW** (not sharp).
- Coverage by hour shows the calibration is only apparent: it drops to about 59% at 07:00 and
  to roughly 75-78% around 23:00-00:00 (steep ramps), while it is close to 100% at night and in
  the afternoon (intervals too wide). A fixed noise level cannot represent time-varying
  uncertainty, which motivates adaptive or seasonal Kalman filters.
  (Hourly values are read approximately from the coverage plot in `analysis.ipynb`.)

## Limitations
- 1-hour-ahead is the easiest horizon; day-ahead forecasting is more realistic.
- No weather data and no holiday feature.
- Fixed (non-adaptive) Kalman noise parameters.
- Single run per model; no hyperparameter search.

## Future work
Day-ahead (24 h) forecasting, holiday and temperature features, adaptive Kalman filtering
(online Q/R estimation), seasonal Kalman states, multiple seeds, Transformer models.

## Files
- `analysis.ipynb`: data cleaning, features, naive/linear/Random Forest baselines, error
  analysis, Kalman filter (run locally).
- `lstm_colab.ipynb`: LSTM model (run on Google Colab with GPU).
- `requirements.txt`: Python dependencies. `torch` is only needed for the LSTM notebook.

## How to run
Download `PJME_hourly.csv` from Kaggle and place it next to the notebooks, then:
```
git clone https://github.com/emirsakarcan15/electricity-load-forecasting.git
cd electricity-load-forecasting
pip install -r requirements.txt
jupyter notebook
```

## AI assistance
This project was built with help from AI assistants (Claude by Anthropic and Gemini by Google)
for planning, code drafting, debugging and explaining concepts. I ran all experiments myself
and reviewed the results.
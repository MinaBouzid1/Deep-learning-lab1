
# LAB 1 DEEP LEARNING 

## PART 1 : Stock Price Regression with PyTorch 

- **Dataset:** [NYSE prices (split-adjusted)](https://www.kaggle.com/datasets/dgawlik/nyse) (`prices-split-adjusted.csv`)
- **Notebook:** `regression.ipynb`
- **Tools:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, PyTorch, skorch

---

## 1. Exploratory Data Analysis

| Property | Value |
|---|---|
| Rows | 851,264 |
| Columns | `date, symbol, open, close, low, high, volume` |
| Stocks | 501 |
| Period | 2010-01-04 to 2016-12-30 |
| Missing values / duplicates | 0 / 0 |

Main observations:

- **Right-skewed prices:** mean close is about 65 and the median about 48, with a maximum of about 1,578. A few very expensive stocks pull the mean up. They are real stocks, not errors, so they were kept.
- **Volume is on a very different scale** (0 to about 860 million, standard deviation more than twice the mean), so **feature scaling is required**.
- **Open, high, low and close are almost perfectly correlated** (about 1.0). Volume is only weakly correlated with them.
- **Time series plots** (AAPL, and AAPL/GOOGL/AMZN together) show clear trends and very different price levels between stocks.

## 2. Model Architecture (PyTorch)

**Task:** use `open, high, low, close, volume` of day *t* to predict `close` of day *t+1*.

Pipeline:

1. **Target creation:** `groupby('symbol')['close'].shift(-1)` so a stock's last day never borrows another stock's price.
2. **Time-based split** (not random): 70% oldest dates for train, 15% validation, 15% most recent for test.
3. **Scaling:** `StandardScaler` fitted on the training set only (no leakage).
4. **Mini-batch training** with `DataLoader` (batch size 256).
5. **Model:** MLP `5 -> h1 -> h2 -> 1` with ReLU, no activation on the output (continuous value).
6. **Loss/optimizer:** MSE loss, Adam.

## 3. Hyperparameter Tuning (GridSearchCV + skorch)

skorch wraps the PyTorch model so scikit-learn's `GridSearchCV` can be used. A `PredefinedSplit` keeps the time order (older rows train, later rows validate). A subsample (100k train / 30k validation rows) was used for speed.

Grid: learning rate `{0.001, 0.01}`, optimizer `{Adam, SGD}`, epochs `{10, 20}`, hidden layer 1 `{32, 64, 128}`, hidden layer 2 `{16, 32}` (48 combinations). Scoring: negative MAE.

**Best parameters:** `lr=0.001`, `Adam`, `20 epochs`, `h1=128`, `h2=32` (validation MAE about 1.06).

Findings:

- **SGD performed very poorly** (MAE about 22 to 44): too slow at lr=0.001 and divergent at lr=0.01.
- **lr=0.01 often produced NaN** predictions (training divergence, since the target is unscaled prices).
- Wider first layer and more epochs helped slightly.

## 4. Training Curves

Loss (MSE, log scale) and **MAE vs epochs** were plotted for train and test data. MAE is used as the "accuracy" metric because accuracy (percentage of correct labels) does not exist for regression.

Interpretation:

- The model converges after the first epoch (MAE already below 1.2), then plateaus around 0.68 (train) and 1.0 (test).
- **No overfitting:** train and test curves move together with a small, stable gap.
- **Noisy curves** with spikes (epochs 6 and 14): caused by a constant learning rate, mini-batches mixing stocks with very different price levels, and an unscaled target.
- Test error is higher than train error because the test period (2015-2016) has different, generally higher prices.
- The last epoch (MAE 1.113) is not the best one (epoch 13: 0.985), which motivates early stopping.

## 5. Regularization and Comparison

Techniques tested (on the best architecture 128/32): **dropout (p=0.1)**, **weight decay (L2, 1e-4)**, **early stopping + ReduceLROnPlateau scheduler** (based on validation loss), and a **combination** of all.

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Naive (tomorrow = today) | 0.9502 | 2.1350 | 0.9996 |
| Baseline MLP | 1.1130 | 2.4065 | 0.9995 |
| Dropout | 1.0682 | 2.2472 | 0.9995 |
| Weight decay | 0.9785 | 2.1497 | 0.9996 |
| EarlyStop + LR scheduler | **0.9650** | 2.1468 | 0.9996 |
| Combined | 1.0450 | 2.2791 | 0.9995 |

Interpretation:

- **The naive baseline beats every neural network** (MAE 0.95). The network mostly learned to copy today's close price.
- **Early stopping + scheduler gave the best MLP** (MAE 0.965) and smoother curves.
- **Weight decay helped**, **dropout helped less**, and **combining everything was not better**: the baseline was not overfitting, so extra regularization added noise rather than value.
- R2 is about 0.9995 for all models, so it does not discriminate between them. MAE and RMSE are more informative.

### Limitations

- The regularized runs restore the best-validation weights, while the baseline reports its last-epoch weights. Part of the improvement therefore comes from checkpointing, not regularization alone.
- Epochs differ (baseline 20, regularized up to 30).
- Single run with one random seed: differences of about 0.05 MAE are within noise.
- All 501 stocks are mixed in one model despite very different price levels.

---

## What I Learned

- I learned that data analysis and scaling are important, and the data must be split by date to avoid leakage.
- I compared full-batch training with mini-batches and got better results using mini-batches.
- I learned how to build and train a PyTorch model, use `DataLoader`, and apply `GridSearchCV` with skorch.
- Adam worked better than SGD, and early stopping helped improve training.
- A high R² does not always mean good predictions, so comparing with a simple baseline is important.
- To improve the model, I could predict returns instead of prices, try LSTM/GRU, and repeat experiments with different seeds.

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch skorch
```

1. Download `prices-split-adjusted.csv` from the Kaggle dataset and place it next to the notebook (or update the path).
2. Open `regression.ipynb` and run all cells from top to bottom.

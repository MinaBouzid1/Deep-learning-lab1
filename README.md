
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

## PART 2 : Multi-class Classification with PyTorch (Predictive Maintenance)

Dataset: AI4I 2020 Predictive Maintenance (predictive_maintenance.csv, Kaggle: shivamb/machine-predictive-maintenance-classification)
Notebook: multiclass_classification.ipynb
Tools: Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, PyTorch, skorch

## 1. Pre-processing

| Property | Value |
|---|---|
| Rows | 10,000 (9,991 after cleaning) |
| Columns | UDI, Product ID, Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Target, Failure Type |
| Task | Predict **Failure Type** (6 classes) from 6 features |
| Missing values / duplicates | 0 / 0 |

Steps (and why):

- **Cleaned column names** (`Air temperature [K]` became `Air_temperature`) so they are easy to type and less error-prone.
- **Removed 9 contradictory rows** (`Target = 1` but `Failure Type = "No Failure"`): a failure happened but the label says none, so the label is wrong. I did **not** remove the 18 "Random Failures" rows even though their `Target` is 0, because that is how the dataset was built, and dropping them would delete a whole class.
- **Dropped `UDI` and `Product ID`** (just identifiers, a network could memorize them) and **`Target`** (data leakage: it basically says "is the Failure Type different from No Failure", so it gives away the answer).
- **Encoding:** `Type` (L < M < H) was ordinally encoded as 0/1/2 because the quality levels have a natural order. `Failure Type` was label-encoded to 0 to 5 because `CrossEntropyLoss` needs integer class ids.
- **Stratified split 70 / 15 / 15** (6,993 train, 1,499 validation, 1,499 test). Stratifying keeps the class proportions the same in every split, which is essential when a class has only 18 rows.
- **Standardization** with `StandardScaler` fitted on the **training set only** (no leakage). Without scaling, rotational speed (about 1,500) would dominate torque (about 40) and `Type` (0 to 2).

Order matters: **split, then scale, then SMOTE**. Anything learned from data (mean/std, synthetic samples) must come from the training set only.

## 2. Exploratory Data Analysis

Main observations:

- **Severe class imbalance.** "No Failure" is 96.52% of the data, while the failure classes have only 112, 95, 78, 45 and 18 rows. The imbalance ratio is about 536:1. A model that always answers "No Failure" would get 96.5% accuracy while detecting nothing, so accuracy cannot be trusted here.
- **Sensors are well behaved.** Temperatures, torque and tool wear are roughly symmetric (skewness close to 0). Only rotational speed is right-skewed (skewness 1.99).
- **Outliers kept.** The IQR rule flags 418 outliers in rotational speed and 70 in torque. I kept them because extreme torque or speed is exactly what comes before a failure, so removing them would remove the signal.
- **Each failure has a "signature"** (class-mean heatmap in standard deviations):
  - Heat Dissipation: low speed (-1.12), high torque (+1.28) and a high air temperature (+1.28).
  - Power: high speed (+1.26) and high torque (+0.86).
  - Overstrain: very high torque (+1.69) together with high tool wear (+1.58).
  - Tool Wear: high tool wear (+1.71), and almost nothing else is unusual.
  - Random Failures: every value is close to normal (largest +0.51), so they are almost indistinguishable from healthy machines. This makes sense because they are random by design.
- **Correlations:** air and process temperature are strongly positively correlated (0.88), and rotational speed and torque are strongly negatively correlated (-0.87), which follows from physics (power = torque x speed). All other pairs are close to 0. A neural network does not suffer from this redundancy, so I kept all features.
- **Product quality matters:** low-quality (L) machines fail more often (4.05% vs 2.71% for M and 2.40% for H), and Overstrain is about 10 times more frequent in L (1.22%) than in M or H (about 0.1%). This supports the ordinal encoding.
- **Very few samples in the smallest classes.** The stratified split gives Random Failures 13 / 2 / 3 rows and Tool Wear 32 / 7 / 6 rows (train / validation / test), so metrics on these classes will be noisy.

## 3. Data Augmentation (SMOTE)

Because the failure classes are so rare, the network sees almost no failures and learns to always say "No Failure". I balanced the training set with **SMOTENC** (SMOTE for nominal and continuous features, k = 5 neighbours):

- SMOTE creates a new sample on the line between a real minority sample and one of its nearest neighbours of the same class.
- **SMOTENC instead of plain SMOTE** so that `Type` stays a valid category (0, 1 or 2) instead of becoming something like 0.37.
- **Applied to the training set only, after scaling.** If synthetic samples were created from validation or test rows, those sets would no longer be independent and results would look better than they really are. It comes after scaling because SMOTE uses Euclidean distance, so features must be on comparable scales.
- Result: training set 6,993 rows to **40,494 rows** (33,501 synthetic), with **6,749 rows in every class**. Validation and test were left untouched, with the real 96.5% / 3.5% distribution.
- Sanity checks passed: the original rows are unchanged, `Type` only contains {0, 1, 2}, and there are no NaN values.

A limit worth knowing: SMOTE creates no new information. Random Failures goes from 13 real rows to 6,749 samples that are all interpolations of those 13, and since the real ones already look like normal machines, so do the synthetic ones.

## 4. Model Architecture (PyTorch)

Task: use the 6 features of a machine to predict one of 6 failure classes.

- **Model:** MLP 6 -> h1 -> h2 -> 6 with ReLU in the hidden layers. The output layer gives raw scores (logits) with **no softmax inside the model**, because `CrossEntropyLoss` already applies log-softmax internally (adding a second softmax would hurt training).
- **Loss / optimizer:** CrossEntropyLoss, Adam for the first baseline.
- **Mini-batch training** with `DataLoader` (shuffled, because SMOTE appends the rows grouped by class).
- **Baseline model:** hidden layers (64, 32), Adam, lr = 0.001, 50 epochs, batch size 64, **2,726 trainable parameters**.
- **Sanity check:** before training the loss was 1.797, very close to ln(6) = 1.792, which is what a model that guesses randomly among 6 classes should give.
- Baseline after 50 epochs: train accuracy 99.2%, validation accuracy 94.4%.

## 5. Hyperparameter Tuning (GridSearchCV + skorch)

skorch wraps the PyTorch model so scikit-learn's `GridSearchCV` can be used, including the architecture.

- **Fixed split instead of k-fold (`PredefinedSplit`):** the model trains on the SMOTE training set and is scored on the **real, untouched validation set**. With normal k-fold, a synthetic sample could land in the training fold while its real "parent" lands in the validation fold, which would inflate the scores.
- **`refit=False`:** by default GridSearchCV retrains the best model on all the data it was given, which here would include the validation rows. I retrained the winner myself on the training set only.
- **Scoring: macro F1** (the average of the F1 of every class). Accuracy is dominated by "No Failure" and would pick a model that ignores failures. Macro recall and accuracy were also recorded.
- **Grid (36 combinations):** optimizer {Adam, RMSprop, SGD with momentum 0.9}, learning rate {0.001, 0.01} for Adam/RMSprop and {0.01, 0.1} for SGD (each optimizer needs its own range: plain SGD is far too slow at 0.001), epochs {30, 60}, architecture {(32), (64, 32), (128, 64, 32)}. Batch size was fixed to 128.

**Best parameters:** SGD (momentum 0.9), lr = 0.1, 30 epochs, hidden layers (64, 32). Validation macro F1 about 0.594 and validation accuracy about 0.959.

Findings:

- **The top 10 configurations are almost tied** (macro F1 between 0.577 and 0.594) and include all three optimizers and all three architectures. So the choice of optimizer or depth hardly matters here; the limit comes from the data (very few failures), not from the network.
- Validation accuracy stays between 93% and 97% for almost every configuration, which again shows that accuracy cannot separate good models from bad ones.
- The validation set has only 2 Random Failures and 7 Tool Wear Failures, so the "winner" may partly be luck.
- The best configuration is also fast (about 20 seconds per fit).

## 6. Training Curves

Loss (cross-entropy) and accuracy vs epochs were plotted for train, validation and test.

Interpretation:

- **Training** loss falls quickly (from about 0.15 to about 0.03) and training accuracy reaches about 99%, so the network learns the balanced training set very well.
- **Validation and test loss** stay between about 0.25 and 0.45 and zig-zag. The minimum validation loss is at epoch 18 (about 0.24), and there is no real improvement after epoch 10. There is no dramatic overfitting, but the model does not keep improving either.
- Validation and test curves move together, so tuning on the validation set did not overfit it. Final test loss (0.351) is a bit higher than validation loss (0.300).
- **Test accuracy climbs from about 81% to about 95.7%** but stays just **below the "always No Failure" line (96.5%)**. That does not mean the model is useless: it is trained on balanced data, so it believes failures are more common than they really are and raises false alarms on healthy machines. Section 7 measures this.
- **The curves are jagged** for two reasons: SGD with a large learning rate (0.1) takes big steps, and the real validation/test sets have only a handful of samples for some classes, so a single prediction changes the numbers.
- **The gap between train and test accuracy should not be read as pure overfitting**, because train is balanced (a dumb model would score about 17%) while test is real (a dumb model would score 96.5%). The fair comparison is "train (real rows only)" vs "test", shown next.

## 7. Metrics

Because of the imbalance, accuracy alone is misleading. I report accuracy, balanced accuracy (= macro sensitivity/recall), macro F1, the share of real failures detected (any type), and the false alarm rate (healthy machines flagged as failing).

| Model | Set | Accuracy | Balanced acc. | Macro F1 | Failures detected | False alarms |
|---|---|---|---|---|---|---|
| Naive (always "No Failure") | Test | 0.9653 | 0.1667 | 0.1637 | 0.0% | 0.0% |
| Baseline MLP | Train (balanced, SMOTE) | 0.9923 | 0.9923 | 0.9923 | 99.97% | 4.06% |
| Baseline MLP | Train (real rows only) | 0.9595 | 0.9448 | 0.6760 | 96.3% | 4.06% |
| Baseline MLP | Test | 0.9460 | 0.6018 | 0.5159 | 82.7% | 4.56% |
| Tuned MLP (Model 1) | Train (balanced, SMOTE) | 0.9938 | 0.9938 | 0.9937 | 99.9% | 3.20% |
| Tuned MLP (Model 1) | Train (real rows only) | 0.9678 | 0.9720 | 0.7164 | 96.3% | 3.20% |
| Tuned MLP (Model 1) | Validation | 0.9506 | 0.5724 | 0.5744 | 67.3% | 3.94% |
| Tuned MLP (Model 1) | Test | 0.9573 | 0.5585 | 0.5304 | 71.2% | 3.11% |

Per-class results of the tuned model on the test set:

| Class | Support | Precision | Recall (sensitivity) | F1 |
|---|---|---|---|---|
| No Failure | 1447 | 0.989 | 0.969 | 0.979 |
| Heat Dissipation | 17 | 0.882 | 0.882 | 0.882 |
| Power | 14 | 0.500 | 0.500 | 0.500 |
| Overstrain | 12 | 0.667 | 0.833 | 0.741 |
| Tool Wear | 6 | 0.053 | 0.167 | 0.080 |
| Random Failures | 3 | 0.000 | 0.000 | 0.000 |

Interpretation:

- **Accuracy (95.7%) looks great but is not.** It is lower than the 96.5% of a model that always answers "No Failure". The honest numbers are balanced accuracy (0.56) and macro F1 (0.53).
- **What the model does in practice:** it flags **71% of the real failures** (37 of 52 in the test set) and raises **3.1% false alarms** (45 healthy machines out of 1,447). About 15 real failures were missed.
- **Best classes:** Heat Dissipation (F1 0.88) and Overstrain (F1 0.74). These have a clear sensor signature in the EDA heatmap (low speed + high torque, and high torque + high wear).
- **Power is medium (F1 0.50)**, **Tool Wear is very poor (F1 0.08):** only 1 of the 6 real Tool Wear cases was caught while about 18 healthy machines were wrongly labelled Tool Wear. Its only clue is high tool wear, which many healthy and Overstrain machines also have.
- **Random Failures: 0 of 3 detected.** This is expected and not a bug: they look exactly like normal machines, so no model can find them from these sensors.
- **There is real overfitting.** On real training rows the macro F1 is 0.72 but on the test set it is 0.53. The model even finds 100% of the Random Failures in the training rows (13 of 13) and 0% in the test rows, which means it memorized those few points (and their SMOTE copies). Precision is also low on training rows for the rare classes (Tool Wear 0.25, Random 0.13): oversampling made the model predict these classes too often.
- **Baseline vs tuned:** the tuned model has a slightly better macro F1 (0.530 vs 0.516) and fewer false alarms (3.1% vs 4.6%), but the baseline finds more failures (82.7% vs 71.2%). The F1 difference is tiny, so they are basically tied, with a different precision/recall trade-off.
- **Per-class results are fragile:** one single prediction moves the recall by 7% (Heat Dissipation) up to 33% (Random Failures).

## 8. Regularization and Comparison

Techniques tested (on the same architecture, optimizer, learning rate, epochs and seed as Model 1, so only the regularizer changes): L2 weight decay (1e-3), dropout (p = 0.3), batch normalization, early stopping on validation loss (patience 10, best weights restored), and a combination of dropout + L2 + early stopping. Early stopping stopped at epoch 28 (best epoch 18), and the combination at epoch 21 (best epoch 11). The other variants ran the full 30 epochs.

| Model | Val macro F1 | Train macro F1 (real rows) | Test macro F1 | Train-test gap | Test accuracy | Balanced acc. | Failures detected | False alarms |
|---|---|---|---|---|---|---|---|---|
| Model 1 (no regularization) | 0.5744 | 0.7164 | 0.5304 | 0.186 | 0.9573 | 0.5585 | 71.2% | 3.1% |
| L2 (wd = 1e-3) | 0.5437 | 0.5397 | 0.4892 | 0.051 | 0.8773 | 0.6155 | 84.6% | 11.8% |
| Dropout (p = 0.3) | 0.5274 | 0.5273 | 0.4820 | 0.045 | 0.8806 | 0.6749 | 88.5% | 11.8% |
| Batch normalization | 0.5816 | 0.6417 | 0.5213 | 0.120 | 0.9113 | 0.6034 | 80.8% | 8.2% |
| Early stopping | 0.5449 | 0.6997 | 0.5062 | 0.194 | 0.9526 | 0.5185 | 67.3% | 3.4% |
| Dropout + L2 + early stopping | 0.4307 | 0.4308 | 0.3927 | 0.038 | 0.8439 | 0.6511 | 88.5% | 15.4% |

Interpretation:

- **Regularization did its job on overfitting:** L2, dropout and the combination shrink the train-test gap from 0.186 to about 0.04 to 0.05. But the gap closed mostly because the **training score went down** (0.72 to about 0.53), not because the test score went up.
- **No technique clearly beat Model 1 on macro F1.** Batch normalization was chosen by validation macro F1 (0.582), but its advantage over Model 1 on validation is only 0.007 (noise), and on the test set it is slightly below Model 1 (0.521 vs 0.530). Early stopping gave almost the same result as Model 1, which makes sense because the validation curve was flat after epoch 10.
- **The real effect is a different trade-off.** L2, dropout and the combination catch **more failures** (85% to 88% instead of 71%) but they also raise **more false alarms** (12% to 15% instead of 3%), so accuracy drops to 84% to 88%. They make the model less sure of itself, so it flags more machines as failing.
- **Which one is "best" depends on the cost.** If a missed failure (unplanned breakdown) is much more expensive than an extra inspection, dropout is attractive (88.5% detected, best balanced accuracy 0.675). If inspections are expensive, Model 1 or early stopping are better (about 3% false alarms).
- **The combination was too much:** it stopped early (epoch 21) and has the lowest macro F1 (0.39) and the most false alarms (15.4%). Stacking three regularizers on a tiny network (2,726 parameters) caused underfitting.
- **Why regularization could not fix much:** the real problem is the data (13 real Random Failure rows and 32 Tool Wear rows). Regularization can smooth the model, but it cannot create information that is not in the data.

## Limitations

- **Tiny test classes.** Test has only 3 Random Failures, 6 Tool Wear, 12 Overstrain, 14 Power and 17 Heat Dissipation samples. Per-class metrics are very noisy, and differences smaller than about 0.02 in macro F1 should be read as ties.
- **Single run with one random seed and one split.** The hyperparameter search was also decided on a small validation set, so the "best" configuration is probably optimistic.
- **SMOTE is aggressive.** Going from 13 to 6,749 samples (full balancing) produces many false alarms and a model that memorizes the few real points. I did not compare it with partial oversampling or class weights.
- **Random Failures are unpredictable by construction** in this dataset, so no model can solve that class.
- **No decision-threshold tuning.** Because the model was trained on balanced data, it overestimates how common failures are. Adjusting the decision threshold could reduce the false alarms.
- The early stopping variants restore the best-validation weights, while Model 1 reports its last epoch, so part of their behaviour comes from checkpointing.
- The grid search used a fixed batch size (128), and L2/dropout/batch norm strengths were not tuned.

## What I Learned

- On imbalanced data, **accuracy can be completely misleading**: 95.7% accuracy was actually worse than a model that does nothing. Macro F1, recall per class and the confusion matrix tell the truth.
- The order of the pipeline matters: **split first, then scale, then SMOTE**, and only on the training set, otherwise information leaks from the test set.
- SMOTE helps the model notice failures, but it also creates false alarms and cannot invent information that is not in the data.
- Normal cross-validation can be wrong when the training data contains synthetic samples, so I used a fixed validation split with `PredefinedSplit` in GridSearchCV with skorch.
- More regularization is not always better: it reduced overfitting but also reduced performance, and it mostly moved the balance between missed failures and false alarms.
- With very few samples in some classes, I have to be careful about what I conclude from the metrics.

To improve the model, I could try class weights or a partial SMOTE, tune the decision threshold according to the cost of missed failures, repeat the experiments with several seeds (or stratified cross-validation with SMOTE inside a pipeline), add engineered features (temperature difference and power = torque x speed), and compare with tree-based models such as Random Forest or XGBoost.

## How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn torch skorch
```

1. Download `predictive_maintenance.csv` from the Kaggle dataset (shivamb/machine-predictive-maintenance-classification) and place it next to the notebook (or update the path).
2. Open `multiclass_classification.ipynb` and run all cells from top to bottom .

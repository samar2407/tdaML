# Ensemble Learning: Model Comparison
### XGBoost vs AdaBoost vs Random Forest

---

## Overview

This project compares three ensemble learning models on different tasks:

| Model | Task | Dataset |
|---|---|---|
| XGBoost | Regression | Metro Interstate Traffic Volume |
| AdaBoost | Classification | Synthetic (make_classification) |
| Random Forest | Regression | Metro Interstate Traffic Volume |

---

## Models

### 1. XGBoost (`xgb.ipynb`)
XGBoost (Extreme Gradient Boosting) is a sequential boosting algorithm that builds trees one at a time, where each new tree corrects the residual errors of the previous ones. It uses second-order gradients, regularization (L1/L2), and subsampling to prevent overfitting.

**Implementation:** `xgboost` library  
**Task:** Regression — predicting hourly traffic volume  
**Key hyperparameters (tuned via GridSearchCV):**
- `max_depth = 4`
- `learning_rate = 0.1`
- `gamma = 0.25`
- `reg_lambda = 10`
- `subsample = 0.9`
- `colsample_bytree = 0.5`

**Results:**

| Model | RMSE | R² |
|---|---|---|
| Preliminary XGBoost | 411.46 | 0.9572 |
| Optimized XGBoost | 484.96 | 0.9405 |

**Top features by gain:** `hour`, `is_weekend`, `temp`

---

### 2. AdaBoost (`adab.ipynb`)
AdaBoost (Adaptive Boosting) is a sequential boosting algorithm that trains weak learners (decision stumps) one at a time. Each subsequent learner focuses more on the samples that the previous one misclassified, by updating sample weights. The final prediction is a weighted vote of all weak learners.

**Implementation:** Built from scratch using NumPy  
**Task:** Binary Classification — synthetic dataset  
**Key parameters:**
- `n_estimators = 50`
- Base learner: `DecisionTreeClassifier(max_depth=1)` (decision stump)
- Weight update: `w *= exp(-alpha * y * predictions)`
- Alpha formula: `alpha = 0.5 * log((1 - err) / err)`

**Results:**

| Metric | Score |
|---|---|
| Accuracy | 84.0% |
| Precision | 0.8365 |
| Recall | 0.8581 |
| F1 Score | 0.8471 |
| ROC-AUC | 0.8394 |

---

### 3. Random Forest (`rando.ipynb`)
Random Forest is a bagging algorithm that trains multiple decision trees independently and in parallel on random subsets of the data (bootstrap sampling) and random subsets of features. The final prediction is the average of all trees (regression) or majority vote (classification).

**Implementation:** `sklearn` library  
**Task:** Regression — predicting hourly traffic volume  
**Key parameters:**
- `n_estimators = 100`
- `oob_score = True`
- Features used: `temp`, `rain_1h`, `snow_1h`, `clouds_all`

**Results:**

| Metric | Score |
|---|---|
| OOB Score | -0.0135 |
| MSE | 3,981,716.61 |
| R² | -0.0071 |

> **Note:** The poor R² is due to limited feature selection — only 4 numeric features were used, excluding the most important ones like `hour`, `day_of_week`, and `month`.

---

## Comparison

| | XGBoost | AdaBoost | Random Forest |
|---|---|---|---|
| Type | Boosting | Boosting | Bagging |
| Trees built | Sequentially | Sequentially | Independently (parallel) |
| Focus | Residual errors (gradients) | Misclassified samples (weights) | Random subsets of data & features |
| Regularization | L1 + L2 built-in | None | Via feature/sample randomness |
| Implementation | Library | From scratch | Library |
| Task | Regression | Classification | Regression |
| Best metric | R² = 0.9572 | Accuracy = 84% | R² = -0.007 (limited features) |

---

## Key Takeaways

- **XGBoost** performed best on the traffic volume dataset, with R² of 0.957 even before tuning. The most important features were `hour`, `is_weekend`, and `temp` — confirming that time-of-day is the dominant factor in traffic patterns.
- **AdaBoost** implemented from scratch correctly replicated the boosting logic — 84% accuracy on a synthetic binary classification task with 50 weak learners.
- **Random Forest** underperformed due to feature selection — only 4 weather features were used, missing the critical time-based features. With `hour`, `day_of_week`, and `month` included, performance would be significantly better.
- Boosting methods (XGBoost, AdaBoost) generally outperform bagging (Random Forest) when features are informative and data is structured.

---


## Dataset

**Metro Interstate Traffic Volume**  
Source: UCI Machine Learning Repository  
Records: 48,204 hourly observations (2012–2018)  
Target: `traffic_volume` (vehicles per hour)  
Features: temperature, weather conditions, holiday, date/time

---

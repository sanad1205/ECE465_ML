# IECE 465 – Introduction to Machine Learning for Engineers
## Final Project: Marathon Finish Time Prediction

🌐 **Live site:** [https://sanad1205.github.io/marathon-ml/](https://sanad1205.github.io/marathon-ml/)

---

## 📌 Project Overview

This project predicts net marathon finish times using machine learning applied to the 2023 Boston Marathon dataset (26,598 runners). Given a runner's age group, gender, and half-marathon split, the system estimates their final net finish time, then ranks four candidate models, automatically selects the best one by RMSE, and exports a fully reproducible package of results.

---

## 📊 Dataset

- **Source:** 2023 Boston Marathon
- **Samples:** 26,598 runners (after cleaning)
- **Features:**
  - `age_group` (categorical, 11 bins from 18–24 to 70+)
  - `gender` (categorical: M / F)
  - `half_time_sec` (numerical)
- **Target:**
  - `finish_net_sec` (numerical)

---

## ⚙️ Methodology

An eight-stage, fully reproducible pipeline (`rng seed = 42`):
**Load → Clean → Split → Encode → Train → Evaluate → Select → Export**

### 1. Data Preprocessing
- Removed missing entries with `rmmissing`
- Filtered to positive race times only (`half_time_sec > 0 & finish_net_sec > 0`)
- Removed unrealistic finish/half ratios (kept `1.80 ≤ ratio ≤ 3.20`) to drop DNFs encoded as zero and pacing patterns inconsistent with finishing a marathon
- 80 / 20 train–test holdout
- One-hot encoded categoricals using category lists defined from the full dataset (so test folds never contain unseen categories)
- Standardized numerical features using **train-only** mean/std (no leakage)

### 2. Model Development
Four candidate models trained head-to-head:

| # | Model | Notes |
|---|---|---|
| ① | **Baseline Ratio** | `finish ≈ half × avgRatio` — establishes the floor |
| ② | **Linear Regression** | `fitlm` with one-hot categoricals |
| ③ | **Neural Network** | `fitrnet`, layers `[64 32 16]`, ReLU activations |
| ④ | **Bagged Trees Ensemble** ⭐ | `fitrensemble`, 150 learners, min-leaf-size 10 |

### 3. Model Evaluation — Nine Metrics
- Mean Absolute Error (MAE), in seconds and minutes
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Median Absolute Error (robust to outliers)
- Practical accuracy bands: % of predictions within ±5, ±10, ±15 minutes
- Auto-rank: all metrics propagate into the model leaderboard

### 4. Auto Best-Model Selection
The pipeline programmatically selects the winner by RMSE and exports a saved `.mat` model package for reuse.

---

## 📈 Results

All four models were ranked on the 20 % held-out test set:

| Place | Model | MAE (min) | RMSE (min) | R² | ≤ 10 min |
|:-:|---|:-:|:-:|:-:|:-:|
| 🥇 | **Bagged Trees Ensemble** | 6.31 | **9.42** | 0.953 | 78 % |
| 🥈 | Linear Regression | 6.44 | 9.70 | 0.951 | 77 % |
| 🥉 | Neural Network | 6.39 | 9.75 | 0.950 | 77 % |
| 4 | Baseline Ratio | 6.83 | 10.11 | 0.947 | 72 % |

**Headline numbers:** RMSE = 9.42 min, R² = 0.953, 78 % of predictions within ±10 minutes. The ensemble wins by a small margin — but linear regression remains within ~0.3 min of it and is far easier to explain.

---

## 📊 Visualizations
- Model leaderboard / RMSE comparison chart
- Practical accuracy bands (% within ±5 / ±10 / ±15 minutes)
- Over/under decision accuracy by benchmark finish time
- Actual vs. predicted plots and residual analysis
- Performance breakdown by age group and gender

---

## 🎯 Key Insights
1. **One feature dominates.** The half-marathon split alone explains most of the variance in final finish time. Adding age and gender helps, but the gain is marginal.
2. **Simple beats complex when the signal is linear.** The ensemble edged ahead, but linear regression came within ~0.3 min RMSE — and is far easier to explain.
3. **Units shape interpretation.** Reporting errors in seconds was useless to runners. Switching to minutes plus within-X-minute bands made results actionable.
4. **Reproducibility is a feature.** Saving every output (results table, predictions, threshold table, figures, `.mat` package) turned a working script into a system anyone can re-run cleanly.

---

## 🔍 Extensions
**Sports-betting-style over/under decisions:**
The regression output is converted into a binary "over/under" decision against benchmark finish times (3:00, 3:30, 4:00, 4:30, 5:00). This translates continuous error into actionable predictions and reaches **94–98 % decision accuracy** across thresholds.

---

## 🛠️ Tools & Technologies
- MATLAB (primary implementation)
- Statistics and Machine Learning Toolbox (`fitlm`, `fitrnet`, `fitrensemble`)
- Reproducible export to `.mat`, CSV, and figure files
- Project website: HTML / CSS / JavaScript (single-file, fully self-contained)

---

## 📂 Repository Contents
- `index.html` — interactive project website (also live at the URL above)
- `marathon_models.m` — full end-to-end MATLAB pipeline
- `boston_marathon_2023.csv` — cleaned dataset
- `Final_Presentation_Marathon_ML.pdf` — final presentation deck
- `Marathon_ML_Website.pdf` — printable PDF version of the website

---

## 👥 Team
- **Sanad Sahawneh** — ML System Lead (pipeline integration, evaluation framework, auto best-model selection, visualization suite)
- **Brian Allen** — Data & Presentation (ML code skeleton, preprocessing & cleaning v2, outlier filter design, presentation prep)
- **Mahib Rahman** — Modeling & Validation (feature encoding & normalization, model tuning, performance validation, threshold extension validation)

---

## 📅 Course Information
**Course:** IECE 465 – Introduction to Machine Learning for Engineers
**Semester:** Spring 2026
**Faculty Mentor:** Dr. Hany Elgala
**Teaching Assistant:** Honglan Chen
**Institution:** University at Albany, SUNY

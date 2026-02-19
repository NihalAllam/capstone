# Diabetes Imputation Study – Robustness Analysis Framework

## Objective

To analyze how different imputation strategies and machine learning models behave under increasing levels of missing data in the PIMA Diabetes dataset.

This project began by rebuilding the experimental pipeline to eliminate data leakage and incorrect preprocessing from earlier implementations. It now extends into a controlled missingness robustness study.

---

## Phase 1 – Clean Baseline

We rebuilt the pipeline from scratch and applied the following fixes:

1. Replaced medically invalid zero values with NaN.
2. Performed train-test split before imputation.
3. Fitted imputation models only on training data.
4. Evaluated using Accuracy, Recall, and ROC-AUC.

Baseline comparisons completed:

- Raw Data (zeros kept)
- Drop Rows
- Mean Imputation
- Median Imputation
- Mode Imputation
- KNN Imputation

Models evaluated:
- Logistic Regression
- Random Forest

Observation:
Imputation methods showed only marginal differences in performance. Model choice influenced results more than simple imputation strategy.

---

## Phase 2 – Robustness Under Missingness (Current Direction)

Most PIMA-based studies compare imputation methods on a fixed dataset.

Our focus shifts to:

**How do model–imputation combinations behave when missing data increases?**

Planned experiment:

1. Artificially introduce missing data at:
   - 10%
   - 20%
   - 30%
   - (possibly 40%)

2. Apply imputation methods:
   - Mean
   - Median
   - KNN
   - (Optional: MICE)

3. Evaluate models:
   - Logistic Regression
   - Random Forest
   - XGBoost

4. Measure:
   - Accuracy
   - Recall (primary metric)
   - ROC-AUC

Goal:
To identify which combinations remain stable and reliable under increasing missingness, particularly in medical contexts where incomplete data is common.

---

## Why This Direction?

Previous PIMA research typically:
- Compares models
- Compares imputation methods
- Reports accuracy improvements

Few studies deeply analyze:
- Performance degradation under increasing missingness
- Model sensitivity to imputation
- Stability of recall in medical prediction

This project aims to address that gap.

---

## Project Structure

```text
diabetes-imputation-project/
|
├── data/
│ └── PIMA.csv
│
├── notebooks/
│ └── 01_clean_pipeline.ipynb
│
├── src/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── modeling.py
│ └── evaluation.py
│
└── PROJECT_LOG.md
```

---

## Current Status

- Clean leakage-free baseline completed
- Logistic Regression and Random Forest evaluated
- Imputation comparison documented
- Research direction finalized
- Robustness experiment implementation starting

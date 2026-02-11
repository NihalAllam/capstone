# Diabetes Imputation Study – Clean Experimental Pipeline

## Objective

To evaluate how proper handling of invalid zero values affects diabetes classification performance using the PIMA dataset.

This project rebuilds the experimental pipeline from scratch to eliminate data leakage and incorrect preprocessing from earlier implementations.

---

## Core Problem

In the PIMA dataset, missing values are encoded as zeros in specific medical columns.

Invalid zero columns:
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Leaving zeros untreated causes the model to learn medically incorrect patterns.

---

## Key Fixes Applied

1. Replace invalid zeros with NaN.
2. Perform train-test split before imputation.
3. Fit imputer only on training data.
4. Evaluate using Accuracy, Recall, and ROC-AUC.

---

## Experimental Scenarios

1. Raw Data (zeros kept)
2. Drop rows with missing values
3. Mean Imputation (correct implementation)

---

## Project Structure

```
diabetes-imputation-project/
│
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

## Current Focus

Establishing a clean, leakage-free baseline before expanding to:

- Advanced imputation (KNN, MICE)
- Ensemble methods
- Controlled missingness experiments
- Statistical validation

---

## Status

Baseline pipeline implemented.
Further expansion planned after validation of initial results.

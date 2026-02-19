# Project Log – Ensemble Learning Framework for Improved Diabetes Classification

---

# 1. Objective

To study how different missing-value imputation strategies affect diabetes classification performance, and whether a properly designed ensemble framework improves reliability over naive preprocessing.

---

# 2. What We Inherited

We received:
- PIMA dataset
- Local Healthcare dataset
- Multiple notebooks (Mean, Median, Mode, Proposed)

After reviewing the code, we observed conceptual and experimental issues.

### 2.5 Their results (Forgot to focus on this before, did in week 2)

They did outlier detection and replaced them with imputation values.

### - Pima results:
```mathematica
Logistic Regression Accuracy: 0.74
Decision Tree Accuracy: 0.65
Random Forest Accuracy: 0.74
SVM Accuracy: 0.71
KNN Accuracy: 0.68
```

---

# 3. Identified Problems in Previous Implementation

## Problem 1 (Critical): Zeros Treated as Missing Without Justification
All zero values were treated as missing, even in columns where zero can be valid (e.g., Pregnancies).

## Problem 2: Imputation Before Train-Test Split
Imputation was performed before splitting the data, causing data leakage.

## Problem 3: No Clear Baseline
There was no comparison against:
- Raw data (no imputation)
- Dropping rows with missing values

## Problem 4: Over-Reliance on Accuracy
Accuracy was the main metric used, which is weak for medical classification. Recall is the better metric (explained later).

## Problem 5: No Statistical Validation
No confidence intervals or statistical tests were performed.

## Problem 6: No Clear Experimental Control
Different notebooks used slightly different setups, making comparison unreliable.

---
# 4. Why Recall Matters Most

In diabetes classification, the real risk isn’t predicting someone has diabetes when they don’t,
it’s failing to detect someone who actually does.

A false negative means a patient walks away undiagnosed.  
No early intervention. No monitoring. No lifestyle correction.  
And diabetes is a condition where delayed action directly increases long-term complications.

That’s why recall is the priority in this project.

We are testing different imputation strategies. If an imputation method slightly improves accuracy but lowers recall, it is not acceptable. The goal is not just better overall scores — it is preserving the model’s ability to correctly identify diabetic patients after handling missing data.

In short:

> Missing a diabetic case is costlier than raising a false alarm.  
> Therefore, recall is the metric that matters most.

---

# 5. Thought Process Evolution

We considered multiple approaches:

## Option A – Remove Rows with Missing Data
Pros:
- Simple
- No imputation bias

Cons:
- Reduces dataset size
- May remove useful data

## Option B – Train As-Is
Keep zeros and train directly.

Pros:
- No assumptions

Cons:
- Model learns incorrect values
- Medically invalid inputs

## Option C – Simple Imputation
Mean / Median / Mode replacement.

Pros:
- Easy baseline
- Common in literature

Cons:
- Ignores feature relationships

## Option D – Advanced Imputation
KNN, MICE, or literature-based method.

Pros:
- More realistic
- Potentially better performance

Cons:
- More complex

---

# 6. Baseline Design (Very Important)

We will create:

1. Baseline 1: Raw data (zeros kept)
2. Baseline 2: Drop rows with invalid zeros
3. Baseline 3: Mean/Median imputation (properly after train-test split)
4. Advanced imputation methods

---

# 7. Research Question (Locked)

How do different missing-value imputation strategies affect ensemble classifier performance for diabetes prediction under controlled missingness conditions?

---

# 8. Non-Negotiable Next Steps

## Step 1: Redesign Experiment
- Train-test split first
- Then imputation
- No leakage

## Step 2: Compare Imputation Methods
- Mean
- Median
- KNN
- MICE

## Step 3: Proper Ensemble Usage
- Voting
- Stacking
- Explain each base model’s purpose

## Step 4: Proper Evaluation
Report:
- Accuracy
- Recall (Sensitivity)
- Specificity
- ROC-AUC
- Confidence intervals

---

# 9. Final Direction

The final goal is to produce:

- A clean experimental pipeline
- A literature-supported imputation comparison
- A justified ensemble framework
- A reproducible and publishable structure

---

# 10. Weekly Progress Tracking

(We will update this section weekly with observations, failures, improvements, and results.)

# Week 1 Update

- Reviewed inherited project issues.
- Rebuilt preprocessing pipeline from scratch.
- Implemented three baseline scenarios:
  1. Raw data
  2. Drop rows with invalid zeros
  3. Mean imputation (after train-test split)
- Evaluated using Accuracy, Recall, ROC-AUC.
- Confirmed pipeline runs without data leakage.

## Baseline Results (Week 1)

| Method            | Accuracy | Recall |
|-------------------|----------|--------|
| Raw Data          | 74.67    | 67.27  |
| Drop Rows         | 77.21    | 59.26  |
| Mean Imputation   | 75.32    | 61.82  |
| Median Imputation | 75.97    | 63.64  |
| Mode Imputation   | 75.79    | 63.64  |
| KNN Imputation    | 76.62    | 63.64  |

### Baseline Observations from table

- Raw data has higher recall because keeping zeros makes the model more likely to predict positives.

- Dropping rows reduces recall since fewer diabetic samples remain for training.

- Drop Rows shows highest accuracy but lowest recall because it predicts negatives more confidently while missing true positives.

- Accuracy alone is misleading — it can increase even when the model fails to detect real diabetic cases.

# Week 1 – Phase 2 Update

- Implemented KNN imputation (n_neighbors=5).
- Compared KNN against Raw, Drop, Mean, Median, Mode.
- Evaluated using Accuracy and Recall.
- Observed performance differences across strategies.

# Week 2 – Direction Finalized

## Status Update

After completing baseline comparisons (Raw, Drop, Mean, Median, Mode, KNN) with Logistic Regression and Random Forest, we observed:

- Imputation methods show only marginal performance differences.
- Median and KNN slightly outperform Mean.
- Model choice affects performance more than imputation choice.
- No dramatic accuracy improvements from basic imputation changes.

This indicates that simple imputation comparison alone is not a strong research contribution.

---

# Problem Reframed

Instead of asking:

"Which imputation gives best accuracy?"

We are now asking:

"How do different imputation strategies and models behave when missing data increases?"

In real medical datasets, missing values are common.  
So robustness under missingness is more important than minor accuracy gains.

---

# New Research Direction (Accepted)

We will perform a controlled missingness experiment on the PIMA dataset.

Plan:

1. Artificially introduce missing data at:
   - 10%
   - 20%
   - 30%
   - (possibly 40%)

2. Apply imputation methods:
   - Mean
   - Median
   - KNN
   - (MICE if feasible)

3. Train and evaluate models:
   - Logistic Regression (linear model)
   - Random Forest (tree-based model)
   - XGBoost (native missing handling)

4. Measure:
   - Accuracy
   - Recall (primary focus)
   - ROC-AUC

---

# Objective

To analyze:

- How performance degrades as missing data increases.
- Which imputation strategy is more stable.
- Whether tree-based models are less sensitive to missing data than linear models.
- Which model + imputation combination maintains recall under higher missingness.

This shifts focus from "improving raw accuracy" to "improving robustness and reliability."

---

# Expected Contribution

Unlike previous PIMA studies that only compare imputation methods on fixed datasets, we aim to:

- Study model–imputation interaction.
- Evaluate performance degradation under controlled missingness.
- Provide practical guidance for handling incomplete medical datasets.

---

# Why This Direction

Most PIMA-based papers:
- Compare models.
- Compare imputation methods.
- Report accuracy.

Few deeply analyze:
- Stability under increasing missing data.
- Clinical reliability (recall-focused evaluation).

We believe this direction is more meaningful and aligned with real-world medical scenarios.

---

# Next Immediate Tasks

- Implement MCAR missingness generator.
- Validate missingness injection correctness.
- Run 10% experiment first.
- Compare degradation across models.

---

Current Status:  
Baseline complete.  
Research direction finalized.  
Implementation phase starting.

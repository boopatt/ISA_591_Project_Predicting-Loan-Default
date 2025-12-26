
---

# Loan Default Prediction – End-to-End Modeling Project

The goal of this project was to predict **loan default** (`loan_default`) using historical applicant data across five structured phases, with a primary strategic focus on minimizing **False Negatives (FN)**, which represent high financial loss.

Project Team Members: Thenmozhi Boopathy and Aaron Black

Instructor: Dr. Allison Jones Farmer

---

## Project 1: Data Cleaning and Preparation - Aaron Black

The initial dataset contained **237,730 observations** and **36 variables**. This phase focused on transforming the data into a modeling-ready structure.

| Cleaning Step                | Variables / Features                                                                                                                                        | Rationale & Outcome                                                                                                                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Variable Removal**         | `installment`, `mths_since_last_delinq`, `emp_title`, `title`, `sub_grade`, `application_type`, `hardship_flag`, `address`, `acc_now_delinq`, `delinq_2yrs` | Removed `installment` due to high correlation with `loan_amnt` (R² = 0.95). Removed `mths_since_last_delinq` due to 53.22% missing values. Other variables were removed due to excessive levels or low predictive value.             |
| **Missing Value Handling**   | `mort_acc`, `pub_rec_bankruptcies`, `revol_util`                                                                                                            | `mort_acc` (22,854 missing) and `pub_rec_bankruptcies` (336 missing) were imputed using the median. 177 rows were removed due to missing `revol_util`.                                                                               |
| **Outlier Management**       | `revol_util`, `dti`                                                                                                                                         | Extreme outliers were capped using the IQR method.                                                                                                                                                                                   |
| **Dimensionality Reduction** | `home_ownership`, `purpose`, `emp_length`                                                                                                                   | Levels "OTHER" and "NONE" were removed from `home_ownership`. `purpose` was grouped into four categories (`debt_consolidation`, `credit_card`, `home_improvement`, `other`). `emp_length` was binned into four chronological groups. |
| **Feature Engineering**      | `dti`, `int_rate`                                                                                                                                           | Created interaction term `interaction_dti_interest = dti × int_rate`.                                                                                                                                                                |

---

## Project 2: Decision Tree Models - Thenmozhi Boopathy

This phase demonstrated the severe impact of class imbalance (182,927 “No” vs. 43,507 “Yes”).

| Tree Model            | Data Preparation        | Sensitivity | AUC        | Observation                                                        |
| --------------------- | ----------------------- | ----------- | ---------- | ------------------------------------------------------------------ |
| **Tree 1 (Default)**  | Unbalanced              | **0.0691**  | 0.5346     | Missed nearly all actual defaulters.                               |
| **Tree 3 (Balanced)** | Downsampled             | **0.6190**  | 0.6904     | Downsampling significantly improved detection of defaulters.       |
| **Tree 4 (Pruned)**   | Downsampled + 5-fold CV | **0.6431**  | **0.7153** | Pruned using CP = **0.0003064641** — best single-tree performance. |

By reviewing all the table metrics, xgbopt_cutoff has best in capturing the tp(Actual Defaulters),F1(capturing defaulters and avoiding false alarms) and AUC(Identify  the difference between defaulting and non-defaulting borrowers)

Recall = 0.64 → catches ~64% of actual defaulters
F1 = 0.44 → strong balance of precision & recall
AUC = 0.74 → reliable ranking capability
Best Model for Loan Defaults: xgbopt_cutoff

---

## Project 3: Logistic Regression with Elastic Net - Thenmozhi Boopathy

* **Training Data:** Downsampled dataset (43,507 per class).
* **Reference Level:** `loan_default = "Yes"`.
* **Tuning:** 10-fold CV optimizing ROC.

| Parameter      | Value     |
| -------------- | --------- |
| **α (alpha)**  | **0.1**   |
| **λ (lambda)** | **0.001** |
| **AUC**        | **0.73**  |

### Threshold Optimization

* Default cutoff (0.5): Sensitivity = 0.6550
* **Youden’s J optimal cutoff:** **0.4683588**

| Metric          | Value      |
| --------------- | ---------- |
| **Sensitivity** | **0.7149** |
| **Specificity** | 0.6149     |

This threshold was chosen to minimize **False Negatives**.

We checked with the two optimal cutoffs of 0.5187898 and 0.4683588 so we made decision with the level of 0.4683588 as possible threshold. So Youden threshold gives the best accuracy of finding the loan_default with its est in capturing the tp(Actual Defaulters) and minimizing fn (false negatives). Since our goal is to reduce missed defaulters, the Youden threshold achieves the best recall at an acceptable false-positive rate. Hence, using the selected youden’s threshold in logistic regression model, we achieve the better predictions in loan_default.

---

## Project 4: Neural Network Models - Thenmozhi Boopathy

All NN models used scaled and downsampled data.

| Model Variant       | Architecture                 | AUC       |
| ------------------- | ---------------------------- | --------- |
| **Basic NN**        | Single hidden layer (`nnet`) | **0.740** |
| **Keras NN**        | Two layers (64, 32 units)    | 0.738     |
| **NN-Logit Hybrid** | 14 logit-selected variables  | 0.718     |

**Best NN Model:** Basic NN
**Youden Cutoff:** 0.511
**Accuracy:** 68.92%
**Balanced Accuracy:** 0.6739

By comparing all the three models, basic nn_model and nnmodel with keras have AUC 74%. But while considering the context of loan default we have chosen the model based on confusion matrix and accuracy value. The reason for using youdens j statistic is, if we see our confusion matrix of nn_model my tp has 29009 and fn has 14498 and 67.77% accuracy. In the loan default scenario, minimizing fn and increasing tp is the target. so if i am using the youden’s J statistic my confusion matrix reveals the best posssible values in finding fn and tp are 15265 and 28242 with 68.92% accuracy. so for achieving the better results i choose this probabilty would be the best in performance. Using the youden’s J in basic nn_network model my AUC has 74%. So we would suggest considering basic nn_model for achieving the goal in loan_default.

---

## Project 5: Champion Model Selection

| Model               | AUC        |
| ------------------- | ---------- |
| Logistic Regression | 0.73       |
| Neural Network      | 0.74       |
| **XGBoost**         | **0.7415** |

### XGBoost Threshold Tuning

| Metric              | Cutoff    |
| ------------------- | --------- |
| Optimal F1          | 0.2266894 |
| Optimal Youden’s J  | 0.1886004 |
| **Selected Cutoff** | **0.20**  |

---

## 🏆 Champion Model – XGBoost @ Cutoff 0.20 - Thenmozhi Boopathy

| Metric                   | Value      | Rationale                              |
| ------------------------ | ---------- | -------------------------------------- |
| **AUC**                  | **0.7415** | Strong ranking capability.             |
| **Sensitivity (Recall)** | **0.6399** | Captures ~64% of defaulters.           |
| **F1 Score**             | **0.4425** | Balanced precision–recall tradeoff.    |
| **Accuracy**             | 0.6899     | Acceptable trade-off for FN reduction. |

**Final Selection:** XGBoost with optimized 0.2 cutoff — best balance between ranking power and minimizing costly False Negatives.

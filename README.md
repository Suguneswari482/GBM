# Gradient Boosting Machine (GBM) from Scratch — Regression Project

## 📌 Overview

This project implements a custom Gradient Boosting Regressor using only NumPy, without relying on high-level libraries like XGBoost or scikit-learn's GradientBoostingRegressor. It demonstrates a deep understanding of boosting algorithms, decision tree construction, and model evaluation through cross-validation.

The project also compares the performance of the custom GBM against scikit-learn’s implementation using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## 🧠 Objectives

- Implement core GBM logic for regression using Squared Error loss.
- Build simple decision trees (CART) as base learners using NumPy.
- Generate and preprocess a moderately complex regression dataset.
- Perform hyperparameter tuning across learning rate, number of estimators, and tree depth.
- Evaluate both models using 5-fold cross-validation.
- Analyze convergence behavior and accuracy trade-offs.

---

## 🛠 Technologies Used

- Python 3.9+
- NumPy
- scikit-learn (for dataset generation and benchmarking)
- StandardScaler (for feature normalization)

---

## 📂 Project Structure

- gbm_full_project.py: Main script containing all logic — data generation, model implementation, evaluation, and reporting.
- data/regression_dataset.csv: Generated dataset saved for reproducibility.
- gbm_comparison_report.md: Markdown report with performance tables and analysis.

---

## ⚙ How to Run

1. *Install dependencies*:
   ```bash
   pip install numpy scikit-learn

## Key Insights- Lower learning rates require more estimators but yield smoother convergence.
- Higher learning rates converge faster but risk overfitting with deeper trees.
- Increasing depth improves accuracy but can increase variance.
- Custom GBM achieves performance comparable to scikit-learn, validating the implementation.
 
## Deliverables- Custom GBM class code (NumPy-only)
- Full script for data generation, training, and evaluation
- Markdown report with metrics and analysis

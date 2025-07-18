# Comorbid Mental Disorder Risk Prediction System for Adolescents（TabTransformer-Based Prediction Framework）

This project implements and validates a **TabTransformer-based predictive framework** for assessing comorbid mental disorder risk in adolescents. By integrating **obsessive-compulsive symptom scores**, **standardized physical fitness data**, and **demographic and socioeconomic information**, the model demonstrates significantly superior performance compared to traditional machine learning baselines.

---

##  Project Overview

- Utilizes **multi-source data** including physical fitness tests, psychological questionnaires, and demographic variables.
- Compares various baseline classifiers such as **Logistic Regression**, **SVM**, **XGBoost**, and **Random Forest**.
- Leverages **TabTransformer** to process structured tabular features, enhanced with **early stopping** and **Focal Loss** for performance optimization.

---

##  Project Structure

```plaintext
mental-disorder-prediction/
│
├── src/
│   ├── data/                # Data preprocessing and loading
│   ├── models/              # TabTransformer & baseline model definitions
│   ├── train/               # Training scripts for transformer and baselines
│   ├── utils/               # Metrics & result export tools
│   ├── config.py            # Global configuration
│
├── README.md                # Project documentation
└── requirements.txt         # Required dependencies
```
---

##  Quick Start

###  1. Setup Environment (Recommended: Conda)

```bash
conda create -n mental python=3.10
conda activate mental
pip install -r requirements.txt
```

### 2. Train the TabTransformer Model

```bash
python -m src.train.train_transformer
```

### 3. Train Baseline Machine Learning Models

```bash
python -m src.train.train_baselines
```

## Model Performance Comparison

The **TabTransformer** outperforms traditional models in terms of **test accuracy**, **recall**, **F1-score**, and **multi-class AUC**.
<img width="7048" height="5255" alt="metrics_panel_labeled" src="https://github.com/user-attachments/assets/376534fb-22be-478e-a689-b682f65caa99" />

## Data Description (De-identified)

- **Original Data Source**: Adolescent physical fitness test data combined with SCL-90 psychological questionnaire results.
- **Features Include**:
  - **Physiological indicators**: BMI, vital capacity, 1000-meter run, etc.
  - **Psychological dimensions**: e.g., obsessive-compulsive symptom scores.
  - **Demographic & socioeconomic variables**: Gender, grade, family economic status, etc.

> Data Source: [National Population Health Data Archive (PHDA), China](https://www.ncmi.cn)

## License

This project is intended **for academic and research purposes only**. Commercial use is not permitted without prior authorization.

"""
Save the classification report and feature importance to an Excel file
"""

import os, pandas as pd
from datetime import datetime
from ..config import OUTPUT_DIR

# -------- Save the classification report --------
def save_results_to_excel(model_name, accuracy, class_report_dict, conf_matrix, auc_value=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    file = OUTPUT_DIR / "model_results.xlsx"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {'timestamp': ts, 'Model': model_name, 'Accuracy': accuracy, 'AUC': auc_value}

    for cls in ['0', '1', '2']:
        if cls in class_report_dict:
            row.update({f'F{cls}_{k}': v for k, v in class_report_dict[cls].items()})

    for avg in ['macro avg', 'weighted avg']:
        if avg in class_report_dict:
            prefix = 'macro' if avg == 'macro avg' else 'weighted'
            row.update({f'{prefix}_{k}': v for k, v in class_report_dict[avg].items()})

    df = pd.concat([pd.read_excel(file) if file.exists() else pd.DataFrame(), pd.DataFrame([row])])
    df.to_excel(file, index=False)
    print(f"[Excel] result -> {file}")

# -------- Feature importance --------
def save_feature_importance(model_name, feature_names, importance_values):
    OUTPUT_DIR.mkdir(exist_ok=True)
    file = OUTPUT_DIR / "feature_importance.xlsx"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new = pd.DataFrame({
        'timestamp': ts, 'model': model_name,
        'feature_name': feature_names,
        'importance': importance_values})
    df = pd.concat([pd.read_excel(file) if file.exists() else pd.DataFrame(), new])
    df.to_excel(file, index=False)
    print(f"[Excel] feature_importance -> {file}")

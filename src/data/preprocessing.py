"""
Data preprocessing
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from imblearn.over_sampling import SMOTE
import torch
from src.config import DATA_PATH, LABEL_MAP

def enhanced_mlp_preprocessing():
    data = pd.read_excel(DATA_PATH)
    data = data.dropna(subset=['labels']).reset_index(drop=True)

    cat_features = ['...'] # Categorical feature column name list
    num_features = ['...'] # Numerical feature column name list

    train_data, test_data = train_test_split(
        data, test_size=0.2, stratify=data['labels'], random_state=42)
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, stratify=train_data['labels'], random_state=42)

    preprocessor = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', QuantileTransformer(n_quantiles=1000, output_distribution='normal'))
        ]), num_features)
    ])

    preprocessor.fit(train_data)

    X_train = preprocessor.transform(train_data)
    X_val   = preprocessor.transform(val_data)
    X_test  = preprocessor.transform(test_data)

    y_train = train_data['labels'].map(LABEL_MAP).values
    y_val   = val_data['labels'].map(LABEL_MAP).values
    y_test  = test_data['labels'].map(LABEL_MAP).values

    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.long)

    return (X_train_t, y_train_t), (X_val_t, y_val_t), (X_test_t, y_test_t), preprocessor

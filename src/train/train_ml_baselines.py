"""
Train and evaluate multiple traditional machine learning baseline models
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

from src.data.preprocessing import enhanced_mlp_preprocessing
from ..utils.metrics import save_results_to_excel

MODELS = {
    "LogReg"     : LogisticRegression(max_iter=1000),
    "SVC"        : SVC(probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN"        : KNeighborsClassifier(),
    "XGBoost"    : XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
}

def main():
    (X_train, y_train), (_, _), (X_test, y_test), _ = enhanced_mlp_preprocessing()
    X_train, y_train = X_train.numpy(), y_train.numpy()
    X_test, y_test = X_test.numpy(), y_test.numpy()

    model_prob, cm_dict, pca_dict = {}, {}, {}
    for name, clf in MODELS.items():
        print(f"\n===== {name} =====")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc    = (preds == y_test).mean()
        cr_dict = classification_report(y_test, preds, output_dict=True)
        cm      = confusion_matrix(y_test, preds)
        prob    = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        auc_val = roc_auc_score(y_test, prob, multi_class='ovr') if prob is not None else None
        save_results_to_excel(name, acc, cr_dict, cm, auc_val)

        cm_dict[name] = cm
        if prob is not None:
            model_prob[name] = prob
            pca_dict[name] = PCA(2).fit_transform(prob)

if __name__ == "__main__":
    main()

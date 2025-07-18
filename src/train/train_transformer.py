"""
Training TabTransformer & Evaluate
"""

import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from torch.utils.data import DataLoader
from ..config import DEVICE, MAX_EPOCHS, PATIENCE, TEST_BATCH
from src.data.preprocessing import enhanced_mlp_preprocessing
from src.data.dataset import FeatureDataset
from ..models.tab_transformer import TabTransformer
from ..train.trainer import MLPTrainer
from ..utils.metrics import save_results_to_excel, save_feature_importance
from ..utils.visualization import save_figure

def main():
    # -------- Dataset --------
    (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor = enhanced_mlp_preprocessing()
    # print("train =", X_train.shape, "val =", X_val.shape, "test =", X_test.shape)
    train_ds = FeatureDataset(X_train, y_train)
    val_ds   = FeatureDataset(X_val, y_val)
    test_ds = FeatureDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH)

    # -------- Model & Trainer --------
    model = TabTransformer(input_dim=X_train.size(1), num_classes=3, nhead=8, dropout=0.2).to(DEVICE)
    trainer = MLPTrainer(model, train_ds, val_ds)

    # print("X_train.shape[1] =", X_train.shape[1])
    # print("model.pos_embedding.shape =", model.pos_embedding.shape)

    # -------- Training cycle --------
    best_val, no_improve = 0, 0
    for epoch in range(MAX_EPOCHS):
        loss = trainer.train_epoch(DEVICE)
        train_acc, _ = trainer.evaluate(trainer.train_loader, DEVICE)
        val_acc, _   = trainer.evaluate(trainer.val_loader,   DEVICE)
        trainer.scheduler.step(val_acc)
        print(f"Ep{epoch:03d} | loss={loss:.4f} | train={train_acc:.3f} | val={val_acc:.3f}")

        if val_acc > best_val:
            best_val, no_improve = val_acc, 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early‑stopping.")
                break

    # -------- Test --------
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    _, _ = trainer.evaluate(test_loader, DEVICE)
    test_acc, test_logits = trainer.evaluate(test_loader, DEVICE)
    print(f"[Test] acc={test_acc:.4f}")

    y_pred = test_logits.argmax(1).numpy()
    y_true = torch.cat([y for _, y in test_loader]).numpy()

    cr_dict = classification_report(y_true, y_pred, output_dict=True)
    cm      = confusion_matrix(y_true, y_pred)
    y_prob  = test_logits.softmax(1).numpy()
    auc_val = roc_auc_score(y_true, y_prob, multi_class='ovr')
    save_results_to_excel("TabTransformer", test_acc, cr_dict, cm, auc_val)

    # -------- Feature Importance --------
    model.eval()
    saliency = []
    for xb, yb in test_loader:
        xb = xb.to(DEVICE).requires_grad_(True)
        prob = model(xb)
        one_hot = torch.zeros_like(prob).scatter_(1, prob.argmax(1, keepdim=True), 1.0)
        prob.backward(gradient=one_hot)
        saliency.append(xb.grad.abs().cpu().numpy())
        model.zero_grad()
    saliency = np.concatenate(saliency).mean(0)
    saliency /= saliency.sum()
    idx_sort = saliency.argsort()[::-1]
    fn = preprocessor.get_feature_names_out()
    save_feature_importance("TabTransformer", fn[idx_sort], saliency[idx_sort])


if __name__ == "__main__":
    main()

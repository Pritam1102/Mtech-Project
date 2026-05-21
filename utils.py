import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
        precision_score,
        recall_score,
        roc_curve,
        f1_score,
        precision_recall_curve
    )

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, title=None):
    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss', marker = 'o')
    plt.plot(val_losses, label='Validation Loss', marker = 's')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Accuracy', marker = 'o')
    plt.plot(val_accuracies, label = "Validation Accuracy", marker = 's')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()



def getMetricsAtThreshold(model, test_loader, device, threshold=0.5):
    model.eval()
    model.to(device)
    all_probs = []
    all_labels = []
    all_indices =[]

    with torch.no_grad():
        for imgs, labels,indices in test_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            # print(outputs.shape)
            # print(labels.shape)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            probs = torch.sigmoid(outputs).flatten().cpu().numpy()
            labels = labels.flatten().cpu().numpy()
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_indices.extend(indices)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)
    preds = (all_probs >= threshold).astype(int)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    false_positive_indices = all_indices[(all_labels == 0) & (preds == 1)].tolist()
    false_negative_negative = all_indices[(all_labels == 1) & (preds == 0)].tolist()
    all_false_indices = all_indices[all_labels != preds].tolist()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = precision_score(all_labels, preds, zero_division=0)
    recall_val = recall_score(all_labels, preds, zero_division=0)
    f1_score_val = f1_score(all_labels, preds, zero_division=0)
    specificity = tn / (tn + fp)

    print(f"\nMetrics at Threshold = {threshold:.3f}")
    print(f"AUROC       : {auroc:.4f}")
    print(f"AUPRC       : {auprc:.4f}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision_val:.4f}")
    print(f"Recall      : {recall_val:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1 Score    : {f1_score_val:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array([[tn, fp],
                    [fn, tp]]))

    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    precision_curve, recall_curve, _ = precision_recall_curve(
        all_labels,
        all_probs
    )

    plt.figure(figsize=(12, 5))

    # ROC Curve
    plt.subplot(1, 2, 1)

    plt.plot(fpr, tpr,
             label=f'AUROC = {auroc:.4f}')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    # PR Curve
    plt.subplot(1, 2, 2)

    plt.plot(recall_curve,
             precision_curve,
             label=f'AUPRC = {auprc:.4f}')

    baseline = np.mean(all_labels)

    plt.hlines(baseline,0,
        1,
        colors='gray',
        linestyles='--',
        label=f'Baseline = {baseline:.3f}'
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    return {
        "threshold": threshold,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "precision": precision_val,
        "recall": recall_val,
        "specificity": specificity,
        "f1_score": f1_score_val,
        "false_positive_indices": false_positive_indices,
        "false_negative_indices": false_negative_negative,
        "all_false_indices": all_false_indices
    }

def apply_elliptical_mask(img):
    """
    img: PIL Image
    masks everything outside ellipse (background removal)
    """

    img_np = np.array(img)
    h, w = img_np.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.45))  # ellipse size

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    masked = cv2.bitwise_and(img_np, img_np, mask=mask)

    return Image.fromarray(masked)

def residual_gaussian_blur(img, p=0.5):
    if random.random() > p:
        return img

    img_np = np.array(img).astype(np.float32)

    blurred = cv2.GaussianBlur(img_np, (9, 9), 2.0)

    residual = img_np + blurred
    residual = np.clip(residual, 0, 255).astype(np.uint8)

    return Image.fromarray(residual)

def apply_clahe(img, p=0.5):
    if random.random() > p:
        return img

    img_np = np.array(img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(enhanced)

def getLatency(model, device, input_size=(1, 3, 448, 448), runs=50, warmup=20):
    model = model.to(device)
    # device = next(model.parameters()).device
    model.eval()
    torch.set_num_threads(4)
    # Dummy input (no data pipeline)
    dummy_input = torch.randn(input_size).to(device)

    # Disable autograd completely
    with torch.inference_mode():

        # Warm-up (CPU cache, thread pool, JIT)
        for _ in range(warmup):
            _ = model(dummy_input)

        latencies = []

        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append(end - start)

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"Mean Inference Latency: {mean_latency*1000:.2f} ms")
    print(f"Std Dev: {std_latency*1000:.2f} ms")

    # return round(mean_latency*1000, 2), round(std_latency*1000, 2)

def getScriptedModel(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 448, 448)

    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model = torch.jit.freeze(scripted_model)
    return scripted_model

def getModelSize(model):
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_in_mb = buffer.getbuffer().nbytes / (1024 * 1024)
    return round(size_in_mb, 2)

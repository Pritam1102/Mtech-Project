
from utils import *
import torchvision.transforms.functional as TF
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image
from torchvision.models import efficientnet_b7, efficientnet_b0, squeezenet1_1, mobilenet_v3_large, shufflenet_v2_x2_0, convnext_large, resnet34
import cv2

def getAUC(labels, outputs):
    from sklearn.metrics import roc_auc_score
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()
    try:
        auc = roc_auc_score(labels, outputs)
    except ValueError:
        auc = float('nan')
    return auc

def trainer(train_loader, val_loader, model, criterion, optimizer, scheduler, device, epochs, path, patience=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    aucs = []
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        model.train()
        running_loss, running_correct, total_samples = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc="Training", leave=False)
        for imgs, labels,_ in train_bar:
            imgs = imgs.to(device)
            labels = labels.float().to(device)  # shape (batch,)
            # print("Labels shape", labels.shape)
            labels = labels.flatten()  # shape -> (batch,)
            optimizer.zero_grad()

            outputs = model(imgs)              # shape (batch, 1) or (batch,)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            # outputs = outputs.squeeze()    # shape -> (batch,)
            outputs = outputs.flatten().clone()      # shape -> (batch,)

            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()

            probs = torch.sigmoid(outputs.detach())     # sigmoid(logit)
            preds = (probs > 0.5).float()

            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            train_bar.set_postfix(
                loss=f"{running_loss/total_samples:.4f}",
                acc=f"{(running_correct/total_samples)*100:.2f}%"
            )

        train_losses.append(running_loss / total_samples)
        train_accuracies.append(running_correct / total_samples)

        # Validation. During validation, gradients are not computed and dropout is disabled. 
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_probs, all_labels = [], []

        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for imgs, labels, _ in val_bar:
                imgs = imgs.to(device)
                labels = labels.float().to(device)
                labels = labels.flatten()  # shape -> (batch,)
                outputs = model(imgs)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                outputs = outputs.flatten().clone()

                loss = criterion(outputs, labels)
                probs = torch.sigmoid(outputs.detach())
                preds = (probs > 0.5).float()

                val_loss += loss.item() * labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

                val_bar.set_postfix(
                    loss=f"{val_loss/val_total:.4f}",
                    acc=f"{(val_correct/val_total)*100:.2f}%"
                )

        # Compute full epoch AUC
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        auc = getAUC(all_labels, all_probs)

        aucs.append(auc)
        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        scheduler.step(auc)
        # Save the model if it has the best AUC so far
        if auc >= max(aucs[:-1], default=0):
            torch.save(model.state_dict(), path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:  # Early stopping after "patience" epochs without improvement
                print("Early stopping triggered.")
                break
        # if val_losses[-1] < best_val_loss:
        #     best_val_loss = val_losses[-1]
        #     torch.save(model.state_dict(), path)
        print(f"Epoch {epoch:02d}: "
              f"Train Loss={train_losses[-1]:.4f}, Train Acc={train_accuracies[-1]*100:.2f}% | "
              f"Val Loss={val_losses[-1]:.4f}, Val Acc={val_accuracies[-1]*100:.2f}% | "
              f"AUC={auc:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

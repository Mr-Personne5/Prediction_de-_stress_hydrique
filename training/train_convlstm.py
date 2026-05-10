"""
training/train_convlstm.py
===========================
Entraînement du modèle principal ConvLSTM encoder-decoder.

Stratégie patches :
Le dataset de 81 cartes 90x90 est trop petit pour entraîner un ConvLSTM
directement. On découpe chaque carte en patches de 16x16 pixels avec un
stride de 8 pixels (chevauchement 50%). Cela multiplie le nombre
d'exemples par ~36 et permet une convergence stable.

Auteur : Djiba Kaba — Master IASD UKAG
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import logging
from pathlib import Path
from datetime import datetime

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.convlstm import (
    ConvLSTMEncoderDecoder, prepare_convlstm_sequences,
    compute_metrics_spatial, SEQ_LEN
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
SPLITS_DIR  = BASE_DIR / "data" / "processed" / "splits" / "haute_guinee"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "tables"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Hyperparamètres ---
HIDDEN_DIMS   = [16, 32]
KERNEL_SIZE   = 3
LEARNING_RATE = 1e-4
BATCH_SIZE    = 32
N_EPOCHS      = 100
PATIENCE      = 15
PATCH_SIZE    = 16
STRIDE_TRAIN  = 8    # Chevauchement 50% pour maximiser les exemples
STRIDE_TEST   = 16   # Pas de chevauchement pour le test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensors():
    log.info("Chargement des tenseurs...")
    X_train = torch.load(SPLITS_DIR / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(SPLITS_DIR / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(SPLITS_DIR / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(SPLITS_DIR / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(SPLITS_DIR / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(SPLITS_DIR / "y_test.pt",  weights_only=True).numpy()
    log.info("  X_train : %s | y_train : %s", X_train.shape, y_train.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TRAIN):
    """
    Découpe les cartes en patches pour augmenter le dataset.

    Au lieu d'entraîner sur 81 cartes 90x90, on découpe chaque carte
    en patches de patch_size x patch_size avec un pas de stride pixels.
    Cela multiplie le nombre d'exemples par environ (90/stride)^2.

    Args:
        X_seq      : Tensor (N, seq_len, H, W, 2)
        y_seq      : Tensor (N, H, W)
        patch_size : taille du patch en pixels
        stride     : pas entre les patches (stride < patch_size = chevauchement)

    Returns:
        X_patches : Tensor (N_patches, seq_len, patch_size, patch_size, 2)
        y_patches : Tensor (N_patches, patch_size, patch_size)
    """
    N, seq_len, H, W, F = X_seq.shape
    X_patches, y_patches = [], []

    for i in range(N):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                X_patches.append(
                    X_seq[i, :, h:h+patch_size, w:w+patch_size, :]
                )
                y_patches.append(
                    y_seq[i, h:h+patch_size, w:w+patch_size]
                )

    return torch.stack(X_patches), torch.stack(y_patches)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return total_loss / len(loader.dataset), preds, targets


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" PHASE 3 — ConvLSTM Encoder-Decoder (modèle principal)")
    log.info(" Device      : %s", DEVICE)
    log.info(" Hidden dims : %s | Kernel : %d", HIDDEN_DIMS, KERNEL_SIZE)
    log.info(" Patch size  : %d | Stride train : %d", PATCH_SIZE, STRIDE_TRAIN)
    log.info(" Batch size  : %d | Seq len : %d", BATCH_SIZE, SEQ_LEN)
    log.info(" Epochs      : %d | Patience : %d", N_EPOCHS, PATIENCE)
    log.info("=" * 55)

    # --- Charger ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_tensors()

    # --- Préparer les séquences ---
    log.info("Préparation des séquences (seq_len=%d)...", SEQ_LEN)
    X_tr_seq, y_tr_seq = prepare_convlstm_sequences(X_train, y_train)
    X_va_seq, y_va_seq = prepare_convlstm_sequences(X_val,   y_val)
    X_te_seq, y_te_seq = prepare_convlstm_sequences(X_test,  y_test)

    # --- Extraction des patches ---
    log.info("Extraction des patches (size=%d, stride=%d)...",
             PATCH_SIZE, STRIDE_TRAIN)
    X_tr_seq, y_tr_seq = extract_patches(
        X_tr_seq, y_tr_seq, PATCH_SIZE, STRIDE_TRAIN)
    X_va_seq, y_va_seq = extract_patches(
        X_va_seq, y_va_seq, PATCH_SIZE, STRIDE_TRAIN)
    X_te_seq, y_te_seq = extract_patches(
        X_te_seq, y_te_seq, PATCH_SIZE, STRIDE_TEST)

    log.info("  Train : %d patches | Val : %d | Test : %d",
             len(X_tr_seq), len(X_va_seq), len(X_te_seq))
    log.info("  Shape X_train patches : %s", tuple(X_tr_seq.shape))

    # --- DataLoaders ---
    train_ds = TensorDataset(X_tr_seq.float(), y_tr_seq.float())
    val_ds   = TensorDataset(X_va_seq.float(), y_va_seq.float())
    test_ds  = TensorDataset(X_te_seq.float(), y_te_seq.float())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # --- Modèle ---
    model = ConvLSTMEncoderDecoder(
        input_dim=2,
        hidden_dims=HIDDEN_DIMS,
        kernel_size=KERNEL_SIZE
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Paramètres du modèle : %d", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # --- Entraînement ---
    log.info("Entraînement...")
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    start = datetime.now()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d — train=%.4f | val=%.4f",
                     epoch, N_EPOCHS, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       CHECKPOINTS / "convlstm_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("  Early stopping epoch %d (meilleur : epoch %d)",
                         epoch, best_epoch)
                break

    duration = (datetime.now() - start).total_seconds()
    log.info("Terminé en %.1f sec (%.1f min)", duration, duration / 60)

    # --- Évaluation finale ---
    log.info("Chargement meilleur modèle (epoch %d)...", best_epoch)
    model.load_state_dict(
        torch.load(CHECKPOINTS / "convlstm_best.pt", weights_only=True)
    )

    _, tr_preds, tr_targets = eval_epoch(model, train_loader, criterion)
    _, va_preds, va_targets = eval_epoch(model, val_loader,   criterion)
    _, te_preds, te_targets = eval_epoch(model, test_loader,  criterion)

    m_train = compute_metrics_spatial(tr_targets, tr_preds)
    m_val   = compute_metrics_spatial(va_targets, va_preds)
    m_test  = compute_metrics_spatial(te_targets, te_preds)

    log.info("  [TRAIN] RMSE=%.4f | R²=%.4f | r=%.4f",
             m_train["rmse"], m_train["r2"], m_train["pearson_r"])
    log.info("  [VAL]   RMSE=%.4f | R²=%.4f | r=%.4f",
             m_val["rmse"],   m_val["r2"],   m_val["pearson_r"])
    log.info("  [TEST]  RMSE=%.4f | R²=%.4f | r=%.4f",
             m_test["rmse"],  m_test["r2"],  m_test["pearson_r"])

    # Sauvegarder les prédictions test
    np.save(CHECKPOINTS / "convlstm_test_preds.npy",   te_preds)
    np.save(CHECKPOINTS / "convlstm_test_targets.npy", te_targets)

    # --- Sauvegarder les résultats ---
    results = {
        "model": "ConvLSTM encoder-decoder",
        "zone": "haute_guinee",
        "hyperparams": {
            "hidden_dims": HIDDEN_DIMS,
            "kernel_size": KERNEL_SIZE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "patch_size": PATCH_SIZE,
            "stride_train": STRIDE_TRAIN,
            "best_epoch": best_epoch,
        },
        "training_time_sec": round(duration, 1),
        "metrics": {
            "train": m_train,
            "val":   m_val,
            "test":  m_test
        },
        "baselines": {
            "rf":   {"rmse": 0.6190, "r2": 0.2494},
            "lstm": {"rmse": 0.9878, "r2": 0.3186}
        }
    }

    with open(RESULTS_DIR / "convlstm_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Résumé ---
    log.info("=" * 55)
    log.info(" RÉSUMÉ COMPARATIF — Phase 3")
    log.info("=" * 55)
    log.info("  Modèle          RMSE test   R² test")
    log.info("  Random Forest   0.6190      0.2494  (baseline)")
    log.info("  LSTM pixel      0.9878      0.3186  (baseline)")
    log.info("  ConvLSTM        %.4f      %.4f  <- modèle principal",
             m_test["rmse"], m_test["r2"])
    if m_test["r2"] > 0.3186:
        log.info("  ConvLSTM surpasse le LSTM sur R2 ✓")
    if m_test["rmse"] < 0.6190:
        log.info("  ConvLSTM surpasse le RF sur RMSE ✓")
    log.info("=" * 55)
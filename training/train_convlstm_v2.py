"""
training/train_convlstm_v2.py
==============================
ConvLSTM v2 — Entraînement sur données combinées HG + MG.

Stratégie :
Les grilles HG (90x90) et MG (46x68) ont des dimensions différentes.
On extrait des patches de 16x16 depuis les deux zones et on les
concatène. Les patches deviennent des exemples indépendants.

Train : patches HG train + patches MG train
Val   : patches HG val uniquement (évaluation sur zone principale)
Test  : patches HG test uniquement (évaluation sur zone principale)

Cela double environ le nombre d'exemples d'entraînement.

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
SPLITS_HG   = BASE_DIR / "data" / "processed" / "splits" / "haute_guinee"
SPLITS_MG   = BASE_DIR / "data" / "processed" / "splits" / "moyenne_guinee"
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
STRIDE_TRAIN  = 8
STRIDE_TEST   = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_zone(splits_dir):
    """Charge les tenseurs d'une zone."""
    X_train = torch.load(splits_dir / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(splits_dir / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(splits_dir / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(splits_dir / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(splits_dir / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(splits_dir / "y_test.pt",  weights_only=True).numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TRAIN):
    """Découpe les cartes en patches."""
    N, seq_len, H, W, F = X_seq.shape
    X_patches, y_patches = [], []
    for i in range(N):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                X_patches.append(X_seq[i, :, h:h+patch_size, w:w+patch_size, :])
                y_patches.append(y_seq[i, h:h+patch_size, w:w+patch_size])
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

    log.info("=" * 60)
    log.info(" PHASE 3 — ConvLSTM v2 (données combinées HG + MG)")
    log.info(" Device      : %s", DEVICE)
    log.info(" Hidden dims : %s | Patch size : %d", HIDDEN_DIMS, PATCH_SIZE)
    log.info(" Batch size  : %d | Seq len : %d", BATCH_SIZE, SEQ_LEN)
    log.info("=" * 60)

    # --- Charger les deux zones ---
    log.info("Chargement Haute Guinée...")
    X_tr_hg, X_va_hg, X_te_hg, y_tr_hg, y_va_hg, y_te_hg = load_zone(SPLITS_HG)
    log.info("  HG train : %s", X_tr_hg.shape)

    log.info("Chargement Moyenne Guinée...")
    X_tr_mg, X_va_mg, X_te_mg, y_tr_mg, y_va_mg, y_te_mg = load_zone(SPLITS_MG)
    log.info("  MG train : %s", X_tr_mg.shape)

    # --- Préparer les séquences ---
    log.info("Préparation des séquences (seq_len=%d)...", SEQ_LEN)
    X_tr_hg_seq, y_tr_hg_seq = prepare_convlstm_sequences(X_tr_hg, y_tr_hg)
    X_tr_mg_seq, y_tr_mg_seq = prepare_convlstm_sequences(X_tr_mg, y_tr_mg)
    X_va_hg_seq, y_va_hg_seq = prepare_convlstm_sequences(X_va_hg, y_va_hg)
    X_te_hg_seq, y_te_hg_seq = prepare_convlstm_sequences(X_te_hg, y_te_hg)

    # --- Extraction des patches ---
    log.info("Extraction des patches...")
    X_tr_hg_p, y_tr_hg_p = extract_patches(X_tr_hg_seq, y_tr_hg_seq,
                                            PATCH_SIZE, STRIDE_TRAIN)
    X_tr_mg_p, y_tr_mg_p = extract_patches(X_tr_mg_seq, y_tr_mg_seq,
                                            PATCH_SIZE, STRIDE_TRAIN)
    X_va_p, y_va_p = extract_patches(X_va_hg_seq, y_va_hg_seq,
                                     PATCH_SIZE, STRIDE_TRAIN)
    X_te_p, y_te_p = extract_patches(X_te_hg_seq, y_te_hg_seq,
                                     PATCH_SIZE, STRIDE_TEST)

    # Combiner les patches HG + MG pour l'entraînement
    X_train_comb = torch.cat([X_tr_hg_p, X_tr_mg_p], dim=0)
    y_train_comb = torch.cat([y_tr_hg_p, y_tr_mg_p], dim=0)

    log.info("  Patches HG train   : %d", len(X_tr_hg_p))
    log.info("  Patches MG train   : %d", len(X_tr_mg_p))
    log.info("  Patches combinés   : %d (x%.1f vs v1)",
             len(X_train_comb), len(X_train_comb) / len(X_tr_hg_p))
    log.info("  Patches val (HG)   : %d", len(X_va_p))
    log.info("  Patches test (HG)  : %d", len(X_te_p))

    # --- DataLoaders ---
    train_ds = TensorDataset(X_train_comb.float(), y_train_comb.float())
    val_ds   = TensorDataset(X_va_p.float(), y_va_p.float())
    test_ds  = TensorDataset(X_te_p.float(), y_te_p.float())

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
                       CHECKPOINTS / "convlstm_v2_best.pt")
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
        torch.load(CHECKPOINTS / "convlstm_v2_best.pt", weights_only=True)
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
    np.save(CHECKPOINTS / "convlstm_v2_test_preds.npy",   te_preds)
    np.save(CHECKPOINTS / "convlstm_v2_test_targets.npy", te_targets)

    # --- Sauvegarder résultats ---
    results = {
        "model": "ConvLSTM v2 — données combinées HG+MG",
        "zone_train": "haute_guinee + moyenne_guinee",
        "zone_eval": "haute_guinee",
        "hyperparams": {
            "hidden_dims": HIDDEN_DIMS,
            "kernel_size": KERNEL_SIZE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "patch_size": PATCH_SIZE,
            "best_epoch": best_epoch,
            "n_patches_train": len(X_train_comb),
        },
        "training_time_sec": round(duration, 1),
        "metrics": {
            "train": m_train,
            "val":   m_val,
            "test":  m_test
        },
        "comparison": {
            "rf_v1":       {"rmse": 0.6190, "r2": 0.2494},
            "lstm_v1":     {"rmse": 0.9878, "r2": 0.3186},
            "convlstm_v1": {"rmse": 1.0194, "r2": 0.2255},
        }
    }

    with open(RESULTS_DIR / "convlstm_v2_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Résumé final ---
    log.info("=" * 60)
    log.info(" RÉSUMÉ COMPARATIF FINAL")
    log.info("=" * 60)
    log.info("  Modèle               RMSE test   R² test")
    log.info("  Random Forest        0.6190      0.2494")
    log.info("  LSTM pixel           0.9878      0.3186")
    log.info("  ConvLSTM v1 (HG)     1.0194      0.2255")
    log.info("  ConvLSTM v2 (HG+MG)  %.4f      %.4f",
             m_test["rmse"], m_test["r2"])
    if m_test["r2"] > 0.2255:
        log.info("  v2 ameliore v1 sur R2 ✓ (+%.4f)",
                 m_test["r2"] - 0.2255)
    if m_test["rmse"] < 1.0194:
        log.info("  v2 ameliore v1 sur RMSE ✓ (-%.4f)",
                 1.0194 - m_test["rmse"])
    log.info("=" * 60)
"""
training/train_lstm.py
=======================
Entraînement du baseline LSTM pixel-by-pixel.

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

# Reproductibilité
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline_lstm import (
    LSTMBaseline, prepare_lstm_sequences,
    compute_metrics_torch, SEQ_LEN
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Chemins ---
BASE_DIR    = Path(__file__).parent.parent
SPLITS_DIR  = BASE_DIR / "data" / "processed" / "splits" / "haute_guinee"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "tables"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Hyperparamètres ---
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE   = 2048
N_EPOCHS     = 50
PATIENCE     = 10   # Early stopping

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensors():
    log.info("Chargement des tenseurs...")
    X_train = torch.load(SPLITS_DIR / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(SPLITS_DIR / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(SPLITS_DIR / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(SPLITS_DIR / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(SPLITS_DIR / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(SPLITS_DIR / "y_test.pt",  weights_only=True).numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_epoch(model, loader, optimizer, criterion):
    """Une époque d'entraînement."""
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
    """Une époque d'évaluation."""
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
    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return total_loss / len(loader.dataset), preds, targets


if __name__ == "__main__":

    log.info("=" * 50)
    log.info(" PHASE 3 — Baseline LSTM pixel-by-pixel")
    log.info(" Device : %s", DEVICE)
    log.info(" Seq_len : %d mois", SEQ_LEN)
    log.info(" Epochs : %d | Patience : %d", N_EPOCHS, PATIENCE)
    log.info("=" * 50)

    # --- Charger ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_tensors()

    # --- Préparer les séquences ---
    log.info("Préparation des séquences LSTM (seq_len=%d)...", SEQ_LEN)
    X_tr_seq, y_tr_seq = prepare_lstm_sequences(X_train, y_train)
    X_va_seq, y_va_seq = prepare_lstm_sequences(X_val,   y_val)
    X_te_seq, y_te_seq = prepare_lstm_sequences(X_test,  y_test)

    log.info("  Train : %s séquences", X_tr_seq.shape[0])
    log.info("  Val   : %s séquences", X_va_seq.shape[0])
    log.info("  Test  : %s séquences", X_te_seq.shape[0])

    # --- DataLoaders ---
    train_ds = TensorDataset(X_tr_seq.float(), y_tr_seq.float())
    val_ds   = TensorDataset(X_va_seq.float(), y_va_seq.float())
    test_ds  = TensorDataset(X_te_seq.float(), y_te_seq.float())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # --- Modèle ---
    model = LSTMBaseline(
        input_size=2,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Paramètres du modèle : %d", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # --- Entraînement avec early stopping ---
    log.info("Entraînement...")
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_counter = 0
    history = []
    start = datetime.now()

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 5 == 0 or epoch == 1:
            log.info("  Epoch %2d/%d — train_loss=%.4f | val_loss=%.4f",
                     epoch, N_EPOCHS, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINTS / "lstm_baseline_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info("  Early stopping à l'epoch %d (meilleure val : epoch %d)",
                         epoch, best_epoch)
                break

    duration = (datetime.now() - start).total_seconds()
    log.info("Entraînement terminé en %.1f secondes", duration)

    # --- Évaluation finale avec le meilleur modèle ---
    log.info("Chargement du meilleur modèle (epoch %d)...", best_epoch)
    model.load_state_dict(
        torch.load(CHECKPOINTS / "lstm_baseline_best.pt", weights_only=True)
    )

    _, train_preds, train_targets = eval_epoch(model, train_loader, criterion)
    _, val_preds,   val_targets   = eval_epoch(model, val_loader,   criterion)
    _, test_preds,  test_targets  = eval_epoch(model, test_loader,  criterion)

    metrics_train = compute_metrics_torch(train_targets, train_preds)
    metrics_val   = compute_metrics_torch(val_targets,   val_preds)
    metrics_test  = compute_metrics_torch(test_targets,  test_preds)

    log.info("  [TRAIN] RMSE=%.4f | MAE=%.4f | R²=%.4f | r=%.4f",
             metrics_train["rmse"], metrics_train["mae"],
             metrics_train["r2"], metrics_train["pearson_r"])
    log.info("  [VAL]   RMSE=%.4f | MAE=%.4f | R²=%.4f | r=%.4f",
             metrics_val["rmse"], metrics_val["mae"],
             metrics_val["r2"], metrics_val["pearson_r"])
    log.info("  [TEST]  RMSE=%.4f | MAE=%.4f | R²=%.4f | r=%.4f",
             metrics_test["rmse"], metrics_test["mae"],
             metrics_test["r2"], metrics_test["pearson_r"])

    # --- Sauvegarder les résultats ---
    results = {
        "model": "LSTM pixel-by-pixel",
        "zone": "haute_guinee",
        "hyperparams": {
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "best_epoch": best_epoch,
        },
        "training_time_sec": round(duration, 1),
        "metrics": {
            "train": metrics_train,
            "val":   metrics_val,
            "test":  metrics_test
        }
    }

    with open(RESULTS_DIR / "lstm_baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Résumé ---
    log.info("=" * 50)
    log.info(" RÉSUMÉ — LSTM Baseline")
    log.info("=" * 50)
    log.info("  TRAIN : RMSE=%.4f | R²=%.4f", metrics_train["rmse"], metrics_train["r2"])
    log.info("  VAL   : RMSE=%.4f | R²=%.4f", metrics_val["rmse"],   metrics_val["r2"])
    log.info("  TEST  : RMSE=%.4f | R²=%.4f", metrics_test["rmse"],  metrics_test["r2"])
    log.info("  Comparaison RF   -> RMSE=0.6190 | R²=0.2494")
    log.info("  Comparaison LSTM -> RMSE=%.4f | R²=%.4f",
             metrics_test["rmse"], metrics_test["r2"])
    log.info("=" * 50)
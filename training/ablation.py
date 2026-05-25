"""
training/ablation.py
=====================
Ablation study — contribution de chaque source de données.

3 configurations testées :
- Config A : NDVI seul     (feature index 0)
- Config B : Précip seul   (feature index 1)
- Config C : Fusion NDVI + Précip (les deux — configuration normale)

Objectif :
Quantifier la contribution marginale de chaque source de données.
Si la fusion (C) surpasse les deux mono-source (A et B) d'au moins 5%
en R², le choix bi-source est scientifiquement justifié.

Méthode :
On réutilise exactement la même architecture ConvLSTM v1 et les mêmes
hyperparamètres. Seules les features d'entrée changent.

Auteur : Djiba Kaba — Chercheur indépendant
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

# Hyperparamètres identiques à ConvLSTM v1
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

# Définition des configurations
CONFIGS = {
    "A_ndvi_seul":    {"indices": [0],    "input_dim": 1, "label": "NDVI seul"},
    "B_precip_seul":  {"indices": [1],    "input_dim": 1, "label": "Précip seul"},
    "C_fusion":       {"indices": [0, 1], "input_dim": 2, "label": "Fusion NDVI+Précip"},
}


def load_tensors():
    X_train = torch.load(SPLITS_DIR / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(SPLITS_DIR / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(SPLITS_DIR / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(SPLITS_DIR / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(SPLITS_DIR / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(SPLITS_DIR / "y_test.pt",  weights_only=True).numpy()
    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TRAIN):
    N, seq_len, H, W, F = X_seq.shape
    X_patches, y_patches = [], []
    for i in range(N):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                X_patches.append(X_seq[i, :, h:h+patch_size, w:w+patch_size, :])
                y_patches.append(y_seq[i, h:h+patch_size, w:w+patch_size])
    return torch.stack(X_patches), torch.stack(y_patches)


def train_eval_config(config_name, config, X_tr, X_va, X_te, y_tr, y_va, y_te):
    """
    Entraîne et évalue le ConvLSTM pour une configuration de features.

    Args:
        config_name : str — identifiant de la config
        config      : dict — indices et input_dim
        X_tr/va/te  : tenseurs patches (N, seq_len, patch, patch, 2)
        y_tr/va/te  : tenseurs cibles

    Returns:
        dict résultats
    """
    log.info("--- Config %s : %s ---", config_name, config["label"])

    indices   = config["indices"]
    input_dim = config["input_dim"]

    # Sélectionner uniquement les features de cette config
    # X shape : (N, seq_len, patch, patch, 2) -> sélection sur dim -1
    X_tr_cfg = X_tr[:, :, :, :, indices] if len(indices) > 1 \
               else X_tr[:, :, :, :, indices[0]:indices[0]+1]
    X_va_cfg = X_va[:, :, :, :, indices] if len(indices) > 1 \
               else X_va[:, :, :, :, indices[0]:indices[0]+1]
    X_te_cfg = X_te[:, :, :, :, indices] if len(indices) > 1 \
               else X_te[:, :, :, :, indices[0]:indices[0]+1]

    log.info("  Shape X train : %s", tuple(X_tr_cfg.shape))

    # DataLoaders
    train_ds = TensorDataset(X_tr_cfg.float(), y_tr.float())
    val_ds   = TensorDataset(X_va_cfg.float(), y_va.float())
    test_ds  = TensorDataset(X_te_cfg.float(), y_te.float())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Modèle avec input_dim adapté
    model = ConvLSTMEncoderDecoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        kernel_size=KERNEL_SIZE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # Entraînement
    best_val_loss    = float("inf")
    best_epoch       = 0
    patience_counter = 0
    ckpt_path = CHECKPOINTS / f"ablation_{config_name}.pt"
    start = datetime.now()

    for epoch in range(1, N_EPOCHS + 1):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                val_loss += criterion(pred, y_b).item() * len(y_b)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    duration = (datetime.now() - start).total_seconds()
    log.info("  Terminé epoch %d (meilleur : %d) en %.0f sec",
             epoch, best_epoch, duration)

    # Évaluation finale
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            pred = model(X_b.to(DEVICE))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_b.numpy())

    te_preds   = np.concatenate(all_preds)
    te_targets = np.concatenate(all_targets)
    metrics = compute_metrics_spatial(te_targets, te_preds)

    log.info("  TEST : RMSE=%.4f | R²=%.4f | r=%.4f",
             metrics["rmse"], metrics["r2"], metrics["pearson_r"])

    return {
        "config": config_name,
        "label": config["label"],
        "best_epoch": best_epoch,
        "training_time_sec": round(duration, 1),
        "metrics_test": metrics
    }


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" PHASE 3 — Ablation Study")
    log.info(" 3 configurations : NDVI / Précip / Fusion")
    log.info(" Architecture : ConvLSTM v1 (même hyperparamètres)")
    log.info("=" * 55)

    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test = load_tensors()

    # Préparer les séquences et patches
    X_tr_seq, y_tr_seq = prepare_convlstm_sequences(X_train, y_train)
    X_va_seq, y_va_seq = prepare_convlstm_sequences(X_val,   y_val)
    X_te_seq, y_te_seq = prepare_convlstm_sequences(X_test,  y_test)

    X_tr_p, y_tr_p = extract_patches(X_tr_seq, y_tr_seq, PATCH_SIZE, STRIDE_TRAIN)
    X_va_p, y_va_p = extract_patches(X_va_seq, y_va_seq, PATCH_SIZE, STRIDE_TRAIN)
    X_te_p, y_te_p = extract_patches(X_te_seq, y_te_seq, PATCH_SIZE, STRIDE_TEST)

    log.info("Patches : train=%d | val=%d | test=%d",
             len(X_tr_p), len(X_va_p), len(X_te_p))

    # Entraîner les 3 configurations
    all_results = []
    for config_name, config in CONFIGS.items():
        result = train_eval_config(
            config_name, config,
            X_tr_p, X_va_p, X_te_p,
            y_tr_p, y_va_p, y_te_p
        )
        all_results.append(result)

    # Sauvegarder les résultats
    ablation_summary = {
        "study": "ablation_features",
        "model": "ConvLSTM encoder-decoder",
        "zone": "haute_guinee",
        "configurations": all_results
    }
    with open(RESULTS_DIR / "ablation_results.json", "w",
              encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2, ensure_ascii=False)

    # Résumé final
    log.info("=" * 55)
    log.info(" RÉSUMÉ ABLATION STUDY")
    log.info("=" * 55)
    log.info("  %-25s  RMSE     R²", "Configuration")
    log.info("  " + "-" * 45)
    for r in all_results:
        m = r["metrics_test"]
        log.info("  %-25s  %.4f   %.4f",
                 r["label"], m["rmse"], m["r2"])

    # Calculer le gain de la fusion
    r2_ndvi   = next(r["metrics_test"]["r2"] for r in all_results
                     if r["config"] == "A_ndvi_seul")
    r2_precip = next(r["metrics_test"]["r2"] for r in all_results
                     if r["config"] == "B_precip_seul")
    r2_fusion = next(r["metrics_test"]["r2"] for r in all_results
                     if r["config"] == "C_fusion")

    best_mono = max(r2_ndvi, r2_precip)
    gain_pct  = (r2_fusion - best_mono) / abs(best_mono) * 100 \
                if best_mono != 0 else 0

    log.info("  " + "-" * 45)
    log.info("  Gain fusion vs meilleur mono-source : %.1f%%", gain_pct)
    if gain_pct >= 5.0:
        log.info("  Seuil 5%% atteint — fusion justifiee ✓")
    else:
        log.info("  Seuil 5%% non atteint — gain=%.1f%% (a documenter)", gain_pct)
    log.info("=" * 55)
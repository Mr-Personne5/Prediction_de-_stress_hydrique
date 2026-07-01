"""
training/train_transfer.py
============================
SPRINT 2 — Transfer Learning ConvLSTM HG -> MG.

Stratégie en 2 étapes :

Étape 1 — Pré-entraînement (pre-training) :
  Entraîner le ConvLSTM sur HG + MG combinés.
  Le modèle apprend une représentation générale de la dynamique
  climatique ouest-africaine (pas spécifique à une zone).
  Objectif : encoder des patterns transférables.

Étape 2 — Fine-tuning sur HG :
  Charger les poids du pré-entraînement.
  Geler l'encoder (couches basses = patterns généraux) et n'entraîner
  que le decoder (couches hautes = spécialisation HG).
  Puis dégeler progressivement et fine-tuner tout le modèle avec lr très bas.

Évaluation :
  - Sur HG test (comparaison directe avec v1)
  - Sur MG test (mesure de la généralisation améliorée)

Testé sur les deux configurations de features :
  - Config 1 : 2 features (NDVI + Précip)
  - Config 2 : 4 features (NDVI + Précip + LST + VHI_v2)

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 2 - 01/07/26
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
SPLITS_V2   = BASE_DIR / "data" / "processed" / "splits_v2"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "tables"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_DIMS  = [16, 32]
KERNEL_SIZE  = 3
PATCH_SIZE   = 16
STRIDE_TRAIN = 8
STRIDE_TEST  = 16
BATCH_SIZE   = 32
PATIENCE     = 15

# Learning rates
LR_PRETRAIN  = 1e-4   # Pré-entraînement HG+MG
LR_FINETUNE1 = 5e-5   # Fine-tuning decoder seul
LR_FINETUNE2 = 1e-5   # Fine-tuning complet (très bas)

EPOCHS_PRETRAIN  = 80
EPOCHS_FINETUNE1 = 40  # Decoder seul
EPOCHS_FINETUNE2 = 40  # Tout le modèle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensors(config_name, zone):
    d = SPLITS_V2 / config_name / zone
    X_train = torch.load(d / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(d / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(d / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(d / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(d / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(d / "y_test.pt",  weights_only=True).numpy()
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


def make_loader(X_seq, y_seq, patch_size, stride, shuffle=True):
    X_p, y_p = extract_patches(X_seq, y_seq, patch_size, stride)
    ds = TensorDataset(X_p.float(), y_p.float())
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        pred = model(X_b)
        loss = criterion(pred, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y_b)
    return total / len(loader.dataset)


def eval_model(model, loader, criterion):
    model.eval()
    total, preds, targets = 0.0, [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            pred = model(X_b)
            total += criterion(pred, y_b).item() * len(y_b)
            preds.append(pred.cpu().numpy())
            targets.append(y_b.cpu().numpy())
    return (total / len(loader.dataset),
            np.concatenate(preds), np.concatenate(targets))


def run_training(model, train_loader, val_loader, optimizer, criterion,
                 n_epochs, patience, ckpt_path, phase_name):
    best_val, best_ep, pat_cnt = float("inf"), 0, 0
    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = eval_model(model, val_loader, criterion)
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        ).step(val_loss)
        if epoch % 10 == 0 or epoch == 1:
            log.info("  [%s] Epoch %3d — train=%.4f | val=%.4f",
                     phase_name, epoch, tr_loss, val_loss)
        if val_loss < best_val:
            best_val, best_ep, pat_cnt = val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                log.info("  [%s] Early stopping epoch %d (best: %d)",
                         phase_name, epoch, best_ep)
                break
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return best_ep


def run_config(config_name, n_features):
    log.info("=" * 60)
    log.info(" SPRINT 2 — Transfer Learning | %s (%d features)",
             config_name.upper(), n_features)
    log.info("=" * 60)

    # ---------------------------------------------------------------
    # Charger les données
    # ---------------------------------------------------------------
    log.info("Chargement données HG et MG...")
    X_tr_hg, X_va_hg, X_te_hg, y_tr_hg, y_va_hg, y_te_hg = load_tensors(config_name, "haute_guinee")
    X_tr_mg, X_va_mg, X_te_mg, y_tr_mg, y_va_mg, y_te_mg = load_tensors(config_name, "moyenne_guinee")

    log.info("  HG train : %s | MG train : %s", X_tr_hg.shape, X_tr_mg.shape)

    # Préparer les séquences
    X_tr_hg_seq, y_tr_hg_seq = prepare_convlstm_sequences(X_tr_hg, y_tr_hg)
    X_va_hg_seq, y_va_hg_seq = prepare_convlstm_sequences(X_va_hg, y_va_hg)
    X_te_hg_seq, y_te_hg_seq = prepare_convlstm_sequences(X_te_hg, y_te_hg)
    X_tr_mg_seq, y_tr_mg_seq = prepare_convlstm_sequences(X_tr_mg, y_tr_mg)
    X_te_mg_seq, y_te_mg_seq = prepare_convlstm_sequences(X_te_mg, y_te_mg)

    # DataLoaders
    # Pré-entraînement : HG + MG combinés
    X_tr_comb = torch.cat([
        extract_patches(X_tr_hg_seq, y_tr_hg_seq)[0],
        extract_patches(X_tr_mg_seq, y_tr_mg_seq)[0]
    ])
    y_tr_comb = torch.cat([
        extract_patches(X_tr_hg_seq, y_tr_hg_seq)[1],
        extract_patches(X_tr_mg_seq, y_tr_mg_seq)[1]
    ])
    pretrain_loader = DataLoader(
        TensorDataset(X_tr_comb.float(), y_tr_comb.float()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = make_loader(X_va_hg_seq, y_va_hg_seq, PATCH_SIZE, STRIDE_TRAIN)
    te_hg_loader = make_loader(X_te_hg_seq, y_te_hg_seq, PATCH_SIZE, STRIDE_TEST, shuffle=False)
    te_mg_loader = make_loader(X_te_mg_seq, y_te_mg_seq, PATCH_SIZE, STRIDE_TEST, shuffle=False)

    log.info("  Patches pré-entraînement : %d (HG+MG)", len(X_tr_comb))

    criterion = nn.MSELoss()
    ckpt_pre  = CHECKPOINTS / f"transfer_{config_name}_pretrain.pt"
    ckpt_ft1  = CHECKPOINTS / f"transfer_{config_name}_finetune1.pt"
    ckpt_ft2  = CHECKPOINTS / f"transfer_{config_name}_finetune2.pt"

    # ---------------------------------------------------------------
    # ÉTAPE 1 : Pré-entraînement HG + MG
    # ---------------------------------------------------------------
    log.info("--- ÉTAPE 1 : Pré-entraînement HG+MG (lr=%.0e) ---", LR_PRETRAIN)
    model = ConvLSTMEncoderDecoder(
        input_dim=n_features, hidden_dims=HIDDEN_DIMS, kernel_size=KERNEL_SIZE
    ).to(DEVICE)
    log.info("  Paramètres : %d", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-5)
    start = datetime.now()
    run_training(model, pretrain_loader, val_loader, opt, criterion,
                 EPOCHS_PRETRAIN, PATIENCE, ckpt_pre, "PRE-TRAIN")
    log.info("  Pré-entraînement terminé en %.1fs", (datetime.now()-start).total_seconds())

    # ---------------------------------------------------------------
    # ÉTAPE 2a : Fine-tuning — decoder seul (encoder gelé)
    # ---------------------------------------------------------------
    log.info("--- ÉTAPE 2a : Fine-tuning decoder seul (encoder gelé, lr=%.0e) ---", LR_FINETUNE1)

    # Geler l'encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    # Seul le decoder est entraînable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info("  Paramètres entraînables : %d / %d",
             sum(p.numel() for p in trainable_params),
             sum(p.numel() for p in model.parameters()))

    # DataLoader HG seul pour fine-tuning
    ft_loader = make_loader(X_tr_hg_seq, y_tr_hg_seq, PATCH_SIZE, STRIDE_TRAIN)
    opt_ft1 = torch.optim.Adam(trainable_params, lr=LR_FINETUNE1, weight_decay=1e-5)

    run_training(model, ft_loader, val_loader, opt_ft1, criterion,
                 EPOCHS_FINETUNE1, PATIENCE, ckpt_ft1, "FINE-TUNE-1")

    # ---------------------------------------------------------------
    # ÉTAPE 2b : Fine-tuning complet (tout dégelé, lr très bas)
    # ---------------------------------------------------------------
    log.info("--- ÉTAPE 2b : Fine-tuning complet (lr=%.0e) ---", LR_FINETUNE2)

    # Dégeler tout
    for param in model.parameters():
        param.requires_grad = True

    opt_ft2 = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE2, weight_decay=1e-5)
    run_training(model, ft_loader, val_loader, opt_ft2, criterion,
                 EPOCHS_FINETUNE2, PATIENCE, ckpt_ft2, "FINE-TUNE-2")

    # ---------------------------------------------------------------
    # Évaluation finale
    # ---------------------------------------------------------------
    log.info("--- Évaluation finale ---")
    model.load_state_dict(torch.load(ckpt_ft2, weights_only=True))

    _, te_hg_preds, te_hg_targets = eval_model(model, te_hg_loader, criterion)
    _, te_mg_preds, te_mg_targets = eval_model(model, te_mg_loader, criterion)

    m_hg = compute_metrics_spatial(te_hg_targets, te_hg_preds)
    m_mg = compute_metrics_spatial(te_mg_targets, te_mg_preds)

    log.info("  HG test : RMSE=%.4f | R²=%.4f | r=%.4f",
             m_hg["rmse"], m_hg["r2"], m_hg["pearson_r"])
    log.info("  MG test : RMSE=%.4f | R²=%.4f | r=%.4f",
             m_mg["rmse"], m_mg["r2"], m_mg["pearson_r"])

    # Dégradation
    deg_r2 = (m_hg["r2"] - m_mg["r2"]) / abs(m_hg["r2"]) * 100 if m_hg["r2"] != 0 else 0
    log.info("  Dégradation R² HG->MG : %+.1f%% (v1 était +172.7%%)", deg_r2)

    # Sauvegarder
    results = {
        "config": config_name,
        "n_features": n_features,
        "strategy": "pretrain HG+MG -> finetune decoder -> finetune full",
        "metrics_hg_test": m_hg,
        "metrics_mg_test": m_mg,
        "degradation_r2_pct": round(deg_r2, 1),
        "v1_degradation_r2_pct": 172.7,
        "improvement": round(172.7 - deg_r2, 1) if deg_r2 < 172.7 else 0,
    }
    path = RESULTS_DIR / f"transfer_{config_name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("  Résultats sauvegardés : %s", path)
    return m_hg, m_mg


if __name__ == "__main__":

    all_results = {}

    # Config 1 — 2 features
    m_hg_c1, m_mg_c1 = run_config("config1", n_features=2)
    all_results["config1"] = {"hg": m_hg_c1, "mg": m_mg_c1}

    # Config 2 — 4 features
    m_hg_c2, m_mg_c2 = run_config("config2", n_features=4)
    all_results["config2"] = {"hg": m_hg_c2, "mg": m_mg_c2}

    # Résumé comparatif final
    log.info("=" * 60)
    log.info(" RÉSUMÉ SPRINT 2 — Transfer Learning")
    log.info("=" * 60)
    log.info("  %-12s  %-10s  %-10s  %-10s  %-10s  Dégradation R²",
             "Config", "HG RMSE", "HG R²", "MG RMSE", "MG R²")
    log.info("  " + "-" * 65)
    for cfg, res in all_results.items():
        hg, mg = res["hg"], res["mg"]
        deg = (hg["r2"] - mg["r2"]) / abs(hg["r2"]) * 100 if hg["r2"] != 0 else 0
        log.info("  %-12s  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %+.1f%%",
                 cfg, hg["rmse"], hg["r2"], mg["rmse"], mg["r2"], deg)
    log.info("  Référence v1 (RF, sans transfer) : dégradation R² = +172.7%%")
    log.info("=" * 60)
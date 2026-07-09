"""
training/train_convlstm_v4.py
================================
Sprint 4 — Transfer Learning ConvLSTM avec lead time +1 mois.

Différence clé avec train_convlstm_v3.py :
  Les tenseurs splits_v4/ sont pré-séquencés (N, SEQ_LEN, H, W, F).
  prepare_convlstm_sequences n'est PAS appelé — les séquences sont déjà construites.
  Lead time : X[t-SEQ_LEN:t] → y[t+1]

Stratégie identique au Sprint 3 (transfer learning) :
  Étape 1 : Pré-entraînement HG + MG combinés
  Étape 2a : Fine-tuning decoder seul (encoder gelé)
  Étape 2b : Fine-tuning complet (tout dégelé, lr très bas)

Hyperparamètres identiques aux Sprints 2 et 3 :
  HIDDEN_DIMS      = [16, 32]
  KERNEL_SIZE      = 3
  PATCH_SIZE       = 16, STRIDE_TRAIN = 8, STRIDE_TEST = 16
  BATCH_SIZE       = 32
  LR_PRETRAIN      = 1e-4
  LR_FINETUNE1     = 5e-5
  LR_FINETUNE2     = 1e-5
  EPOCHS_PRETRAIN  = 80
  EPOCHS_FINETUNE1 = 40
  EPOCHS_FINETUNE2 = 40
  PATIENCE         = 15

Source : splits_v4/config1/{zone}/ et splits_v4/config2/{zone}/
Config 1 : input_dim = 2 (NDVI + Précip)
Config 2 : input_dim = 4 (NDVI + Précip + LST + VHI_v3)

Checkpoints : results/checkpoints/v4_transfer_config{1,2}_*.pt
Résultats   : results/tables/v4_transfer_results.json

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 4 — Juillet 2026
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
    ConvLSTMEncoderDecoder, compute_metrics_spatial, SEQ_LEN
)
# Note : prepare_convlstm_sequences non importé — tenseurs v4 déjà pré-séquencés

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
SPLITS_V4   = BASE_DIR / "data" / "processed" / "splits_v4"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "tables"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparamètres identiques aux Sprints 2 et 3
HIDDEN_DIMS      = [16, 32]
KERNEL_SIZE      = 3
PATCH_SIZE       = 16
STRIDE_TRAIN     = 8
STRIDE_TEST      = 16
BATCH_SIZE       = 32
PATIENCE         = 15
LR_PRETRAIN      = 1e-4
LR_FINETUNE1     = 5e-5
LR_FINETUNE2     = 1e-5
EPOCHS_PRETRAIN  = 80
EPOCHS_FINETUNE1 = 40
EPOCHS_FINETUNE2 = 40

# Résultats Sprint 3 pour comparaison
SPRINT3_DEG_CONFIG1 = None   # à remplir après train_convlstm_v3.py
SPRINT3_DEG_CONFIG2 = None
SPRINT2_DEG_CONFIG1 = -7.5
SPRINT2_DEG_CONFIG2 = 51.7
V1_DEGRADATION      = 172.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# CHARGEMENT DES TENSEURS
# =====================================================================

def load_tensors(config_name, zone):
    """
    Charge les tenseurs pré-séquencés depuis splits_v4/.
    Retourne des numpy arrays :
      X_train : (N_train, SEQ_LEN, H, W, F)
      y_train : (N_train, H, W)
    """
    d = SPLITS_V4 / config_name / zone
    if not d.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {d}\n"
            "Lancez d'abord preprocessing/14_build_tensors_v4.py"
        )
    X_train = torch.load(d / "X_train.pt", weights_only=True)
    X_val   = torch.load(d / "X_val.pt",   weights_only=True)
    X_test  = torch.load(d / "X_test.pt",  weights_only=True)
    y_train = torch.load(d / "y_train.pt", weights_only=True)
    y_val   = torch.load(d / "y_val.pt",   weights_only=True)
    y_test  = torch.load(d / "y_test.pt",  weights_only=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# LOSS FUNCTION
# =====================================================================

def weighted_mse_loss(pred, target, epsilon=1e-6):
    """
    Weighted MSE : penalise davantage les erreurs sur les extremes SPI-3.
    weight = 1 + |target|
    Pour SPI-3 = 0 (normal)    : weight = 1.0 (poids standard)
    Pour SPI-3 = -1.5 (severe) : weight = 2.5 (2.5x plus penalise)
    Pour SPI-3 = -3.0 (extreme): weight = 4.0 (4x plus penalise)
    Pas d'hyperparametre arbitraire — poids deterministe.
    """
    weight = 1.0 + torch.abs(target)
    return (weight * (pred - target) ** 2).mean()


# =====================================================================
# EXTRACTION DE PATCHES (identique aux versions précédentes)
# =====================================================================

def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TRAIN):
    """
    Extrait des patches spatiaux depuis des séquences (N, SEQ_LEN, H, W, F).
    Fonctionne directement sur numpy arrays ou tenseurs.
    """
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


# =====================================================================
# BOUCLES D'ENTRAÎNEMENT ET D'ÉVALUATION (identiques aux versions précédentes)
# =====================================================================

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
                log.info("  [%s] Early stopping epoch %d (best: %d, val=%.4f)",
                         phase_name, epoch, best_ep, best_val)
                break
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return best_ep


# =====================================================================
# PIPELINE PAR CONFIG
# =====================================================================

def run_config(config_name, n_features):
    log.info("=" * 60)
    log.info(" SPRINT 4 — Transfer Learning | %s (%d features) | Lead time +1 mois",
             config_name.upper(), n_features)
    log.info(" Device : %s", DEVICE)
    log.info("=" * 60)

    # --- Chargement (tenseurs pré-séquencés : (N, SEQ_LEN, H, W, F)) ---
    log.info("Chargement données HG et MG depuis splits_v4/...")
    X_tr_hg, X_va_hg, X_te_hg, y_tr_hg, y_va_hg, y_te_hg = load_tensors(config_name, "haute_guinee")
    X_tr_mg, X_va_mg, X_te_mg, y_tr_mg, y_va_mg, y_te_mg = load_tensors(config_name, "moyenne_guinee")

    log.info("  HG : train=%s | val=%s | test=%s",
             X_tr_hg.shape, X_va_hg.shape, X_te_hg.shape)
    log.info("  MG : train=%s | val=%s | test=%s",
             X_tr_mg.shape, X_va_mg.shape, X_te_mg.shape)

    # Vérification 5D (séquences pré-construites)
    assert X_tr_hg.ndim == 5, \
        f"X_train HG attendu 5D (N,SEQ,H,W,F), obtenu {X_tr_hg.ndim}D"
    assert X_tr_hg.shape[1] == SEQ_LEN, \
        f"Dimension SEQ_LEN attendue {SEQ_LEN}, obtenu {X_tr_hg.shape[1]}"

    # --- Séquences déjà prêtes — pas de prepare_convlstm_sequences ---
    # X_tr_hg : (260, SEQ_LEN, 90, 90, F)   → directement vers extract_patches
    X_tr_hg_seq = X_tr_hg
    X_va_hg_seq = X_va_hg
    X_te_hg_seq = X_te_hg
    y_tr_hg_seq = y_tr_hg
    y_va_hg_seq = y_va_hg
    y_te_hg_seq = y_te_hg
    X_tr_mg_seq = X_tr_mg
    X_te_mg_seq = X_te_mg
    y_tr_mg_seq = y_tr_mg
    y_te_mg_seq = y_te_mg

    # --- DataLoaders ---
    X_tr_comb = torch.cat([
        extract_patches(X_tr_hg_seq, y_tr_hg_seq)[0],
        extract_patches(X_tr_mg_seq, y_tr_mg_seq)[0],
    ])
    y_tr_comb = torch.cat([
        extract_patches(X_tr_hg_seq, y_tr_hg_seq)[1],
        extract_patches(X_tr_mg_seq, y_tr_mg_seq)[1],
    ])
    pretrain_loader = DataLoader(
        TensorDataset(X_tr_comb.float(), y_tr_comb.float()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader    = make_loader(X_va_hg_seq, y_va_hg_seq, PATCH_SIZE, STRIDE_TRAIN)
    te_hg_loader  = make_loader(X_te_hg_seq, y_te_hg_seq, PATCH_SIZE, STRIDE_TEST, shuffle=False)
    te_mg_loader  = make_loader(X_te_mg_seq, y_te_mg_seq, PATCH_SIZE, STRIDE_TEST, shuffle=False)
    ft_loader     = make_loader(X_tr_hg_seq, y_tr_hg_seq, PATCH_SIZE, STRIDE_TRAIN)

    log.info("  Patches pré-entraînement : %d (HG+MG)", len(X_tr_comb))

    criterion = weighted_mse_loss
    ckpt_pre  = CHECKPOINTS / f"v4w_transfer_{config_name}_pretrain.pt"
    ckpt_ft1  = CHECKPOINTS / f"v4w_transfer_{config_name}_finetune1.pt"
    ckpt_ft2  = CHECKPOINTS / f"v4w_transfer_{config_name}_finetune2.pt"

    # --- Étape 1 : Pré-entraînement HG + MG ---
    log.info("--- ÉTAPE 1 : Pré-entraînement HG+MG (lr=%.0e, %d epochs) ---",
             LR_PRETRAIN, EPOCHS_PRETRAIN)
    model = ConvLSTMEncoderDecoder(
        input_dim=n_features, hidden_dims=HIDDEN_DIMS, kernel_size=KERNEL_SIZE
    ).to(DEVICE)
    log.info("  Paramètres : %d", sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-5)
    t0  = datetime.now()
    run_training(model, pretrain_loader, val_loader, opt, criterion,
                 EPOCHS_PRETRAIN, PATIENCE, ckpt_pre, "PRE-TRAIN")
    log.info("  Pré-entraînement terminé en %.1fs",
             (datetime.now() - t0).total_seconds())

    # --- Étape 2a : Fine-tuning decoder seul ---
    log.info("--- ÉTAPE 2a : Fine-tuning decoder seul (encoder gelé, lr=%.0e) ---",
             LR_FINETUNE1)
    for param in model.encoder.parameters():
        param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info("  Paramètres entraînables : %d / %d",
             sum(p.numel() for p in trainable),
             sum(p.numel() for p in model.parameters()))
    opt_ft1 = torch.optim.Adam(trainable, lr=LR_FINETUNE1, weight_decay=1e-5)
    run_training(model, ft_loader, val_loader, opt_ft1, criterion,
                 EPOCHS_FINETUNE1, PATIENCE, ckpt_ft1, "FINE-TUNE-1")

    # --- Étape 2b : Fine-tuning complet ---
    log.info("--- ÉTAPE 2b : Fine-tuning complet (lr=%.0e) ---", LR_FINETUNE2)
    for param in model.parameters():
        param.requires_grad = True
    opt_ft2 = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE2, weight_decay=1e-5)
    run_training(model, ft_loader, val_loader, opt_ft2, criterion,
                 EPOCHS_FINETUNE2, PATIENCE, ckpt_ft2, "FINE-TUNE-2")

    # --- Évaluation finale ---
    log.info("--- Évaluation finale ---")
    model.load_state_dict(torch.load(ckpt_ft2, weights_only=True))
    _, te_hg_preds, te_hg_targets = eval_model(model, te_hg_loader, criterion)
    _, te_mg_preds, te_mg_targets = eval_model(model, te_mg_loader, criterion)

    m_hg = compute_metrics_spatial(te_hg_targets, te_hg_preds)
    m_mg = compute_metrics_spatial(te_mg_targets, te_mg_preds)

    deg_r2 = (m_hg["r2"] - m_mg["r2"]) / abs(m_hg["r2"]) * 100 \
        if m_hg["r2"] != 0 else float("nan")
    sprint2_ref = SPRINT2_DEG_CONFIG1 if config_name == "config1" else SPRINT2_DEG_CONFIG2

    log.info("  HG test : RMSE=%.4f | R²=%.4f | r=%.4f",
             m_hg["rmse"], m_hg["r2"], m_hg["pearson_r"])
    log.info("  MG test : RMSE=%.4f | R²=%.4f | r=%.4f",
             m_mg["rmse"], m_mg["r2"], m_mg["pearson_r"])
    log.info("  Dégradation R² HG→MG : %+.1f%%", deg_r2)
    log.info("  Lead time : +1 mois (Sprint 4)")
    log.info("  vs Sprint 2 (%s, ref=%+.1f%%) : Δ = %+.1f%%",
             config_name, sprint2_ref, sprint2_ref - deg_r2)
    log.info("  vs v1 (RF, +172.7%%) : amélioration = %+.1f%%",
             V1_DEGRADATION - deg_r2)

    results = {
        "sprint":              "Sprint 4",
        "lead_time":           "+1 mois",
        "config":              config_name,
        "n_features":          n_features,
        "period":              "2000-03 to 2024-12 (298 mois)",
        "strategy":            "pretrain HG+MG → finetune decoder → finetune full",
        "seq_len":             SEQ_LEN,
        "metrics_hg_test":     m_hg,
        "metrics_mg_test":     m_mg,
        "degradation_r2_pct":       round(deg_r2, 1),
        "sprint2_degradation_r2_pct": sprint2_ref,
        "improvement_vs_sprint2":   round(sprint2_ref - deg_r2, 1),
        "v1_degradation_r2_pct":    V1_DEGRADATION,
        "improvement_vs_v1":        round(V1_DEGRADATION - deg_r2, 1),
    }
    return m_hg, m_mg, results


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 60)
    log.info(" SPRINT 4 — ConvLSTM v4 + Transfer Learning + Lead time +1 mois")
    log.info(" Source : splits_v4/  | Device : %s", DEVICE)
    log.info(" Tenseurs pré-séquencés (N, SEQ_LEN, H, W, F)")
    log.info("=" * 60)

    all_results = {}

    # config1 désactivée — uniquement config2 (Weighted MSE) pour commencer
    # m_hg_c1, m_mg_c1, res_c1 = run_config("config1", n_features=2)
    # all_results["config1"] = res_c1

    m_hg_c2, m_mg_c2, res_c2 = run_config("config2", n_features=4)
    all_results["config2"] = res_c2

    out_path = RESULTS_DIR / "v4w_transfer_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("Résultats sauvegardés : %s", out_path)

    # Résumé final
    log.info("=" * 60)
    log.info(" RÉSUMÉ SPRINT 4 — Lead time +1 mois (période étendue 2000-2024)")
    log.info("=" * 60)
    log.info("  %-12s %-8s %-8s %-8s %-8s %-14s %-14s",
             "Config", "HG RMSE", "HG R²", "MG RMSE", "MG R²",
             "Dégr. R² S4", "vs Sprint 2")
    log.info("  " + "-" * 72)
    for cfg, res in all_results.items():
        hg = res["metrics_hg_test"]
        mg = res["metrics_mg_test"]
        log.info("  %-12s %-8.4f %-8.4f %-8.4f %-8.4f %+14.1f%% %+14.1f%%",
                 cfg,
                 hg["rmse"], hg["r2"],
                 mg["rmse"], mg["r2"],
                 res["degradation_r2_pct"],
                 res["improvement_vs_sprint2"])

    log.info("")
    log.info("  Lead time : +1 mois — prévision SPI-3 du mois suivant")
    log.info("  Comparaison v1 (sans lead time, R²_dégr=+172.7%%) :")
    for cfg, res in all_results.items():
        log.info("    %s : amélioration vs v1 = %+.1f%%",
                 cfg, res["improvement_vs_v1"])
    log.info("=" * 60)

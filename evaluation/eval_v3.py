"""
evaluation/eval_v3.py
=======================
Sprint 3 — Évaluation du ConvLSTM config2 sur la période étendue 2000-2024.

Hyperparamètres identiques à train_convlstm_v3.py :
  input_dim    = 4  (NDVI, Précip, LST, VHI_v3)
  HIDDEN_DIMS  = [16, 32]
  KERNEL_SIZE  = 3
  SEQ_LEN      = 3  (depuis models/convlstm.py)
  PATCH_SIZE   = 16
  STRIDE_TEST  = 16

Checkpoint : results/checkpoints/v3_transfer_config2_finetune2.pt
Tenseurs   : data/processed/splits_v3/config2/haute_guinee/

Métriques calculées :
  1. Pixel-level (flatten) : RMSE, MAE, R², Pearson r
  2. Spatial means         : pred_means = preds.mean(axis=(1,2))
                             RMSE, R², Pearson r sur ces moyennes

Résultats sauvegardés : results/tables/v3_eval_config2_hg.json

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 3 — Juillet 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.convlstm import (
    ConvLSTMEncoderDecoder, prepare_convlstm_sequences, SEQ_LEN
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
SPLITS_V3   = BASE_DIR / "data" / "processed" / "splits_v3"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparamètres — identiques à train_convlstm_v3.py
INPUT_DIM   = 4
HIDDEN_DIMS = [16, 32]
KERNEL_SIZE = 3
PATCH_SIZE  = 16
STRIDE_TEST = 16
BATCH_SIZE  = 32

CHECKPOINT = CHECKPOINTS / "v3_transfer_config2_finetune2.pt"
ZONE       = "haute_guinee"
CONFIG     = "config2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# EXTRACTION DE PATCHES (identique à train_convlstm_v3.py)
# =====================================================================

def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TEST):
    """Extrait des patches non-chevauchants (stride=patch_size pour le test)."""
    N, seq_len, H, W, F = X_seq.shape
    X_patches, y_patches = [], []
    for i in range(N):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                X_patches.append(X_seq[i, :, h:h+patch_size, w:w+patch_size, :])
                y_patches.append(y_seq[i, h:h+patch_size, w:w+patch_size])
    return torch.stack(X_patches), torch.stack(y_patches)


# =====================================================================
# MÉTRIQUES
# =====================================================================

def metrics_pixel(targets, preds):
    """Métriques pixel-level sur les arrays aplatis. Exclut les NaN."""
    t = targets.flatten()
    p = preds.flatten()
    valid = ~(np.isnan(t) | np.isnan(p))
    t, p = t[valid], p[valid]
    rmse = float(np.sqrt(mean_squared_error(t, p)))
    mae  = float(mean_absolute_error(t, p))
    r2   = float(r2_score(t, p))
    r, _ = pearsonr(t, p)
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": float(r), "n": int(valid.sum())}


def metrics_spatial_means(targets, preds):
    """
    Métriques sur les moyennes spatiales de chaque patch.
    pred_means   = preds.mean(axis=(1,2))   # (N,)
    target_means = targets.mean(axis=(1,2)) # (N,)
    """
    pred_means   = preds.mean(axis=(1, 2))    # (N,)
    target_means = targets.mean(axis=(1, 2))  # (N,)
    valid = ~(np.isnan(target_means) | np.isnan(pred_means))
    tm, pm = target_means[valid], pred_means[valid]
    rmse = float(np.sqrt(mean_squared_error(tm, pm)))
    r2   = float(r2_score(tm, pm))
    r, _ = pearsonr(tm, pm)
    return {"rmse": rmse, "r2": r2, "pearson_r": float(r), "n": int(valid.sum())}


# =====================================================================
# INFÉRENCE
# =====================================================================

def run_inference(model, loader):
    """Collecte toutes les prédictions et les cibles sur le loader."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(DEVICE)
            pred = model(X_b).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y_b.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" SPRINT 3 — Évaluation ConvLSTM config2")
    log.info(" Zone      : %s", ZONE)
    log.info(" Checkpoint: %s", CHECKPOINT.name)
    log.info(" Device    : %s", DEVICE)
    log.info("=" * 55)

    # --- Vérification checkpoint ---
    if not CHECKPOINT.exists():
        log.error("Checkpoint introuvable : %s", CHECKPOINT)
        log.error("Lancez d'abord training/train_convlstm_v3.py")
        raise FileNotFoundError(CHECKPOINT)

    # --- Chargement des tenseurs test ---
    zone_dir = SPLITS_V3 / CONFIG / ZONE
    log.info("Chargement tenseurs depuis : %s", zone_dir)

    X_test = torch.load(zone_dir / "X_test.pt", weights_only=True).numpy()
    y_test = torch.load(zone_dir / "y_test.pt", weights_only=True).numpy()
    log.info("  X_test : %s | y_test : %s", X_test.shape, y_test.shape)
    log.info("  X_test features : %d (attendu : %d)", X_test.shape[-1], INPUT_DIM)
    assert X_test.shape[-1] == INPUT_DIM, \
        f"X_test a {X_test.shape[-1]} features, attendu {INPUT_DIM}"

    # --- Préparation des séquences ---
    log.info("Préparation séquences (SEQ_LEN=%d)...", SEQ_LEN)
    X_seq, y_seq = prepare_convlstm_sequences(X_test, y_test, seq_len=SEQ_LEN)
    log.info("  X_seq : %s | y_seq : %s", tuple(X_seq.shape), tuple(y_seq.shape))

    # --- Extraction de patches (stride=patch_size, non chevauchants) ---
    log.info("Extraction patches (size=%d, stride=%d)...", PATCH_SIZE, STRIDE_TEST)
    X_patches, y_patches = extract_patches(X_seq, y_seq, PATCH_SIZE, STRIDE_TEST)
    log.info("  X_patches : %s | y_patches : %s",
             tuple(X_patches.shape), tuple(y_patches.shape))

    loader = DataLoader(
        TensorDataset(X_patches.float(), y_patches.float()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Chargement du modèle ---
    log.info("Chargement modèle depuis : %s", CHECKPOINT.name)
    model = ConvLSTMEncoderDecoder(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        kernel_size=KERNEL_SIZE,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location=DEVICE))
    n_params = sum(p.numel() for p in model.parameters())
    log.info("  Paramètres : %d", n_params)

    # --- Inférence ---
    log.info("Inférence sur %d patches...", len(X_patches))
    preds, targets = run_inference(model, loader)
    log.info("  preds   : %s [%.3f, %.3f]",
             preds.shape, float(preds.min()), float(preds.max()))
    log.info("  targets : %s [%.3f, %.3f]",
             targets.shape, float(np.nanmin(targets)), float(np.nanmax(targets)))

    # --- Métriques pixel-level ---
    log.info("")
    log.info("=== Métriques pixel-level (flatten) ===")
    m_pixel = metrics_pixel(targets, preds)
    log.info("  RMSE      : %.4f", m_pixel["rmse"])
    log.info("  MAE       : %.4f", m_pixel["mae"])
    log.info("  R²        : %.4f", m_pixel["r2"])
    log.info("  Pearson r : %.4f", m_pixel["pearson_r"])
    log.info("  N valid   : %d", m_pixel["n"])

    # --- Métriques sur moyennes spatiales ---
    log.info("")
    log.info("=== Métriques sur moyennes spatiales des patches ===")
    m_spatial = metrics_spatial_means(targets, preds)
    log.info("  pred_means   = preds.mean(axis=(1,2))   — shape (%d,)", preds.shape[0])
    log.info("  target_means = targets.mean(axis=(1,2)) — shape (%d,)", targets.shape[0])
    log.info("  RMSE      : %.4f", m_spatial["rmse"])
    log.info("  R²        : %.4f", m_spatial["r2"])
    log.info("  Pearson r : %.4f", m_spatial["pearson_r"])
    log.info("  N patches : %d", m_spatial["n"])

    # --- Comparaison avec Sprint 2 ---
    log.info("")
    log.info("=== Comparaison avec références ===")
    log.info("  Sprint 3 (ce modèle) HG R²   : %.4f", m_pixel["r2"])
    log.info("  Sprint 2 config2 HG R²        : voir transfer_config2_results.json")
    log.info("  v1 ConvLSTM (sans transfer) R²: 0.199")
    log.info("  v1 Random Forest R²           : 0.249")

    # --- Sauvegarde ---
    results = {
        "sprint":      "Sprint 3",
        "config":      CONFIG,
        "zone":        ZONE,
        "checkpoint":  CHECKPOINT.name,
        "n_features":  INPUT_DIM,
        "seq_len":     SEQ_LEN,
        "patch_size":  PATCH_SIZE,
        "stride_test": STRIDE_TEST,
        "n_test_patches": int(len(X_patches)),
        "metrics_pixel": m_pixel,
        "metrics_spatial_means": m_spatial,
        "references": {
            "v1_convlstm_r2":  0.199,
            "v1_rf_r2":        0.249,
        }
    }
    out_path = RESULTS_DIR / "v3_eval_config2_hg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("")
    log.info("Résultats sauvegardés : %s", out_path)
    log.info("=" * 55)

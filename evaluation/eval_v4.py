"""
evaluation/eval_v4.py
=======================
Sprint 4 — Évaluation ConvLSTM config2 avec lead time +1 mois.

Hyperparamètres identiques à train_convlstm_v4.py :
  input_dim    = 4  (NDVI, Précip, LST, VHI_v3)
  HIDDEN_DIMS  = [16, 32]
  KERNEL_SIZE  = 3
  SEQ_LEN      = 3  (depuis models/convlstm.py)
  PATCH_SIZE   = 16
  STRIDE_TEST  = 16

Checkpoint : results/checkpoints/v4_transfer_config2_finetune2.pt
Tenseurs   : data/processed/splits_v4/config2/haute_guinee/

Différences avec eval_v3.py :
  1. Tenseurs pré-séquencés (N, SEQ_LEN, H, W, F) — pas de prepare_convlstm_sequences
  2. Métriques de classification binaire (sécheresse : SPI-3 < -1.0) :
       - Precision, Recall, F1
       - ROC-AUC (score continu = -pred SPI-3)
       - Matrice de confusion

Résultats sauvegardés : results/tables/v4_eval_config2_hg.json

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
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from scipy.stats import pearsonr
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.convlstm import (
    ConvLSTMEncoderDecoder, SEQ_LEN
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparamètres — identiques à train_convlstm_v4.py
INPUT_DIM        = 4
HIDDEN_DIMS      = [16, 32]
KERNEL_SIZE      = 3
PATCH_SIZE       = 16
STRIDE_TEST      = 16
BATCH_SIZE       = 32

# Seuil pour la classification binaire sécheresse
DROUGHT_THRESHOLD = -1.0   # SPI-3 < -1.0 = sécheresse modérée à extrême

CHECKPOINT = CHECKPOINTS / "v4w_transfer_config2_finetune2.pt"
ZONE       = "haute_guinee"
CONFIG     = "config2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# EXTRACTION DE PATCHES (identique aux versions précédentes)
# =====================================================================

def extract_patches(X_seq, y_seq, patch_size=PATCH_SIZE, stride=STRIDE_TEST):
    """Extrait des patches non-chevauchants depuis (N, SEQ_LEN, H, W, F)."""
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
    """Métriques pixel-level (flatten). Exclut les NaN."""
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
    """Métriques sur les moyennes spatiales des patches."""
    pred_means   = preds.mean(axis=(1, 2))
    target_means = targets.mean(axis=(1, 2))
    valid = ~(np.isnan(target_means) | np.isnan(pred_means))
    tm, pm = target_means[valid], pred_means[valid]
    rmse = float(np.sqrt(mean_squared_error(tm, pm)))
    r2   = float(r2_score(tm, pm))
    r, _ = pearsonr(tm, pm)
    return {"rmse": rmse, "r2": r2, "pearson_r": float(r), "n": int(valid.sum())}


def metrics_classification(targets, preds, threshold=DROUGHT_THRESHOLD):
    """
    Métriques de classification binaire pour la détection de sécheresse.

    Classe positive  : SPI-3 < threshold (sécheresse modérée à extrême)
    Classe négative  : SPI-3 >= threshold (conditions normales à humides)

    Score ROC-AUC : utilise -pred comme score de sécheresse
    (plus le SPI-3 prédit est bas, plus le risque de sécheresse est élevé).
    """
    t = targets.flatten()
    p = preds.flatten()
    valid = ~(np.isnan(t) | np.isnan(p))
    t, p = t[valid], p[valid]

    t_bin = (t < threshold).astype(int)   # 1 = sécheresse
    p_bin = (p < threshold).astype(int)

    n_drought = int(t_bin.sum())
    n_total   = len(t_bin)

    prec = float(precision_score(t_bin, p_bin, zero_division=0))
    rec  = float(recall_score(t_bin, p_bin, zero_division=0))
    f1   = float(f1_score(t_bin, p_bin, zero_division=0))

    # ROC-AUC : -p car score plus élevé = plus grande probabilité de sécheresse
    try:
        auc = float(roc_auc_score(t_bin, -p))
    except ValueError:
        auc = float("nan")   # une seule classe présente

    cm = confusion_matrix(t_bin, p_bin).tolist()

    return {
        "threshold":       threshold,
        "n_total":         n_total,
        "n_drought_true":  n_drought,
        "n_drought_pred":  int(p_bin.sum()),
        "prevalence_pct":  round(n_drought / n_total * 100, 2),
        "precision":       prec,
        "recall":          rec,
        "f1":              f1,
        "roc_auc":         auc,
        "confusion_matrix": cm,   # [[TN, FP], [FN, TP]]
    }


# =====================================================================
# INFÉRENCE
# =====================================================================

def run_inference(model, loader):
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
    log.info(" SPRINT 4 — Évaluation ConvLSTM config2")
    log.info(" Zone       : %s", ZONE)
    log.info(" Checkpoint : %s", CHECKPOINT.name)
    log.info(" Lead time  : +1 mois", )
    log.info(" Device     : %s", DEVICE)
    log.info("=" * 55)

    if not CHECKPOINT.exists():
        log.error("Checkpoint introuvable : %s", CHECKPOINT)
        log.error("Lancez d'abord training/train_convlstm_v4.py")
        raise FileNotFoundError(CHECKPOINT)

    # --- Chargement des tenseurs test ---
    zone_dir = SPLITS_V4 / CONFIG / ZONE
    log.info("Chargement tenseurs depuis : %s", zone_dir)

    X_test = torch.load(zone_dir / "X_test.pt", weights_only=True)
    y_test = torch.load(zone_dir / "y_test.pt", weights_only=True)
    log.info("  X_test : %s | y_test : %s", X_test.shape, y_test.shape)

    assert X_test.ndim == 5, \
        f"X_test attendu 5D (N,SEQ,H,W,F), obtenu {X_test.ndim}D"
    assert X_test.shape[-1] == INPUT_DIM, \
        f"X_test a {X_test.shape[-1]} features, attendu {INPUT_DIM}"
    assert X_test.shape[1] == SEQ_LEN, \
        f"SEQ_LEN attendu {SEQ_LEN}, obtenu {X_test.shape[1]}"

    # --- Tenseurs déjà pré-séquencés — pas de prepare_convlstm_sequences ---
    log.info("  Tenseurs pré-séquencés v4 (N, SEQ_LEN, H, W, F) — prêts pour extract_patches")

    # --- Extraction de patches (stride=patch_size, non chevauchants) ---
    log.info("Extraction patches (size=%d, stride=%d)...", PATCH_SIZE, STRIDE_TEST)
    X_patches, y_patches = extract_patches(X_test, y_test, PATCH_SIZE, STRIDE_TEST)
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

    # --- Métriques de classification binaire ---
    log.info("")
    log.info("=== Métriques classification binaire (seuil SPI-3 < %.1f) ===",
             DROUGHT_THRESHOLD)
    m_clf = metrics_classification(targets, preds, threshold=DROUGHT_THRESHOLD)
    log.info("  N total          : %d", m_clf["n_total"])
    log.info("  Sécheresses réelles : %d (%.1f%%)",
             m_clf["n_drought_true"], m_clf["prevalence_pct"])
    log.info("  Sécheresses prédites : %d", m_clf["n_drought_pred"])
    log.info("  Precision        : %.4f", m_clf["precision"])
    log.info("  Recall           : %.4f", m_clf["recall"])
    log.info("  F1               : %.4f", m_clf["f1"])
    log.info("  ROC-AUC          : %.4f", m_clf["roc_auc"])
    cm = m_clf["confusion_matrix"]
    log.info("  Confusion matrix :")
    log.info("    TN=%d  FP=%d", cm[0][0], cm[0][1])
    log.info("    FN=%d  TP=%d", cm[1][0], cm[1][1])

    # --- Comparaison ---
    log.info("")
    log.info("=== Comparaison avec références ===")
    log.info("  Sprint 4 (lead time +1) HG R² : %.4f", m_pixel["r2"])
    log.info("  Sprint 3 (lead time  0) HG R² : voir v3_eval_config2_hg.json")
    log.info("  v1 ConvLSTM (sans transfer)  R²: 0.199")
    log.info("  v1 Random Forest             R²: 0.249")

    # --- Sauvegarde ---
    results = {
        "sprint":              "Sprint 4",
        "lead_time":           "+1 mois",
        "config":              CONFIG,
        "zone":                ZONE,
        "checkpoint":          CHECKPOINT.name,
        "n_features":          INPUT_DIM,
        "seq_len":             SEQ_LEN,
        "patch_size":          PATCH_SIZE,
        "stride_test":         STRIDE_TEST,
        "drought_threshold":   DROUGHT_THRESHOLD,
        "n_test_sequences":    int(X_test.shape[0]),
        "n_test_patches":      int(len(X_patches)),
        "metrics_pixel":       m_pixel,
        "metrics_spatial_means": m_spatial,
        "metrics_classification": m_clf,
        "references": {
            "v1_convlstm_r2":  0.199,
            "v1_rf_r2":        0.249,
        }
    }
    out_path = RESULTS_DIR / "v4w_eval_config2_hg.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("")
    log.info("Résultats sauvegardés : %s", out_path)
    log.info("=" * 55)

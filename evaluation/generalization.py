"""
evaluation/generalization.py
=============================
Test de généralisation sur la zone de contrôle — Moyenne Guinée.

Applique le meilleur modèle (Random Forest) entraîné sur Haute Guinée
directement sur la Moyenne Guinée sans ré-entraînement.

Objectif :
Mesurer la dégradation de performance entre la zone d'entraînement
(Haute Guinée) et la zone de contrôle (Moyenne Guinée).

Selon le protocole : dégradation acceptable <= 15% du R².

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import torch
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline_rf import prepare_rf_features
import joblib

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
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, label=""):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    if label:
        log.info("  [%s] RMSE=%.4f | MAE=%.4f | R²=%.4f | r=%.4f",
                 label, rmse, mae, r2, float(r))
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": float(r)}


def plot_generalization_comparison(m_hg, m_mg):
    """Graphique comparant les performances HG vs MG."""
    metrics_names = ["RMSE", "R²", "Pearson r"]
    hg_vals = [m_hg["rmse"], m_hg["r2"], m_hg["pearson_r"]]
    mg_vals = [m_mg["rmse"], m_mg["r2"], m_mg["pearson_r"]]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, hg_vals, width, label="Haute Guinée (train)",
                   color="#4575b4", alpha=0.85)
    bars2 = ax.bar(x + width/2, mg_vals, width, label="Moyenne Guinée (contrôle)",
                   color="#74add1", alpha=0.85, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_title("Généralisation Random Forest — Haute Guinée vs Moyenne Guinée\n"
                 "Modèle entraîné sur HG, testé sur MG sans ré-entraînement",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "generalisation_hg_vs_mg.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Graphique généralisation sauvegardé : %s", path)


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" PHASE 4 — Test de généralisation sur Moyenne Guinée")
    log.info(" Modèle : Random Forest (meilleur RMSE)")
    log.info(" Entraîné sur : Haute Guinée")
    log.info(" Testé sur    : Moyenne Guinée (sans ré-entraînement)")
    log.info("=" * 55)

    # --- Charger le modèle RF entraîné sur HG ---
    log.info("Chargement du modèle Random Forest...")
    model = joblib.load(CHECKPOINTS / "rf_baseline.joblib")

    # --- Données Haute Guinée (référence) ---
    log.info("Évaluation sur Haute Guinée (référence)...")
    X_te_hg = torch.load(SPLITS_HG / "X_test.pt",  weights_only=True).numpy()
    y_te_hg = torch.load(SPLITS_HG / "y_test.pt",  weights_only=True).numpy()
    X_tr_hg = torch.load(SPLITS_HG / "X_train.pt", weights_only=True).numpy()
    y_tr_hg = torch.load(SPLITS_HG / "y_train.pt", weights_only=True).numpy()

    X_te_hg_rf, y_te_hg_rf = prepare_rf_features(X_te_hg, y_te_hg)
    pred_hg = model.predict(X_te_hg_rf)
    m_hg = compute_metrics(y_te_hg_rf, pred_hg, "HG test")

    # --- Données Moyenne Guinée (généralisation) ---
    log.info("Évaluation sur Moyenne Guinée (généralisation)...")
    X_te_mg = torch.load(SPLITS_MG / "X_test.pt",  weights_only=True).numpy()
    y_te_mg = torch.load(SPLITS_MG / "y_test.pt",  weights_only=True).numpy()

    X_te_mg_rf, y_te_mg_rf = prepare_rf_features(X_te_mg, y_te_mg)
    pred_mg = model.predict(X_te_mg_rf)
    m_mg = compute_metrics(y_te_mg_rf, pred_mg, "MG test")

    # --- Calcul de la dégradation ---
    deg_rmse = (m_mg["rmse"] - m_hg["rmse"]) / m_hg["rmse"] * 100
    deg_r2   = (m_hg["r2"]  - m_mg["r2"])   / abs(m_hg["r2"]) * 100 \
               if m_hg["r2"] != 0 else 0

    # --- Graphique ---
    plot_generalization_comparison(m_hg, m_mg)

    # --- Résultats ---
    log.info("=" * 55)
    log.info(" RÉSULTATS GÉNÉRALISATION")
    log.info("=" * 55)
    log.info("  %-20s  RMSE    R²      r", "Zone")
    log.info("  " + "-" * 45)
    log.info("  %-20s  %.4f  %.4f  %.4f", "Haute Guinée (ref)", m_hg["rmse"], m_hg["r2"], m_hg["pearson_r"])
    log.info("  %-20s  %.4f  %.4f  %.4f", "Moyenne Guinée", m_mg["rmse"], m_mg["r2"], m_mg["pearson_r"])
    log.info("  " + "-" * 45)
    log.info("  Dégradation RMSE : %+.1f%%", deg_rmse)
    log.info("  Dégradation R²   : %+.1f%%", deg_r2)
    log.info("  Seuil protocole  : <= 15%% de dégradation R²")

    if abs(deg_r2) <= 15:
        verdict = "ACCEPTABLE"
        log.info("  VERDICT : Généralisation ACCEPTABLE ✓")
    else:
        verdict = "DEGRADATION_IMPORTANTE"
        log.info("  VERDICT : Dégradation importante — à documenter")

    # Sauvegarder
    results = {
        "phase": "Phase 4 — Généralisation",
        "model": "Random Forest",
        "train_zone": "haute_guinee",
        "test_zone": "moyenne_guinee",
        "metrics_hg": m_hg,
        "metrics_mg": m_mg,
        "degradation_rmse_pct": round(deg_rmse, 1),
        "degradation_r2_pct":   round(deg_r2, 1),
        "threshold_protocol": 15.0,
        "verdict": verdict
    }
    with open(RESULTS_DIR / "generalisation_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("  Résultats sauvegardés.")
    log.info("=" * 55)
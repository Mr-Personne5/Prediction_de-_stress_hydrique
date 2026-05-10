"""
training/train_rf.py
=====================
Entraînement et évaluation du baseline Random Forest.

Ce script :
1. Charge les tenseurs d'entraînement (Haute Guinée uniquement)
2. Prépare les features tabulaires (agrégation spatiale + lags)
3. Entraîne le Random Forest
4. Évalue sur train, val et test
5. Sauvegarde le modèle et les résultats

Auteur : Djiba Kaba — Master IASD UKAG
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import torch
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

# Fixer la graine aléatoire pour la reproductibilité
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Importer le modèle
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline_rf import (
    prepare_rf_features, build_rf_model, compute_metrics
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Chemins ---
BASE_DIR     = Path(__file__).parent.parent
SPLITS_DIR   = BASE_DIR / "data" / "processed" / "splits" / "haute_guinee"
CHECKPOINTS  = BASE_DIR / "results" / "checkpoints"
RESULTS_DIR  = BASE_DIR / "results" / "tables"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_tensors():
    """Charge les tenseurs PyTorch et les convertit en numpy."""
    log.info("Chargement des tenseurs...")

    X_train = torch.load(SPLITS_DIR / "X_train.pt", weights_only=True).numpy()
    X_val   = torch.load(SPLITS_DIR / "X_val.pt",   weights_only=True).numpy()
    X_test  = torch.load(SPLITS_DIR / "X_test.pt",  weights_only=True).numpy()
    y_train = torch.load(SPLITS_DIR / "y_train.pt", weights_only=True).numpy()
    y_val   = torch.load(SPLITS_DIR / "y_val.pt",   weights_only=True).numpy()
    y_test  = torch.load(SPLITS_DIR / "y_test.pt",  weights_only=True).numpy()

    log.info("  X_train : %s | y_train : %s", X_train.shape, y_train.shape)
    log.info("  X_val   : %s | y_val   : %s", X_val.shape,   y_val.shape)
    log.info("  X_test  : %s | y_test  : %s", X_test.shape,  y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    log.info("=" * 50)
    log.info(" PHASE 3 — Baseline Random Forest")
    log.info(" Zone : Haute Guinée")
    log.info(" Seed : %d", SEED)
    log.info("=" * 50)

    # --- Charger les données ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_tensors()

    # --- Préparer les features RF ---
    log.info("Préparation des features tabulaires (agrégation + lags)...")
    X_train_rf, y_train_rf = prepare_rf_features(X_train, y_train)
    X_val_rf,   y_val_rf   = prepare_rf_features(X_val,   y_val)
    X_test_rf,  y_test_rf  = prepare_rf_features(X_test,  y_test)

    log.info("  X_train_rf : %s", X_train_rf.shape)
    log.info("  X_val_rf   : %s", X_val_rf.shape)
    log.info("  X_test_rf  : %s", X_test_rf.shape)

    # --- Construire et entraîner le modèle ---
    log.info("Construction du modèle Random Forest...")
    model = build_rf_model(n_estimators=200, max_depth=15, random_state=SEED)

    log.info("Entraînement en cours...")
    start = datetime.now()
    model.fit(X_train_rf, y_train_rf)
    duration = (datetime.now() - start).total_seconds()
    log.info("Entraînement terminé en %.1f secondes", duration)

    # --- Évaluation ---
    log.info("Évaluation sur les 3 jeux...")

    pred_train = model.predict(X_train_rf)
    pred_val   = model.predict(X_val_rf)
    pred_test  = model.predict(X_test_rf)

    metrics_train = compute_metrics(y_train_rf, pred_train, "TRAIN")
    metrics_val   = compute_metrics(y_val_rf,   pred_val,   "VAL")
    metrics_test  = compute_metrics(y_test_rf,  pred_test,  "TEST")

    # --- Importance des features ---
    feature_names = []
    for lag in range(4):  # t, t-1, t-2, t-3
        feature_names.append(f"NDVI_t-{lag}")
        feature_names.append(f"Precip_t-{lag}")

    importances = model.feature_importances_
    log.info("Importance des features :")
    for name, imp in sorted(zip(feature_names, importances),
                            key=lambda x: -x[1]):
        log.info("  %-15s : %.4f", name, imp)

    # --- Sauvegarder le modèle ---
    model_path = CHECKPOINTS / "rf_baseline.joblib"
    joblib.dump(model, model_path)
    log.info("Modèle sauvegardé : %s", model_path)

    # --- Sauvegarder les résultats ---
    results = {
        "model": "Random Forest",
        "zone": "haute_guinee",
        "n_estimators": 200,
        "max_depth": 15,
        "n_lags": 3,
        "training_time_sec": round(duration, 1),
        "metrics": {
            "train": metrics_train,
            "val":   metrics_val,
            "test":  metrics_test
        },
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(feature_names, importances)
        }
    }

    results_path = RESULTS_DIR / "rf_baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Résultats sauvegardés : %s", results_path)

    # --- Résumé final ---
    log.info("=" * 50)
    log.info(" RÉSUMÉ — Random Forest Baseline")
    log.info("=" * 50)
    log.info("  TRAIN : RMSE=%.4f | R²=%.4f", metrics_train["rmse"], metrics_train["r2"])
    log.info("  VAL   : RMSE=%.4f | R²=%.4f", metrics_val["rmse"],   metrics_val["r2"])
    log.info("  TEST  : RMSE=%.4f | R²=%.4f", metrics_test["rmse"],  metrics_test["r2"])
    log.info("=" * 50)
    log.info(" Ces valeurs sont la BASELINE à surpasser par le ConvLSTM.")
    log.info("=" * 50)
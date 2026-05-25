"""
models/baseline_rf.py
======================
Baseline 1 — Random Forest non spatial.

Rôle dans l'étude :
Le Random Forest est notre référence non spatiale. Il travaille sur des
séries temporelles agrégées par zone (une valeur moyenne par mois, pas
une carte). Il ne voit pas la géographie — chaque pixel est indépendant.

Si le ConvLSTM ne surpasse pas ce modèle, cela signifie que la dimension
spatiale n'apporte pas de valeur ajoutée pour prédire le SPI-3.

Stratégie d'entrée :
- X_train shape original : (84, 90, 90, 2) — (temps, H, W, features)
- Pour le RF, on agrège spatialement : moyenne sur H et W
- X_rf shape : (84, 2) — (temps, features_moyennes)
- On ajoute des features lag (valeurs des 3 mois précédents) pour
  capturer la mémoire temporelle sans LSTM

Features construites :
- NDVI_t, NDVI_t-1, NDVI_t-2, NDVI_t-3
- Precip_t, Precip_t-1, Precip_t-2, Precip_t-3
Total : 8 features par pas de temps

Cible :
- y_rf : moyenne spatiale du SPI-3 (valeur scalaire par mois)

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import joblib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Nombre de mois de lag (mémoire temporelle)
N_LAGS = 3


def prepare_rf_features(X, y, n_lags=N_LAGS):
    """
    Transforme les tenseurs 4D en features tabulaires pour Random Forest.

    Étapes :
    1. Agrégation spatiale : moyenne sur H et W -> shape (T, 2)
    2. Construction des features lag (t, t-1, t-2, t-3)
    3. Agrégation spatiale de y -> shape (T,)

    Args:
        X      : np.ndarray (T, H, W, 2) — tenseur d'entrée normalisé
        y      : np.ndarray (T, H, W)    — SPI-3
        n_lags : int — nombre de mois de lag

    Returns:
        X_rf : np.ndarray (T - n_lags, n_lags * 2) — features tabulaires
        y_rf : np.ndarray (T - n_lags,)            — cibles scalaires
    """
    T = X.shape[0]

    # Agrégation spatiale : moyenne sur H et W
    # Shape : (T, 2)
    X_mean = X.reshape(T, -1, 2).mean(axis=1)  # (T, 2)
    y_mean = y.reshape(T, -1).mean(axis=1)       # (T,)

    # Construction des features avec lag
    # Pour chaque pas de temps t >= n_lags :
    # features = [NDVI_t, NDVI_t-1, ..., NDVI_t-n, Precip_t, ..., Precip_t-n]
    X_rf_list = []
    y_rf_list = []

    for t in range(n_lags, T):
        # Features : valeurs actuelles et passées
        feat = []
        for lag in range(n_lags + 1):  # t, t-1, t-2, t-3
            feat.append(X_mean[t - lag, 0])  # NDVI
            feat.append(X_mean[t - lag, 1])  # Précipitations
        X_rf_list.append(feat)
        y_rf_list.append(y_mean[t])

    X_rf = np.array(X_rf_list, dtype=np.float32)
    y_rf = np.array(y_rf_list, dtype=np.float32)

    return X_rf, y_rf


def build_rf_model(n_estimators=200, max_depth=15, random_state=42):
    """
    Construit le modèle Random Forest.

    Hyperparamètres :
    - n_estimators : nombre d'arbres (200 = bon compromis vitesse/performance)
    - max_depth    : profondeur max des arbres (15 = évite l'overfitting)
    - random_state : graine aléatoire pour la reproductibilité

    Args:
        n_estimators : int
        max_depth    : int
        random_state : int

    Returns:
        RandomForestRegressor
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,           # Utiliser tous les coeurs CPU
        random_state=random_state,
        verbose=0
    )


def compute_metrics(y_true, y_pred, split_name=""):
    """
    Calcule les 4 métriques d'évaluation standard.

    Args:
        y_true     : np.ndarray 1D — valeurs réelles
        y_pred     : np.ndarray 1D — valeurs prédites
        split_name : str — nom du jeu pour les logs

    Returns:
        dict avec rmse, mae, r2, pearson_r
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)

    if split_name:
        log.info("  [%s] RMSE=%.4f | MAE=%.4f | R²=%.4f | r=%.4f",
                 split_name, rmse, mae, r2, r)

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": float(r)}
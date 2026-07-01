"""
preprocessing/10_build_tensors_v2.py
======================================
SPRINT 2 — Construction des tenseurs v2 avec 4 features.

Features :
  0 : NDVI normalisé
  1 : Précipitations normalisées
  2 : LST normalisée
  3 : VHI_v2 normalisé (vrai TCI)

Cible : SPI-3 (identique à v1)

Deux configs produites :
  - splits_v2/config1/ : NDVI + Précip seulement (comparaison directe v1)
  - splits_v2/config2/ : NDVI + Précip + LST + VHI_v2 (4 features)

Split identique à v1 : Train 2015-2021 / Val 2022 / Test 2023-2024
Normalisation MinMax sur train set uniquement.

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 2 - 01/07/2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import xarray as xr
import pandas as pd
import torch
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
INDICES_DIR = PROC_DIR / "indices"
SPLITS_V2   = PROC_DIR / "splits_v2"

TRAIN_END = 2021
VAL_END   = 2022
TEST_END  = 2024

ZONES = ["haute_guinee", "moyenne_guinee"]

CONFIGS = {
    "config1": {
        "label": "NDVI + Précipitations (comparaison v1)",
        "features": ["NDVI", "precipitation"],
        "sources": ["ndvi", "chirps", None, None],
    },
    "config2": {
        "label": "NDVI + Précipitations + LST + VHI_v2 (4 features)",
        "features": ["NDVI", "precipitation", "LST", "VHI_v2"],
        "sources": ["ndvi", "chirps", "lst", "vhi_v2"],
    },
}


def load_feature(zone, source, var_name):
    """Charge une feature depuis son fichier NetCDF."""
    if source in ["ndvi", "chirps", "lst"]:
        path = PROC_DIR / f"{source}_{zone}.nc"
    else:
        path = INDICES_DIR / f"{source}_{zone}.nc"

    ds = xr.open_dataset(path)

    # Pour CHIRPS : extraire seulement 2015-2024
    if source == "chirps":
        times = pd.to_datetime(ds[var_name].time.values)
        mask  = (times.year >= 2015) & (times.year <= 2024)
        data  = ds[var_name].values[mask].astype(np.float32)
    else:
        data = ds[var_name].values.astype(np.float32)

    ds.close()
    log.info("    %s : shape=%s | min=%.3f | max=%.3f | NaN=%.1f%%",
             var_name, data.shape, float(np.nanmin(data)),
             float(np.nanmax(data)), np.isnan(data).mean()*100)
    return data


def split_data(data, times):
    mask_train = times.year <= TRAIN_END
    mask_val   = times.year == VAL_END
    mask_test  = (times.year > VAL_END) & (times.year <= TEST_END)
    return data[mask_train], data[mask_val], data[mask_test]


def normalize_minmax(train, val, test):
    vmin = float(np.nanmin(train))
    vmax = float(np.nanmax(train))
    rng  = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0
    tr = np.clip((train - vmin) / rng, 0.0, 1.0)
    va = np.clip((val   - vmin) / rng, 0.0, 1.0)
    te = np.clip((test  - vmin) / rng, 0.0, 1.0)
    return tr, va, te, vmin, vmax


def build_config(config_name, config, zone):
    log.info("  Config : %s | %s", config_name, config["label"])

    out_dir = SPLITS_V2 / config_name / zone
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "X_train.pt").exists():
        log.info("    Deja construit — ignore")
        return

    # Charger les features
    feature_names = config["features"]
    source_map = {
        "NDVI": ("ndvi", "NDVI"),
        "precipitation": ("chirps", "precipitation"),
        "LST": ("lst", "LST"),
        "VHI_v2": ("vhi_v2", "VHI_v2"),
    }

    # Charger les temps depuis NDVI (référence 2015-2024)
    ds_ref = xr.open_dataset(PROC_DIR / f"ndvi_{zone}.nc")
    times  = pd.to_datetime(ds_ref["NDVI"].time.values)
    ds_ref.close()

    # Charger et splitter chaque feature
    features_train, features_val, features_test = [], [], []
    norm_params = {}

    for feat_name in feature_names:
        src, var = source_map[feat_name]
        data = load_feature(zone, src, var)

        # Remplir les NaN résiduels (LST a 1.7%)
        if np.isnan(data).any():
            col_mean = np.nanmean(data, axis=0, keepdims=True)
            nan_mask = np.isnan(data)
            data[nan_mask] = np.broadcast_to(col_mean, data.shape)[nan_mask]

        tr, va, te = split_data(data, times)
        tr_n, va_n, te_n, vmin, vmax = normalize_minmax(tr, va, te)

        features_train.append(tr_n)
        features_val.append(va_n)
        features_test.append(te_n)
        norm_params[feat_name] = {"min": vmin, "max": vmax}

    # Stack features -> (T, H, W, F)
    X_train = np.stack(features_train, axis=-1).astype(np.float32)
    X_val   = np.stack(features_val,   axis=-1).astype(np.float32)
    X_test  = np.stack(features_test,  axis=-1).astype(np.float32)

    # Charger la cible SPI-3
    ds_spi = xr.open_dataset(INDICES_DIR / f"spi3_{zone}.nc")
    spi3   = ds_spi["SPI3"].values.astype(np.float32)
    ds_spi.close()

    y_train, y_val, y_test = split_data(spi3, times)

    log.info("    X_train : %s | y_train : %s", X_train.shape, y_train.shape)
    log.info("    X_val   : %s | X_test  : %s", X_val.shape, X_test.shape)

    # Sauvegarder
    torch.save(torch.from_numpy(X_train), out_dir / "X_train.pt")
    torch.save(torch.from_numpy(X_val),   out_dir / "X_val.pt")
    torch.save(torch.from_numpy(X_test),  out_dir / "X_test.pt")
    torch.save(torch.from_numpy(y_train), out_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_val),   out_dir / "y_val.pt")
    torch.save(torch.from_numpy(y_test),  out_dir / "y_test.pt")

    np.savez(out_dir / "norm_params.npz", **{
        f"{k}_{p}": v for k, d in norm_params.items() for p, v in d.items()
    })

    with open(out_dir / "config_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Config : {config_name}\n")
        f.write(f"Label  : {config['label']}\n")
        f.write(f"Zone   : {zone}\n")
        f.write(f"Features ({len(feature_names)}) : {feature_names}\n")
        f.write(f"X_train : {X_train.shape}\n")
        f.write(f"X_val   : {X_val.shape}\n")
        f.write(f"X_test  : {X_test.shape}\n")
        f.write(f"Split : train<={TRAIN_END} | val={VAL_END} | test>{VAL_END}\n")

    log.info("    OK : %s", out_dir)


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" SPRINT 2 — Construction tenseurs v2")
    log.info(" Config 1 : 2 features (NDVI + Précip)")
    log.info(" Config 2 : 4 features (+ LST + VHI_v2)")
    log.info("=" * 55)

    for config_name, config in CONFIGS.items():
        log.info("=== %s ===", config_name.upper())
        for zone in ZONES:
            log.info("  Zone : %s", zone.upper())
            build_config(config_name, config, zone)

    log.info("=" * 55)
    log.info(" Tenseurs dans : %s", SPLITS_V2)
    log.info("=" * 55)
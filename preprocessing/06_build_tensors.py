"""
preprocessing/06_build_tensors.py
=====================================
Construction des tenseurs 4D et split train/val/test.

Ce script est la dernière étape du preprocessing. Il produit les données
directement consommables par les modèles de deep learning.

Tenseur d'entrée X :
- Shape : (T, H, W, 2) — temps × hauteur × largeur × features
- Feature 0 : NDVI normalisé (MinMax sur train set)
- Feature 1 : Précipitations normalisées (MinMax sur train set)
- Période : 2015-2024 (120 mois)

Tenseur de sortie y :
- Shape : (T, H, W) — temps × hauteur × largeur
- Valeurs : SPI-3 (variable cible du modèle)
- Période : 2015-2024 (120 mois)

Split temporel chronologique (règle absolue — ne jamais randomiser) :
- Train : 2015-2021 — 84 mois (70%)
- Val   : 2022      — 12 mois (10%)
- Test  : 2023-2024 — 24 mois (20%)

Règle de normalisation :
- Les bornes MinMax sont calculées UNIQUEMENT sur le train set
- Ces mêmes bornes sont appliquées sur val et test
- Évite la fuite de données (data leakage) du futur vers le passé

Zones :
- Haute Guinée : zone principale (entraînement + évaluation)
- Moyenne Guinée : zone de contrôle (test de généralisation uniquement)

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
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

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Chemins ---
BASE_DIR       = Path(__file__).parent.parent
PROC_DIR       = BASE_DIR / "data" / "processed"
INDICES_DIR    = PROC_DIR / "indices"
SPLITS_DIR     = PROC_DIR / "splits"
SPLITS_V3_DIR  = PROC_DIR / "splits_v3"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_V3_DIR.mkdir(parents=True, exist_ok=True)

# --- Split temporel ---
# Bornes val/test identiques v1 et v3 → résultats comparables
TRAIN_END = 2021   # inclus
VAL_END   = 2022   # inclus
TEST_END  = 2024   # inclus


# =====================================================================
# FONCTIONS
# =====================================================================

def load_data(zone):
    """
    Charge les données NDVI, précipitations et SPI-3 pour une zone.

    Returns:
        ndvi   : np.ndarray (T, H, W) — 120 mois
        precip : np.ndarray (T, H, W) — 120 mois
        spi3   : np.ndarray (T, H, W) — 120 mois
        times  : pd.DatetimeIndex
    """
    log.info("  Chargement des données — %s", zone)

    ds_ndvi   = xr.open_dataset(PROC_DIR / f"ndvi_{zone}.nc")
    ds_chirps = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    ds_spi    = xr.open_dataset(INDICES_DIR / f"spi3_{zone}.nc")

    # Aligner les périodes sur 2015-2024
    times = pd.to_datetime(ds_spi["SPI3"].time.values)

    ndvi   = ds_ndvi["NDVI"].sel(
        time=slice("2015-01-01", "2024-12-31")
    ).values.astype(np.float32)

    precip = ds_chirps["precipitation"].sel(
        time=slice("2015-01-01", "2024-12-31")
    ).values.astype(np.float32)

    spi3   = ds_spi["SPI3"].values.astype(np.float32)

    ds_ndvi.close()
    ds_chirps.close()
    ds_spi.close()

    log.info("    NDVI   shape : %s", ndvi.shape)
    log.info("    Precip shape : %s", precip.shape)
    log.info("    SPI-3  shape : %s", spi3.shape)

    return ndvi, precip, spi3, times


def split_data(data, times, label="data"):
    """
    Split chronologique en train/val/test.

    Args:
        data  : np.ndarray (T, ...) — données à splitter
        times : pd.DatetimeIndex
        label : str — nom pour les logs

    Returns:
        dict avec clés 'train', 'val', 'test'
    """
    mask_train = times.year <= TRAIN_END
    mask_val   = times.year == VAL_END
    mask_test  = (times.year > VAL_END) & (times.year <= TEST_END)

    splits = {
        "train": data[mask_train],
        "val":   data[mask_val],
        "test":  data[mask_test],
    }

    log.info("    %s split : train=%d | val=%d | test=%d mois",
             label,
             splits["train"].shape[0],
             splits["val"].shape[0],
             splits["test"].shape[0])

    return splits


def normalize_minmax(data_train, data_val, data_test):
    """
    Normalisation MinMax calculée sur train, appliquée sur val et test.

    RÈGLE CRITIQUE : les bornes ne sont JAMAIS calculées sur val ou test.
    Cela évite la fuite de données (data leakage).

    Args:
        data_train, data_val, data_test : np.ndarray

    Returns:
        train_norm, val_norm, test_norm, vmin, vmax
    """
    vmin = float(np.nanmin(data_train))
    vmax = float(np.nanmax(data_train))
    rng  = vmax - vmin
    if rng < 1e-8:
        rng = 1.0

    train_norm = (data_train - vmin) / rng
    val_norm   = (data_val   - vmin) / rng
    test_norm  = (data_test  - vmin) / rng

    # Clipper entre 0 et 1 (val/test peuvent légèrement dépasser les bornes train)
    train_norm = np.clip(train_norm, 0.0, 1.0)
    val_norm   = np.clip(val_norm,   0.0, 1.0)
    test_norm  = np.clip(test_norm,  0.0, 1.0)

    return train_norm, val_norm, test_norm, vmin, vmax


def build_tensors(zone):
    """
    Construit et sauvegarde les tenseurs 4D pour une zone.

    Sorties dans data/processed/splits/{zone}/ :
    - X_train.pt, X_val.pt, X_test.pt : tenseurs d'entrée (T, H, W, 2)
    - y_train.pt, y_val.pt, y_test.pt : tenseurs de sortie (T, H, W)
    - norm_params.npz : bornes de normalisation (vmin/vmax par feature)
    - split_info.txt  : résumé du split
    """
    log.info("=== Construction tenseurs — %s ===", zone.upper())

    zone_dir = SPLITS_DIR / zone
    zone_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà fait
    if (zone_dir / "X_train.pt").exists():
        log.info("  Déjà construit — ignoré : %s", zone)
        return

    # --- Charger les données ---
    ndvi, precip, spi3, times = load_data(zone)

    # --- Split ---
    log.info("  Split temporel : train<=%d | val=%d | test>%d",
             TRAIN_END, VAL_END, VAL_END)

    ndvi_splits   = split_data(ndvi,   times, "NDVI")
    precip_splits = split_data(precip, times, "Precip")
    spi3_splits   = split_data(spi3,   times, "SPI-3")

    # --- Normalisation (bornes calculées sur train uniquement) ---
    log.info("  Normalisation MinMax (bornes sur train set uniquement)...")

    ndvi_tr, ndvi_va, ndvi_te, ndvi_min, ndvi_max = normalize_minmax(
        ndvi_splits["train"], ndvi_splits["val"], ndvi_splits["test"]
    )
    precip_tr, precip_va, precip_te, precip_min, precip_max = normalize_minmax(
        precip_splits["train"], precip_splits["val"], precip_splits["test"]
    )

    log.info("    NDVI   : min=%.4f | max=%.4f (train set)",
             ndvi_min, ndvi_max)
    log.info("    Precip : min=%.4f | max=%.4f mm/mois (train set)",
             precip_min, precip_max)

    # --- Construction tenseurs X : (T, H, W, 2) ---
    # Feature 0 : NDVI normalisé
    # Feature 1 : Précipitations normalisées
    X_train = np.stack([ndvi_tr, precip_tr], axis=-1)
    X_val   = np.stack([ndvi_va, precip_va], axis=-1)
    X_test  = np.stack([ndvi_te, precip_te], axis=-1)

    # --- Tenseurs y : (T, H, W) — SPI-3 non normalisé ---
    y_train = spi3_splits["train"]
    y_val   = spi3_splits["val"]
    y_test  = spi3_splits["test"]

    log.info("  Shapes finaux :")
    log.info("    X_train : %s | y_train : %s", X_train.shape, y_train.shape)
    log.info("    X_val   : %s | y_val   : %s", X_val.shape,   y_val.shape)
    log.info("    X_test  : %s | y_test  : %s", X_test.shape,  y_test.shape)

    # --- Sauvegarder en format PyTorch (.pt) ---
    torch.save(torch.from_numpy(X_train), zone_dir / "X_train.pt")
    torch.save(torch.from_numpy(X_val),   zone_dir / "X_val.pt")
    torch.save(torch.from_numpy(X_test),  zone_dir / "X_test.pt")
    torch.save(torch.from_numpy(y_train), zone_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_val),   zone_dir / "y_val.pt")
    torch.save(torch.from_numpy(y_test),  zone_dir / "y_test.pt")

    # --- Sauvegarder les paramètres de normalisation ---
    np.savez(
        zone_dir / "norm_params.npz",
        ndvi_min=ndvi_min, ndvi_max=ndvi_max,
        precip_min=precip_min, precip_max=precip_max
    )

    # --- Résumé ---
    with open(zone_dir / "split_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone : {zone}\n")
        f.write(f"Split : train<={TRAIN_END} | val={VAL_END} | test>{VAL_END}\n")
        f.write(f"X_train : {X_train.shape} — dtype float32\n")
        f.write(f"X_val   : {X_val.shape}\n")
        f.write(f"X_test  : {X_test.shape}\n")
        f.write(f"y_train : {y_train.shape} — SPI-3 non normalisé\n")
        f.write(f"y_val   : {y_val.shape}\n")
        f.write(f"y_test  : {y_test.shape}\n")
        f.write(f"Normalisation NDVI   : min={ndvi_min:.4f} | max={ndvi_max:.4f}\n")
        f.write(f"Normalisation Precip : min={precip_min:.4f} | max={precip_max:.4f} mm/mois\n")
        f.write(f"Features : [0]=NDVI normalisé | [1]=Précipitations normalisées\n")
        f.write(f"Cible    : SPI-3 (non normalisé — régression directe)\n")

    log.info("  ✓ Tenseurs sauvegardés dans : %s", zone_dir)

    # --- Taille des fichiers ---
    total_mb = sum(f.stat().st_size for f in zone_dir.glob("*.pt")) / 1e6
    log.info("  Taille totale fichiers .pt : %.1f MB", total_mb)


# =====================================================================
# SPRINT 3 — TENSEURS V3 : NDVI + PRÉCIP + LST (3 features, 2000-2024)
# =====================================================================

def load_data_v3(zone):
    """
    Charge NDVI v2, LST, CHIRPS et SPI-3 v2, puis aligne sur leur intersection.
    Période effective : 2000-03 à 2024-12 (LST disponible depuis 2000-03).
    """
    log.info("  Chargement données Sprint 3 — %s", zone)

    ds_ndvi   = xr.open_dataset(PROC_DIR / f"ndvi_v2_{zone}.nc")
    ds_lst    = xr.open_dataset(PROC_DIR / f"lst_{zone}.nc")
    ds_chirps = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    ds_spi    = xr.open_dataset(INDICES_DIR / f"spi3_v2_{zone}.nc")

    t_ndvi   = pd.to_datetime(ds_ndvi["NDVI"].time.values)
    t_lst    = pd.to_datetime(ds_lst["LST"].time.values)
    t_chirps = pd.to_datetime(ds_chirps["precipitation"].time.values)
    t_spi    = pd.to_datetime(ds_spi["SPI3"].time.values)

    common = t_ndvi.intersection(t_lst).intersection(t_chirps).intersection(t_spi)
    log.info("  Période commune : %s → %s (%d mois)",
             str(common[0])[:7], str(common[-1])[:7], len(common))

    ndvi   = ds_ndvi["NDVI"].sel(time=common).values.astype(np.float32)
    lst    = ds_lst["LST"].sel(time=common).values.astype(np.float32)
    precip = ds_chirps["precipitation"].sel(time=common).values.astype(np.float32)
    spi3   = ds_spi["SPI3"].sel(time=common).values.astype(np.float32)

    ds_ndvi.close(); ds_lst.close(); ds_chirps.close(); ds_spi.close()

    log.info("    NDVI   : %s", ndvi.shape)
    log.info("    LST    : %s", lst.shape)
    log.info("    Precip : %s", precip.shape)
    log.info("    SPI-3  : %s", spi3.shape)

    return ndvi, lst, precip, spi3, common


def build_tensors_v3(zone):
    """
    Sprint 3 — tenseurs 4D avec 3 features [NDVI, Précip, LST], 2000-2024.
    Split conserve les mêmes bornes val/test que v1 pour comparabilité directe.
    Sorties dans data/processed/splits_v3/{zone}/.
    """
    log.info("=== Construction tenseurs v3 — %s ===", zone.upper())

    zone_dir = SPLITS_V3_DIR / zone
    zone_dir.mkdir(parents=True, exist_ok=True)

    if (zone_dir / "X_train.pt").exists():
        log.info("  Déjà construit — ignoré : %s", zone)
        return

    ndvi, lst, precip, spi3, times = load_data_v3(zone)

    log.info("  Split : train≤%d | val=%d | test>%d", TRAIN_END, VAL_END, VAL_END)
    ndvi_sp   = split_data(ndvi,   times, "NDVI")
    lst_sp    = split_data(lst,    times, "LST")
    precip_sp = split_data(precip, times, "Precip")
    spi3_sp   = split_data(spi3,   times, "SPI-3")

    log.info("  Normalisation MinMax (bornes sur train set uniquement)...")
    ndvi_tr, ndvi_va, ndvi_te, ndvi_min, ndvi_max = normalize_minmax(
        ndvi_sp["train"],   ndvi_sp["val"],   ndvi_sp["test"]
    )
    precip_tr, precip_va, precip_te, precip_min, precip_max = normalize_minmax(
        precip_sp["train"], precip_sp["val"], precip_sp["test"]
    )
    lst_tr, lst_va, lst_te, lst_min, lst_max = normalize_minmax(
        lst_sp["train"],    lst_sp["val"],    lst_sp["test"]
    )

    log.info("    NDVI   : [%.3f, %.3f]", ndvi_min, ndvi_max)
    log.info("    Precip : [%.3f, %.3f] mm/mois", precip_min, precip_max)
    log.info("    LST    : [%.3f, %.3f] °C", lst_min, lst_max)

    # X : (T, H, W, 3) — [NDVI, Précip, LST]
    X_train = np.stack([ndvi_tr, precip_tr, lst_tr], axis=-1)
    X_val   = np.stack([ndvi_va, precip_va, lst_va], axis=-1)
    X_test  = np.stack([ndvi_te, precip_te, lst_te], axis=-1)

    # y : SPI-3 v2 non normalisé
    y_train = spi3_sp["train"]
    y_val   = spi3_sp["val"]
    y_test  = spi3_sp["test"]

    log.info("  Shapes finaux :")
    log.info("    X_train : %s | y_train : %s", X_train.shape, y_train.shape)
    log.info("    X_val   : %s | y_val   : %s", X_val.shape,   y_val.shape)
    log.info("    X_test  : %s | y_test  : %s", X_test.shape,  y_test.shape)

    torch.save(torch.from_numpy(X_train), zone_dir / "X_train.pt")
    torch.save(torch.from_numpy(X_val),   zone_dir / "X_val.pt")
    torch.save(torch.from_numpy(X_test),  zone_dir / "X_test.pt")
    torch.save(torch.from_numpy(y_train), zone_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_val),   zone_dir / "y_val.pt")
    torch.save(torch.from_numpy(y_test),  zone_dir / "y_test.pt")

    np.savez(
        zone_dir / "norm_params.npz",
        ndvi_min=ndvi_min, ndvi_max=ndvi_max,
        precip_min=precip_min, precip_max=precip_max,
        lst_min=lst_min, lst_max=lst_max,
    )

    with open(zone_dir / "split_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone  : {zone}\n")
        f.write(f"Sprint: 3 — extension temporelle 2000-2024 + LST\n")
        f.write(f"Split : train≤{TRAIN_END} | val={VAL_END} | test>{VAL_END}\n")
        f.write(f"X_train : {X_train.shape} — [NDVI, Précip, LST] normalisés\n")
        f.write(f"X_val   : {X_val.shape}\n")
        f.write(f"X_test  : {X_test.shape}\n")
        f.write(f"y_train : {y_train.shape} — SPI-3 v2 non normalisé\n")
        f.write(f"y_val   : {y_val.shape}\n")
        f.write(f"y_test  : {y_test.shape}\n")
        f.write(f"Normalisation NDVI   : min={ndvi_min:.4f} | max={ndvi_max:.4f}\n")
        f.write(f"Normalisation Precip : min={precip_min:.4f} | max={precip_max:.4f} mm/mois\n")
        f.write(f"Normalisation LST    : min={lst_min:.4f} | max={lst_max:.4f} °C\n")
        f.write(f"Features : [0]=NDVI | [1]=Précipitations | [2]=LST\n")
        f.write(f"Cible    : SPI-3 v2 (non normalisé)\n")

    log.info("  ✓ Tenseurs v3 sauvegardés dans : %s", zone_dir)
    total_mb = sum(f.stat().st_size for f in zone_dir.glob("*.pt")) / 1e6
    log.info("  Taille totale fichiers .pt : %.1f MB", total_mb)


# =====================================================================
# VÉRIFICATION FINALE
# =====================================================================

def verify_tensors(zone):
    """Vérifie la cohérence des tenseurs produits."""
    log.info("--- Vérification tenseurs — %s ---", zone.upper())
    zone_dir = SPLITS_DIR / zone

    for split in ["train", "val", "test"]:
        X = torch.load(zone_dir / f"X_{split}.pt", weights_only=True)
        y = torch.load(zone_dir / f"y_{split}.pt", weights_only=True)

        log.info("  %s : X=%s [%.3f, %.3f] | y=%s [%.3f, %.3f]",
                 split,
                 tuple(X.shape),
                 float(X.min()), float(X.max()),
                 tuple(y.shape),
                 float(y.min()), float(y.max()))

        # Vérifications critiques
        assert X.shape[-1] == 2, "X doit avoir 2 features"
        assert X.shape[0]  == y.shape[0], "T doit être identique pour X et y"
        assert float(X.min()) >= -0.01, "X normalisé doit être >= 0"
        assert float(X.max()) <= 1.01,  "X normalisé doit être <= 1"

    log.info("  ✓ Toutes les vérifications passées")


def verify_tensors_v3(zone):
    """Vérifie les tenseurs Sprint 3 (3 features)."""
    log.info("--- Vérification tenseurs v3 — %s ---", zone.upper())
    zone_dir = SPLITS_V3_DIR / zone

    for split in ["train", "val", "test"]:
        X = torch.load(zone_dir / f"X_{split}.pt", weights_only=True)
        y = torch.load(zone_dir / f"y_{split}.pt", weights_only=True)

        log.info("  %s : X=%s [%.3f, %.3f] | y=%s [%.3f, %.3f]",
                 split,
                 tuple(X.shape),
                 float(X.min()), float(X.max()),
                 tuple(y.shape),
                 float(y.min()), float(y.max()))

        assert X.shape[-1] == 3, "X doit avoir 3 features (NDVI, Précip, LST)"
        assert X.shape[0]  == y.shape[0], "T doit être identique pour X et y"
        assert float(X.min()) >= -0.01, "X normalisé doit être >= 0"
        assert float(X.max()) <= 1.01,  "X normalisé doit être <= 1"

    log.info("  ✓ Toutes les vérifications v3 passées")


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("============================================")
    log.info(" PHASE 2 — Script 06 : Construction tenseurs")
    log.info(" Split : train≤%d | val=%d | test>%d",
             TRAIN_END, VAL_END, VAL_END)
    log.info(" v1 : NDVI + Précip (2 features, splits/)")
    log.info(" v3 : NDVI + Précip + LST (3 features, splits_v3/)")
    log.info("============================================")

    for zone in ["haute_guinee", "moyenne_guinee"]:
        # v1 (existant — skip si déjà construit)
        build_tensors(zone)
        verify_tensors(zone)

        # Sprint 3
        build_tensors_v3(zone)
        verify_tensors_v3(zone)
        log.info("")

    log.info("============================================")
    log.info(" Script 06 terminé — Phase 2 COMPLÈTE")
    log.info(" Données dans : %s", SPLITS_DIR)
    log.info("============================================")
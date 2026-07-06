"""
preprocessing/13_build_tensors_v3.py
========================================
Sprint 3 — Construction des tenseurs sur la période étendue 2000-2024.

Période d'alignement commune : 2000-03 → 2024-12 (298 mois)
  (LST disponible depuis 2000-03 — facteur limitant)

Split temporel (acté) :
  Train : 2000-03 → 2021-12 — 262 mois
  Val   : 2022-01 → 2022-12 — 12 mois
  Test  : 2023-01 → 2024-12 — 24 mois

Config 1 — [NDVI, Précip] :
  X shape : (298, H, W, 2) → splits_v3/config1/{zone}/

Config 2 — [NDVI, Précip, LST, VHI_v3] :
  X shape : (298, H, W, 4) → splits_v3/config2/{zone}/

Règles :
  - MinMax calculé sur train uniquement, appliqué sur val/test
  - NaN résiduels LST (~1.7%) : remplacés par la moyenne temporelle du pixel
  - Bornes sauvegardées dans norm_params.npz

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 3 — Juillet 2026
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

BASE_DIR     = Path(__file__).parent.parent
PROC_DIR     = BASE_DIR / "data" / "processed"
INDICES_DIR  = PROC_DIR / "indices"
SPLITS_V3    = PROC_DIR / "splits_v3"
SPLITS_V3.mkdir(parents=True, exist_ok=True)

# Split — identique pour les deux configs
TRAIN_END_YEAR = 2021   # inclus
VAL_YEAR       = 2022   # inclus
TEST_START_YEAR = 2023  # inclus

ZONES = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# CHARGEMENT ET ALIGNEMENT
# =====================================================================

def fill_nan_interp(arr, label=""):
    """
    Remplace les NaN par interpolation linéaire temporelle pixel par pixel,
    sur la série complète AVANT le split.
    Si un pixel est entièrement NaN, remplacé par 0.5 (valeur neutre).
    arr : np.ndarray (T, H, W)
    """
    nan_count = int(np.isnan(arr).sum())
    if nan_count == 0:
        return arr

    T = arr.shape[0]
    t_all = np.arange(T)
    n_full_nan = 0

    for h in range(arr.shape[1]):
        for w in range(arr.shape[2]):
            pixel = arr[:, h, w]
            if not np.isnan(pixel).any():
                continue
            if np.isnan(pixel).all():
                arr[:, h, w] = 0.5
                n_full_nan += 1
            else:
                valid_idx = np.where(~np.isnan(pixel))[0]
                arr[:, h, w] = np.interp(t_all, valid_idx, pixel[valid_idx])

    remaining = int(np.isnan(arr).sum())
    log.info("    %s NaN : %d → %d restants (%d pixels entièrement NaN → 0.5)",
             label, nan_count, remaining, n_full_nan)
    return arr


def load_all(zone):
    """
    Charge NDVI v3, LST v3, CHIRPS et SPI-3 v3.
    Aligne sur la période commune (déterminée par LST : 2000-03 → 2024-12).
    """
    log.info("  Chargement données — %s", zone)

    ds_ndvi   = xr.open_dataset(PROC_DIR    / f"ndvi_v3_{zone}.nc")
    ds_lst    = xr.open_dataset(PROC_DIR    / f"lst_v3_{zone}.nc")
    ds_chirps = xr.open_dataset(PROC_DIR    / f"chirps_{zone}.nc")
    ds_spi    = xr.open_dataset(INDICES_DIR / f"spi3_v3_{zone}.nc")
    ds_vhi    = xr.open_dataset(INDICES_DIR / f"vhi_v3_{zone}.nc")

    t_ndvi   = pd.to_datetime(ds_ndvi["NDVI"].time.values)
    t_lst    = pd.to_datetime(ds_lst["LST"].time.values)
    t_chirps = pd.to_datetime(ds_chirps["precipitation"].time.values)
    t_spi    = pd.to_datetime(ds_spi["SPI3_v3"].time.values)
    t_vhi    = pd.to_datetime(ds_vhi["VHI_v3"].time.values)

    common = t_ndvi.intersection(t_lst).intersection(t_chirps) \
                   .intersection(t_spi).intersection(t_vhi)

    log.info("  Période commune : %s → %s (%d mois)",
             str(common[0])[:7], str(common[-1])[:7], len(common))

    ndvi   = ds_ndvi["NDVI"].sel(time=common).values.astype(np.float32)
    lst    = ds_lst["LST"].sel(time=common).values.astype(np.float32)
    precip = ds_chirps["precipitation"].sel(time=common).values.astype(np.float32)
    spi3   = ds_spi["SPI3_v3"].sel(time=common).values.astype(np.float32)
    vhi    = ds_vhi["VHI_v3"].sel(time=common).values.astype(np.float32)

    ds_ndvi.close(); ds_lst.close(); ds_chirps.close(); ds_spi.close(); ds_vhi.close()

    # Interpolation linéaire sur la série complète AVANT le split
    # LST : ~1.7% NaN résiduels (pixels nuageux non couverts par la médiane mensuelle)
    # VHI : hérite des NaN de LST via TCI — traité aussi
    log.info("  Remplissage NaN par interpolation linéaire (série complète)...")
    lst = fill_nan_interp(lst, "LST")
    vhi = fill_nan_interp(vhi, "VHI")

    log.info("  Après remplissage :")
    log.info("    NDVI   : %s | NaN=%.2f%%", ndvi.shape, np.isnan(ndvi).mean()*100)
    log.info("    Précip : %s | NaN=%.2f%%", precip.shape, np.isnan(precip).mean()*100)
    log.info("    LST    : %s | NaN=%.2f%%", lst.shape, np.isnan(lst).mean()*100)
    log.info("    VHI    : %s | NaN=%.2f%%", vhi.shape, np.isnan(vhi).mean()*100)
    log.info("    SPI-3  : %s | NaN=%.2f%%", spi3.shape, np.isnan(spi3).mean()*100)

    return ndvi, precip, lst, vhi, spi3, common


# =====================================================================
# SPLIT ET NORMALISATION
# =====================================================================

def split_data(arr, times, label=""):
    mask_train = times.year <= TRAIN_END_YEAR
    mask_val   = times.year == VAL_YEAR
    mask_test  = times.year >= TEST_START_YEAR
    log.info("    %s — train=%d | val=%d | test=%d mois",
             label,
             mask_train.sum(), mask_val.sum(), mask_test.sum())
    return arr[mask_train], arr[mask_val], arr[mask_test]


def normalize_minmax(tr, va, te):
    """MinMax sur train, appliqué sur val/test. Retourne (tr_n, va_n, te_n, vmin, vmax)."""
    vmin = float(np.nanmin(tr))
    vmax = float(np.nanmax(tr))
    rng  = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0
    return (
        np.clip((tr - vmin) / rng, 0.0, 1.0),
        np.clip((va - vmin) / rng, 0.0, 1.0),
        np.clip((te - vmin) / rng, 0.0, 1.0),
        vmin, vmax,
    )


# =====================================================================
# CONSTRUCTION TENSEURS PAR CONFIG
# =====================================================================

def build_config(zone, config_name, feature_arrays, feature_names, spi3):
    """
    Construit et sauvegarde les tenseurs pour une config et une zone.
    feature_arrays : list[np.ndarray (T,H,W)] — features dans l'ordre
    feature_names  : list[str] — noms pour les logs et norm_params
    spi3           : np.ndarray (T,H,W) — variable cible
    """
    log.info("  --- %s / %s (%d features) ---",
             zone, config_name, len(feature_arrays))

    zone_dir = SPLITS_V3 / config_name / zone
    zone_dir.mkdir(parents=True, exist_ok=True)

    if (zone_dir / "X_train.pt").exists():
        log.info("  Déjà construit — ignoré : %s/%s", config_name, zone)
        return

    # Charger les times depuis SPI-3 v3 pour faire le split
    ds_spi = xr.open_dataset(INDICES_DIR / f"spi3_v3_{zone}.nc")
    times  = pd.to_datetime(ds_spi["SPI3_v3"].time.values)
    ds_spi.close()

    # Split
    spi_tr, spi_va, spi_te = split_data(spi3, times, "SPI-3")
    feat_splits = [split_data(arr, times, name)
                   for arr, name in zip(feature_arrays, feature_names)]

    # Normalisation MinMax (bornes sur train uniquement)
    norm_params = {}
    normed = []
    for (tr, va, te), name in zip(feat_splits, feature_names):
        tr_n, va_n, te_n, vmin, vmax = normalize_minmax(tr, va, te)
        normed.append((tr_n, va_n, te_n))
        norm_params[f"{name}_min"] = vmin
        norm_params[f"{name}_max"] = vmax
        log.info("    %-8s : [%.3f, %.3f] → normalisé sur train",
                 name, vmin, vmax)

    # Assemblage X : (T, H, W, F)
    X_train = np.stack([tr for tr, _, _  in normed], axis=-1)
    X_val   = np.stack([va for _, va, _  in normed], axis=-1)
    X_test  = np.stack([te for _, _, te  in normed], axis=-1)

    log.info("  Shapes : X_train=%s | X_val=%s | X_test=%s",
             X_train.shape, X_val.shape, X_test.shape)
    log.info("           y_train=%s | y_val=%s  | y_test=%s",
             spi_tr.shape, spi_va.shape, spi_te.shape)

    # Vérification NaN critique avant sauvegarde
    nan_X_train = int(np.isnan(X_train).sum())
    nan_X_val   = int(np.isnan(X_val).sum())
    nan_X_test  = int(np.isnan(X_test).sum())
    if nan_X_train > 0 or nan_X_val > 0 or nan_X_test > 0:
        log.error("  ✗ NaN résiduels détectés — X_train=%d | X_val=%d | X_test=%d",
                  nan_X_train, nan_X_val, nan_X_test)
        raise ValueError(f"NaN résiduels dans les tenseurs {config_name}/{zone} — "
                         "vérifiez fill_nan_interp()")
    log.info("  ✓ np.isnan(X_train).sum() == 0 — aucun NaN résiduel")

    torch.save(torch.from_numpy(X_train), zone_dir / "X_train.pt")
    torch.save(torch.from_numpy(X_val),   zone_dir / "X_val.pt")
    torch.save(torch.from_numpy(X_test),  zone_dir / "X_test.pt")
    torch.save(torch.from_numpy(spi_tr),  zone_dir / "y_train.pt")
    torch.save(torch.from_numpy(spi_va),  zone_dir / "y_val.pt")
    torch.save(torch.from_numpy(spi_te),  zone_dir / "y_test.pt")

    np.savez(zone_dir / "norm_params.npz", **norm_params)

    with open(zone_dir / "split_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone    : {zone}\n")
        f.write(f"Config  : {config_name}\n")
        f.write(f"Sprint  : 3 — extension 2000-2024 + LST\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Split   : train≤{TRAIN_END_YEAR} | val={VAL_YEAR} "
                f"| test≥{TEST_START_YEAR}\n")
        f.write(f"X_train : {X_train.shape}\n")
        f.write(f"X_val   : {X_val.shape}\n")
        f.write(f"X_test  : {X_test.shape}\n")
        f.write(f"y_train : {spi_tr.shape} — SPI3_v3 non normalisé\n")
        for name, val in norm_params.items():
            f.write(f"  {name} = {val:.6f}\n")

    log.info("  ✓ %s/%s sauvegardé", config_name, zone)


# =====================================================================
# VÉRIFICATION
# =====================================================================

def verify(zone, config_name, n_features):
    log.info("--- Vérification %s/%s ---", config_name, zone)
    zone_dir = SPLITS_V3 / config_name / zone
    all_ok = True

    for split in ["train", "val", "test"]:
        X = torch.load(zone_dir / f"X_{split}.pt", weights_only=True)
        y = torch.load(zone_dir / f"y_{split}.pt", weights_only=True)

        ok_feat = X.shape[-1] == n_features
        ok_time = X.shape[0]  == y.shape[0]
        ok_min  = float(X.min()) >= -0.01
        ok_max  = float(X.max()) <= 1.01

        status = "✓" if all([ok_feat, ok_time, ok_min, ok_max]) else "✗"
        log.info("  %s %s : X=%s [%.3f, %.3f] | y=%s [%.3f, %.3f]",
                 status, split,
                 tuple(X.shape), float(X.min()), float(X.max()),
                 tuple(y.shape), float(y.min()), float(y.max()))

        if not ok_feat:
            log.error("    ✗ X.shape[-1]=%d ≠ %d features attendus", X.shape[-1], n_features)
            all_ok = False
        if not ok_time:
            log.error("    ✗ X.shape[0] (%d) ≠ y.shape[0] (%d)", X.shape[0], y.shape[0])
            all_ok = False
        if not ok_min or not ok_max:
            log.error("    ✗ X non dans [0, 1]")
            all_ok = False

    if all_ok:
        log.info("  ✓ Toutes les assertions passées")
    return all_ok


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" SPRINT 3 — Script 13 : Construction tenseurs v3")
    log.info(" Config 1 : [NDVI, Précip]          → 2 features")
    log.info(" Config 2 : [NDVI, Précip, LST, VHI] → 4 features")
    log.info(" Split : train≤%d | val=%d | test≥%d",
             TRAIN_END_YEAR, VAL_YEAR, TEST_START_YEAR)
    log.info("=" * 55)

    for zone in ZONES:
        log.info(">>> Zone : %s", zone.upper())

        ndvi, precip, lst, vhi, spi3, times = load_all(zone)

        # Config 1 — [NDVI, Précip]
        build_config(
            zone, "config1",
            feature_arrays=[ndvi, precip],
            feature_names=["NDVI", "Precip"],
            spi3=spi3,
        )

        # Config 2 — [NDVI, Précip, LST, VHI_v3]
        build_config(
            zone, "config2",
            feature_arrays=[ndvi, precip, lst, vhi],
            feature_names=["NDVI", "Precip", "LST", "VHI"],
            spi3=spi3,
        )

        log.info("")

    log.info("=" * 55)
    log.info(" Vérifications finales")
    log.info("=" * 55)

    for zone in ZONES:
        for cfg, nf in [("config1", 2), ("config2", 4)]:
            verify(zone, cfg, nf)
        log.info("")

    log.info("=" * 55)
    log.info(" Script 13 terminé. Tenseurs dans : %s", SPLITS_V3)
    log.info("=" * 55)

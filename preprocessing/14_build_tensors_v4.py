"""
preprocessing/14_build_tensors_v4.py
========================================
Sprint 4 — Tenseurs avec lead time +1 mois.

Différence clé avec script 13 :
  v3 : X[t-SEQ_LEN:t] → y[t]       (estimation temps réel)
  v4 : X[t-SEQ_LEN:t] → y[t+1]     (prévision 1 mois)

Les tenseurs sont pré-séquencés (N, SEQ_LEN, H, W, F) au lieu de (T, H, W, F).
Le split se fait sur input_end_times (dernier mois du vecteur d'entrée),
pas sur les temps cibles.

Période d'alignement commune : 2000-03 → 2024-12 (T=298 mois)
SEQ_LEN = 3 (inchangé)
Boucle : range(3, 297) → N=294 séquences au total

Split par input_end_times.year :
  Train : input_end ≤ 2021-12  →  N=260  (t = 3..262)
  Val   : input_end ∈ 2022     →  N=12   (t = 263..274)
  Test  : input_end ≥ 2023-01  →  N=22   (t = 275..296)

Config 1 — [NDVI, Précip] :
  X shape : (N, 3, H, W, 2) → splits_v4/config1/{zone}/

Config 2 — [NDVI, Précip, LST, VHI_v3] :
  X shape : (N, 3, H, W, 4) → splits_v4/config2/{zone}/

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 4 — Juillet 2026
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
SPLITS_V4    = PROC_DIR / "splits_v4"

SEQ_LEN        = 3     # identique à toutes les versions précédentes
TRAIN_END_YEAR = 2021
VAL_YEAR       = 2022
TEST_START_YEAR = 2023

ZONES = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# UTILITAIRES (identiques à 13_build_tensors_v3.py)
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
    Aligne sur la période commune 2000-03 → 2024-12 (T=298 mois).
    Identique à 13_build_tensors_v3.py.
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
# CONSTRUCTION DES SÉQUENCES V4 (lead time +1 mois)
# =====================================================================

def build_sequences_v4(feature_arrays, spi3, times):
    """
    Construit les séquences avec lead time +1 mois.

    Pour chaque t dans range(SEQ_LEN, T-1) :
      X[i] = data_array[t-SEQ_LEN : t]   # (SEQ_LEN, H, W, F)
              features aux mois t-3, t-2, t-1
      y[i] = spi3[t+1]                   # (H, W) — mois suivant

    Le split se fait sur input_end_times[i] = times[t-1]
    (dernier mois effectivement observé dans la fenêtre d'entrée).

    Retourne :
      X_all            : (N, SEQ_LEN, H, W, F)
      y_all            : (N, H, W)
      input_end_times  : DatetimeIndex de longueur N
      target_times     : DatetimeIndex de longueur N
    """
    T = spi3.shape[0]
    F = len(feature_arrays)

    data = np.stack(feature_arrays, axis=-1)  # (T, H, W, F)

    sequences_X    = []
    sequences_y    = []
    input_end_times = []   # dernier mois d'entrée (pour le split)
    target_times   = []    # mois cible (pour logging uniquement)

    # --------------------------------------------------------
    # BOUCLE DE CONSTRUCTION — v4 lead time +1 mois
    # --------------------------------------------------------
    for t in range(SEQ_LEN, T - 1):  # T-1 car besoin de spi3[t+1]
        sequences_X.append(data[t - SEQ_LEN : t])     # (SEQ_LEN, H, W, F)
        sequences_y.append(spi3[t + 1])               # (H, W) — mois suivant
        input_end_times.append(times[t - 1])          # dernier mois d'entrée
        target_times.append(times[t + 1])             # mois cible
    # --------------------------------------------------------

    X_all           = np.stack(sequences_X, axis=0)   # (N, SEQ_LEN, H, W, F)
    y_all           = np.stack(sequences_y, axis=0)   # (N, H, W)
    input_end_times = pd.DatetimeIndex(input_end_times)
    target_times    = pd.DatetimeIndex(target_times)

    log.info("  Séquences v4 : N=%d (T=%d, SEQ_LEN=%d, range(SEQ_LEN, T-1))",
             len(sequences_X), T, SEQ_LEN)
    log.info("  input_end : %s → %s",
             str(input_end_times[0])[:7], str(input_end_times[-1])[:7])
    log.info("  target    : %s → %s",
             str(target_times[0])[:7], str(target_times[-1])[:7])
    log.info("  X_all : %s | y_all : %s", X_all.shape, y_all.shape)

    return X_all, y_all, input_end_times, target_times


# =====================================================================
# CONSTRUCTION TENSEURS PAR CONFIG
# =====================================================================

def build_config_v4(zone, config_name, feature_arrays, feature_names, spi3, times):
    """
    Construit, normalise et sauvegarde les tenseurs v4 pour une config et une zone.

    Différences par rapport à build_config() du script 13 :
      - Séquences pré-construites : X (N, SEQ_LEN, H, W, F), y (N, H, W)
      - Split sur input_end_times (pas sur les temps bruts)
      - Normalisation sur X_train[..., f] (pas sur le split temporel brut)
    """
    log.info("  --- %s / %s (%d features) ---",
             zone, config_name, len(feature_arrays))

    zone_dir = SPLITS_V4 / config_name / zone
    zone_dir.mkdir(parents=True, exist_ok=True)

    if (zone_dir / "X_train.pt").exists():
        log.info("  Déjà construit — ignoré : %s/%s", config_name, zone)
        return

    # Construction des séquences
    X_all, y_all, input_end_times, target_times = build_sequences_v4(
        feature_arrays, spi3, times
    )

    # Split sur input_end_times.year (dernier mois d'entrée observé)
    mask_train = input_end_times.year <= TRAIN_END_YEAR
    mask_val   = input_end_times.year == VAL_YEAR
    mask_test  = input_end_times.year >= TEST_START_YEAR

    # Boolean fancy indexing → copies indépendantes
    X_tr = X_all[mask_train].copy()
    X_va = X_all[mask_val].copy()
    X_te = X_all[mask_test].copy()
    y_tr = y_all[mask_train].copy()
    y_va = y_all[mask_val].copy()
    y_te = y_all[mask_test].copy()

    iet = input_end_times
    tt  = target_times
    log.info("  Train : input_end %s → %s | target %s → %s | N=%d",
             str(iet[mask_train][0])[:7], str(iet[mask_train][-1])[:7],
             str(tt[mask_train][0])[:7],  str(tt[mask_train][-1])[:7],
             int(mask_train.sum()))
    log.info("  Val   : input_end %s → %s | target %s → %s | N=%d",
             str(iet[mask_val][0])[:7], str(iet[mask_val][-1])[:7],
             str(tt[mask_val][0])[:7],  str(tt[mask_val][-1])[:7],
             int(mask_val.sum()))
    log.info("  Test  : input_end %s → %s | target %s → %s | N=%d",
             str(iet[mask_test][0])[:7], str(iet[mask_test][-1])[:7],
             str(tt[mask_test][0])[:7],  str(tt[mask_test][-1])[:7],
             int(mask_test.sum()))

    # Normalisation MinMax par feature — bornes calculées sur X_train uniquement
    # X_tr.shape = (N_train, SEQ_LEN, H, W, F)
    norm_params = {}
    for f_idx, name in enumerate(feature_names):
        tr_f = X_tr[..., f_idx]    # (N_train, SEQ_LEN, H, W)
        vmin = float(np.nanmin(tr_f))
        vmax = float(np.nanmax(tr_f))
        rng  = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0
        X_tr[..., f_idx] = np.clip((X_tr[..., f_idx] - vmin) / rng, 0.0, 1.0)
        X_va[..., f_idx] = np.clip((X_va[..., f_idx] - vmin) / rng, 0.0, 1.0)
        X_te[..., f_idx] = np.clip((X_te[..., f_idx] - vmin) / rng, 0.0, 1.0)
        norm_params[f"{name}_min"] = vmin
        norm_params[f"{name}_max"] = vmax
        log.info("    %-8s : [%.3f, %.3f] → normalisé sur X_train", name, vmin, vmax)

    log.info("  Shapes : X_train=%s | X_val=%s | X_test=%s",
             X_tr.shape, X_va.shape, X_te.shape)
    log.info("           y_train=%s | y_val=%s  | y_test=%s",
             y_tr.shape, y_va.shape, y_te.shape)

    # Vérification NaN critique avant sauvegarde
    for arr_name, arr in [("X_train", X_tr), ("X_val", X_va), ("X_test", X_te)]:
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            raise ValueError(
                f"NaN résiduels dans {config_name}/{zone}/{arr_name} : {n_nan} — "
                "vérifiez fill_nan_interp()"
            )
    log.info("  ✓ np.isnan(X_*).sum() == 0 — aucun NaN résiduel")

    # Vérification plage y (SPI-3 non normalisé)
    y_min_tr = float(np.nanmin(y_tr))
    y_max_tr = float(np.nanmax(y_tr))
    log.info("  y_train SPI-3 : min=%.3f | max=%.3f (attendu ≈ [-3.09, +3.09])",
             y_min_tr, y_max_tr)

    # Sauvegarde
    torch.save(torch.from_numpy(X_tr), zone_dir / "X_train.pt")
    torch.save(torch.from_numpy(X_va), zone_dir / "X_val.pt")
    torch.save(torch.from_numpy(X_te), zone_dir / "X_test.pt")
    torch.save(torch.from_numpy(y_tr), zone_dir / "y_train.pt")
    torch.save(torch.from_numpy(y_va), zone_dir / "y_val.pt")
    torch.save(torch.from_numpy(y_te), zone_dir / "y_test.pt")
    np.savez(zone_dir / "norm_params.npz", **norm_params)

    with open(zone_dir / "split_info.txt", "w", encoding="utf-8") as f:
        f.write(f"Zone      : {zone}\n")
        f.write(f"Config    : {config_name}\n")
        f.write(f"Sprint    : 4 — lead time +1 mois\n")
        f.write(f"Features  : {feature_names}\n")
        f.write(f"Lead time : +1 mois (y = SPI-3 du mois suivant le dernier input)\n")
        f.write(f"SEQ_LEN   : {SEQ_LEN}\n")
        f.write(f"Split     : input_end_year ≤ {TRAIN_END_YEAR} | = {VAL_YEAR} | ≥ {TEST_START_YEAR}\n")
        f.write(f"X_train   : {X_tr.shape}  — (N, SEQ_LEN, H, W, F)\n")
        f.write(f"X_val     : {X_va.shape}\n")
        f.write(f"X_test    : {X_te.shape}\n")
        f.write(f"y_train   : {y_tr.shape}  — SPI3_v3 non normalisé\n")
        f.write(f"y_train SPI-3 range : [{y_min_tr:.3f}, {y_max_tr:.3f}]\n")
        f.write(f"Train input_end : {str(iet[mask_train][0])[:7]} → {str(iet[mask_train][-1])[:7]}\n")
        f.write(f"Val   input_end : {str(iet[mask_val][0])[:7]} → {str(iet[mask_val][-1])[:7]}\n")
        f.write(f"Test  input_end : {str(iet[mask_test][0])[:7]} → {str(iet[mask_test][-1])[:7]}\n")
        f.write(f"Train target    : {str(tt[mask_train][0])[:7]} → {str(tt[mask_train][-1])[:7]}\n")
        f.write(f"Val   target    : {str(tt[mask_val][0])[:7]} → {str(tt[mask_val][-1])[:7]}\n")
        f.write(f"Test  target    : {str(tt[mask_test][0])[:7]} → {str(tt[mask_test][-1])[:7]}\n")
        for name, val in norm_params.items():
            f.write(f"  {name} = {val:.6f}\n")

    log.info("  ✓ %s/%s sauvegardé", config_name, zone)


# =====================================================================
# VÉRIFICATION
# =====================================================================

def verify(zone, config_name, n_features):
    log.info("--- Vérification %s/%s ---", config_name, zone)
    zone_dir = SPLITS_V4 / config_name / zone
    all_ok = True

    for split in ["train", "val", "test"]:
        X = torch.load(zone_dir / f"X_{split}.pt", weights_only=True)
        y = torch.load(zone_dir / f"y_{split}.pt", weights_only=True)

        ok_ndim   = X.ndim == 5              # (N, SEQ_LEN, H, W, F)
        ok_seqlen = X.shape[1] == SEQ_LEN
        ok_feat   = X.shape[-1] == n_features
        ok_time   = X.shape[0] == y.shape[0]
        ok_min    = float(X.min()) >= -0.01
        ok_max    = float(X.max()) <= 1.01

        status = "✓" if all([ok_ndim, ok_seqlen, ok_feat, ok_time, ok_min, ok_max]) else "✗"
        log.info("  %s %s : X=%s (5D) [%.3f, %.3f] | y=%s [%.3f, %.3f]",
                 status, split,
                 tuple(X.shape), float(X.min()), float(X.max()),
                 tuple(y.shape), float(y.min()), float(y.max()))

        if not ok_ndim:
            log.error("    ERREUR ndim : attendu 5 (N,SEQ,H,W,F), obtenu %d", X.ndim)
        if not ok_seqlen:
            log.error("    ERREUR SEQ_LEN : attendu %d, obtenu %d", SEQ_LEN, X.shape[1])
        if not ok_feat:
            log.error("    ERREUR features : attendu %d, obtenu %d", n_features, X.shape[-1])
        if not all([ok_ndim, ok_seqlen, ok_feat, ok_time, ok_min, ok_max]):
            all_ok = False

    if all_ok:
        log.info("  Toutes les assertions passées pour %s/%s", config_name, zone)


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 60)
    log.info(" SPRINT 4 — Script 14 : Tenseurs v4 (lead time +1 mois)")
    log.info(" X[t-SEQ_LEN:t] → y[t+1]")
    log.info(" Tenseurs pré-séquencés : (N, SEQ_LEN, H, W, F)")
    log.info(" Split sur input_end_times (dernier mois d'entrée)")
    log.info("=" * 60)

    SPLITS_V4.mkdir(parents=True, exist_ok=True)

    for zone in ZONES:
        log.info(">>> Zone : %s", zone.upper())
        ndvi, precip, lst, vhi, spi3, common = load_all(zone)

        build_config_v4(
            zone, "config1",
            feature_arrays=[ndvi, precip],
            feature_names=["NDVI", "Précip"],
            spi3=spi3, times=common,
        )

        build_config_v4(
            zone, "config2",
            feature_arrays=[ndvi, precip, lst, vhi],
            feature_names=["NDVI", "Précip", "LST", "VHI"],
            spi3=spi3, times=common,
        )

        log.info("")

    log.info("=" * 60)
    log.info(" Vérification finale")
    log.info("=" * 60)
    for zone in ZONES:
        verify(zone, "config1", n_features=2)
        verify(zone, "config2", n_features=4)

    log.info("=" * 60)
    log.info(" Script 14 terminé. Tenseurs dans : %s", SPLITS_V4)
    log.info("=" * 60)

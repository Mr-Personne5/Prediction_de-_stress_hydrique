"""
preprocessing/12_compute_indices_v3.py
========================================
Sprint 3 — Calcul des indices sur la période étendue 2000-2024.

SPI-3 v3 :
  Source  : chirps_{zone}.nc (1981-2024, déjà disponible)
  Calibration : 1981-2014 (identique à v1)
  Extraction : 2000-03 → 2024-12 (aligné sur LST)
  Sortie  : data/processed/indices/spi3_v3_{zone}.nc | variable SPI3_v3

VHI v3 (avec vrai TCI depuis LST v3) :
  Source NDVI : ndvi_v3_{zone}.nc
  Source LST  : lst_v3_{zone}.nc
  Période commune : 2000-03 → 2024-12
  VCI = (NDVI - NDVImin_mois) / (NDVImax_mois - NDVImin_mois)
  TCI = (LSTmax_mois - LST)   / (LSTmax_mois - LSTmin_mois)
  VHI = 0.5 × VCI + 0.5 × TCI
  Sortie : data/processed/indices/vhi_v3_{zone}.nc | variable VHI_v3

Corrige la limitation Sprint 1 (VHI = VCI uniquement, faute de LST).

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 3 — Juillet 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import xarray as xr
import pandas as pd
import logging
from pathlib import Path
from scipy import stats
from climate_indices import compute, indices

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

BASE_DIR    = Path(__file__).parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
INDICES_DIR = PROC_DIR / "indices"
INDICES_DIR.mkdir(parents=True, exist_ok=True)

SCALE             = 3
CALIBRATION_START = 1981
CALIBRATION_END   = 2014
STUDY_START_V3    = "2000-03-01"  # première date commune NDVI v3 ∩ LST v3
STUDY_END         = "2024-12-31"
ZONES             = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# SPI-3 v3 (2000-03 → 2024-12)
# =====================================================================

def compute_spi3_pixel(precip_series):
    """SPI-3 pour une série temporelle 1D (identique à 05_compute_indices.py)."""
    try:
        return indices.spi(
            values=precip_series.astype(float),
            scale=SCALE,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=CALIBRATION_START,
            calibration_year_initial=CALIBRATION_START,
            calibration_year_final=CALIBRATION_END,
        ).astype(np.float32)
    except Exception:
        return np.full(len(precip_series), np.nan, dtype=np.float32)


def compute_spi3_v3(zone):
    """
    Calcule SPI-3 sur la totalité de CHIRPS (1981-2024) avec la calibration
    1981-2014, puis extrait uniquement 2000-03 → 2024-12 pour aligner
    sur la disponibilité de LST.
    """
    log.info("=== SPI-3 v3 — %s ===", zone.upper())
    output = INDICES_DIR / f"spi3_v3_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ds     = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    chirps = ds["precipitation"]
    times  = pd.to_datetime(chirps.time.values)
    T, H, W = chirps.shape
    log.info("  CHIRPS shape : %s | %s → %s",
             chirps.shape, str(times[0])[:7], str(times[-1])[:7])
    log.info("  Calcul SPI-3 pixel par pixel sur %d×%d = %d pixels...", H, W, H*W)

    spi_full = np.full((T, H, W), np.nan, dtype=np.float32)
    done, total = 0, H * W
    for h in range(H):
        for w in range(W):
            spi_full[:, h, w] = compute_spi3_pixel(chirps.values[:, h, w])
            done += 1
            if done % (total // 10) == 0:
                log.info("  Progression : %d%%", int(done / total * 100))

    # Extraction : 2000-03 → 2024-12 (aligné sur LST)
    mask        = (times >= pd.Timestamp(STUDY_START_V3)) & (times <= pd.Timestamp(STUDY_END))
    spi_study   = spi_full[mask]
    times_study = times[mask]

    log.info("  Timesteps extraits : %d (%s → %s)",
             len(times_study), str(times_study[0])[:7], str(times_study[-1])[:7])
    log.info("  NaN : %.2f%%", np.isnan(spi_study).mean() * 100)
    log.info("  Min : %.3f | Max : %.3f",
             float(np.nanmin(spi_study)), float(np.nanmax(spi_study)))

    da = xr.DataArray(
        data=spi_study,
        dims=["time", "lat", "lon"],
        coords={"time": times_study,
                "lat":  chirps.lat.values,
                "lon":  chirps.lon.values},
        attrs={
            "long_name":          "Standardized Precipitation Index (3-month) — Sprint 3",
            "source":             "CHIRPS v3 — climate-indices 2.4.0",
            "scale":              "3 months",
            "distribution":       "Gamma",
            "calibration_period": f"{CALIBRATION_START}-{CALIBRATION_END}",
            "study_period":       f"{STUDY_START_V3[:7]} to {STUDY_END[:7]}",
            "units":              "dimensionless (standardized)",
            "thresholds":         "< -2.0 extreme | < -1.5 severe | < -1.0 moderate",
            "zone":               zone,
            "sprint":             "Sprint 3",
        }
    )
    ds_out = da.to_dataset(name="SPI3_v3")
    ds_out.to_netcdf(output, encoding={"SPI3_v3": {"dtype": "float32", "zlib": True}})
    ds.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


# =====================================================================
# VHI v3 — VCI + TCI réel (depuis LST v3)
# =====================================================================

def compute_vhi_v3(zone):
    """
    VHI complet avec vrai TCI (Sprint 3).
    Alignement sur la période commune NDVI v3 ∩ LST v3 = 2000-03 → 2024-12.
    Corrige la limitation Sprint 1 (VHI = VCI uniquement).
    """
    log.info("=== VHI v3 (VCI + TCI réel) — %s ===", zone.upper())
    output = INDICES_DIR / f"vhi_v3_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ndvi_path = PROC_DIR / f"ndvi_v3_{zone}.nc"
    lst_path  = PROC_DIR / f"lst_v3_{zone}.nc"
    for p in [ndvi_path, lst_path]:
        if not p.exists():
            log.error("  Fichier manquant : %s — lancez d'abord 11_reproject_align_v3.py", p.name)
            return

    ds_ndvi = xr.open_dataset(ndvi_path)
    ds_lst  = xr.open_dataset(lst_path)

    times_ndvi = pd.to_datetime(ds_ndvi["NDVI"].time.values)
    times_lst  = pd.to_datetime(ds_lst["LST"].time.values)
    common     = times_ndvi.intersection(times_lst)

    ndvi  = ds_ndvi["NDVI"].sel(time=common).values.astype(np.float32)
    lst   = ds_lst["LST"].sel(time=common).values.astype(np.float32)
    T, H, W = ndvi.shape
    log.info("  Période commune : %s → %s (%d mois)",
             str(common[0])[:7], str(common[-1])[:7], T)

    vci = np.full_like(ndvi, np.nan)
    tci = np.full_like(lst,  np.nan)

    for month in range(1, 13):
        mask = common.month == month

        # VCI
        ndvi_m    = ndvi[mask]
        ndvi_min  = np.nanmin(ndvi_m, axis=0)
        ndvi_max  = np.nanmax(ndvi_m, axis=0)
        ndvi_rng  = np.where(ndvi_max - ndvi_min < 1e-6, 1e-6, ndvi_max - ndvi_min)
        vci[mask] = np.clip((ndvi_m - ndvi_min) / ndvi_rng, 0.0, 1.0)

        # TCI — LST élevée = stress thermique → relation inverse
        lst_m     = lst[mask]
        lst_min   = np.nanmin(lst_m, axis=0)
        lst_max   = np.nanmax(lst_m, axis=0)
        lst_rng   = np.where(lst_max - lst_min < 1e-6, 1e-6, lst_max - lst_min)
        tci[mask] = np.clip((lst_max - lst_m) / lst_rng, 0.0, 1.0)

    vhi = (0.5 * vci + 0.5 * tci).astype(np.float32)

    log.info("  VCI : min=%.3f | max=%.3f | mean=%.3f",
             float(np.nanmin(vci)), float(np.nanmax(vci)), float(np.nanmean(vci)))
    log.info("  TCI : min=%.3f | max=%.3f | mean=%.3f",
             float(np.nanmin(tci)), float(np.nanmax(tci)), float(np.nanmean(tci)))
    log.info("  VHI : min=%.3f | max=%.3f | mean=%.3f",
             float(np.nanmin(vhi)), float(np.nanmax(vhi)), float(np.nanmean(vhi)))

    da = xr.DataArray(
        data=vhi,
        dims=["time", "lat", "lon"],
        coords={"time": common,
                "lat":  ds_ndvi["NDVI"].lat.values,
                "lon":  ds_ndvi["NDVI"].lon.values},
        attrs={
            "long_name":     "Vegetation Health Index (VCI + TCI réel) — Sprint 3",
            "formula":       "VHI = 0.5×VCI + 0.5×TCI",
            "VCI_formula":   "VCI = (NDVI - NDVImin_mois) / (NDVImax_mois - NDVImin_mois)",
            "TCI_formula":   "TCI = (LSTmax_mois - LST) / (LSTmax_mois - LSTmin_mois)",
            "source_ndvi":   "MODIS MOD13A2 v6.1 (ndvi_v3)",
            "source_lst":    "MODIS MOD11A2 v6.1 (lst_v3)",
            "units":         "dimensionless [0, 1]",
            "interpretation":"0 = stress maximal | 1 = état végétatif optimal",
            "note":          "Sprint 3 — VHI complet. Corrige Sprint 1 (VHI=VCI).",
            "zone":          zone,
            "period":        f"{str(common[0])[:7]} → {str(common[-1])[:7]}",
            "sprint":        "Sprint 3",
        }
    )
    ds_out = da.to_dataset(name="VHI_v3")
    ds_out.to_netcdf(output, encoding={"VHI_v3": {"dtype": "float32", "zlib": True}})
    ds_ndvi.close(); ds_lst.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


# =====================================================================
# VÉRIFICATION + CORRÉLATION SPI-3 vs VHI
# =====================================================================

def verify_and_correlate(zone):
    log.info("--- Vérification indices v3 — %s ---", zone.upper())

    for varname, fname in [("SPI3_v3", f"spi3_v3_{zone}.nc"),
                            ("VHI_v3",  f"vhi_v3_{zone}.nc")]:
        f = INDICES_DIR / fname
        if not f.exists():
            log.error("  MANQUANT : %s", fname)
            continue
        ds  = xr.open_dataset(f)
        var = ds[varname].values
        log.info("  %s : shape=%s | NaN=%.2f%% | min=%.3f | max=%.3f",
                 fname, var.shape,
                 np.isnan(var).mean() * 100,
                 float(np.nanmin(var)), float(np.nanmax(var)))
        ds.close()

    # Corrélation SPI-3 vs VHI
    try:
        spi = xr.open_dataset(INDICES_DIR / f"spi3_v3_{zone}.nc")["SPI3_v3"]
        vhi = xr.open_dataset(INDICES_DIR / f"vhi_v3_{zone}.nc")["VHI_v3"]

        # Aligner sur période commune
        common = pd.to_datetime(spi.time.values).intersection(pd.to_datetime(vhi.time.values))
        s = spi.sel(time=common).values.flatten()
        v = vhi.sel(time=common).values.flatten()
        valid = ~(np.isnan(s) | np.isnan(v))

        if valid.sum() > 2:
            r, _ = stats.pearsonr(s[valid], v[valid])
            status = "✓ attendu (stress hydrique → végétation)" if r > 0.1 else "⚠ faible"
            log.info("  Corrélation SPI-3 vs VHI_v3 : r = %.3f (n=%d) — %s",
                     r, valid.sum(), status)
        spi.close(); vhi.close()
    except Exception as e:
        log.warning("  Corrélation non calculée : %s", e)


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" SPRINT 3 — Script 12 : Indices v3")
    log.info(" SPI-3 v3 : calibration %d-%d | extraction 2000-03→2024-12",
             CALIBRATION_START, CALIBRATION_END)
    log.info(" VHI  v3 : VCI + TCI réel (LST disponible)")
    log.info("=" * 55)

    for zone in ZONES:
        log.info(">>> Zone : %s", zone.upper())
        compute_spi3_v3(zone)
        compute_vhi_v3(zone)
        verify_and_correlate(zone)
        log.info("")

    log.info("=" * 55)
    log.info(" Script 12 terminé. Fichiers dans : %s", INDICES_DIR)
    log.info("=" * 55)

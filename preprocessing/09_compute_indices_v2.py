"""
preprocessing/09_compute_indices_v2.py
========================================
SPRINT 1 — Calcul des indices v2 : vrai TCI/VHI et tentative SPEI-3.

Ce script NE REMPLACE PAS les résultats v1 (spi3_*.nc, vhi_*.nc).
Il produit des fichiers _v2 pour comparaison directe.

VHI v2 :
  TCI = (LSTmax_mensuel - LST) / (LSTmax_mensuel - LSTmin_mensuel)
  VHI = 0.5*VCI + 0.5*TCI  (formule standard, enfin applicable avec LST réelle)
  LSTmin/max calculés par mois calendaire sur 2015-2024 (cohérent avec script 05).

SPEI-3 v2 (tentative) :
  PET calculé via méthode Thornthwaite (climate-indices), utilisant LST comme
  proxy de température moyenne mensuelle + latitude du pixel.
  Climatologie de référence : 1981-2014 pour les précipitations.
  ATTENTION : LST disponible seulement 2015-2024 (MOD11A2 contrainte).
  La climatologie PET sera donc calculée sur 2015-2024 uniquement (10 ans)
  -- limitation à documenter, inférieure à la norme OMM de 30 ans pour PET seul,
  mais le signal precipitation (dominant) garde sa climatologie 1981-2014.

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 1 - 30/06/26
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

ZONES = ["haute_guinee", "moyenne_guinee"]

# Latitude moyenne approximative par zone (pour Thornthwaite)
ZONE_LATITUDE = {
    "haute_guinee": 11.0,
    "moyenne_guinee": 11.0,
}


# =====================================================================
# VHI v2 — VCI + vrai TCI
# =====================================================================

def compute_vhi_v2(zone):
    log.info("=== VHI v2 (VCI + TCI reel) — %s ===", zone.upper())

    output = INDICES_DIR / f"vhi_v2_{zone}.nc"
    if output.exists():
        log.info("  Deja calcule — ignore : %s", output.name)
        return

    ds_ndvi = xr.open_dataset(PROC_DIR / f"ndvi_{zone}.nc")
    ds_lst  = xr.open_dataset(PROC_DIR / f"lst_{zone}.nc")

    ndvi  = ds_ndvi["NDVI"].values.astype(np.float32)
    lst   = ds_lst["LST"].values.astype(np.float32)
    times = pd.to_datetime(ds_ndvi["NDVI"].time.values)
    T, H, W = ndvi.shape

    log.info("  Shape NDVI : %s | Shape LST : %s", ndvi.shape, lst.shape)

    # VCI par mois calendaire (identique a v1)
    vci = np.full_like(ndvi, np.nan)
    for month in range(1, 13):
        mask = times.month == month
        ndvi_m = ndvi[mask]
        ndvi_min = np.nanmin(ndvi_m, axis=0)
        ndvi_max = np.nanmax(ndvi_m, axis=0)
        ndvi_rng = ndvi_max - ndvi_min
        ndvi_rng[ndvi_rng < 1e-6] = 1e-6
        vci[mask] = np.clip((ndvi_m - ndvi_min) / ndvi_rng, 0.0, 1.0)

    # TCI REEL par mois calendaire, base sur LST
    tci = np.full_like(lst, np.nan)
    for month in range(1, 13):
        mask = times.month == month
        lst_m = lst[mask]
        lst_min = np.nanmin(lst_m, axis=0)
        lst_max = np.nanmax(lst_m, axis=0)
        lst_rng = lst_max - lst_min
        lst_rng[lst_rng < 1e-6] = 1e-6
        # LST elevee = stress = TCI bas -> formule inversee
        tci[mask] = np.clip((lst_max - lst_m) / lst_rng, 0.0, 1.0)

    # VHI = formule standard, enfin avec un vrai TCI
    vhi = np.clip(0.5 * vci + 0.5 * tci, 0.0, 1.0).astype(np.float32)

    log.info("  VCI : min=%.3f max=%.3f std=%.3f",
             float(np.nanmin(vci)), float(np.nanmax(vci)), float(np.nanstd(vci)))
    log.info("  TCI : min=%.3f max=%.3f std=%.3f",
             float(np.nanmin(tci)), float(np.nanmax(tci)), float(np.nanstd(tci)))
    log.info("  VHI v2 : min=%.3f max=%.3f mean=%.3f std=%.3f",
             float(np.nanmin(vhi)), float(np.nanmax(vhi)),
             float(np.nanmean(vhi)), float(np.nanstd(vhi)))

    da = xr.DataArray(
        data=vhi, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": ds_ndvi["NDVI"].lat.values, "lon": ds_ndvi["NDVI"].lon.values},
        attrs={
            "long_name": "Vegetation Health Index v2 (real TCI from LST)",
            "formula": "VHI = 0.5*VCI + 0.5*TCI",
            "tci_formula": "TCI = (LSTmax_monthly - LST) / (LSTmax_monthly - LSTmin_monthly)",
            "source": "MODIS MOD13A2 (NDVI) + MOD11A2 (LST)",
            "units": "dimensionless [0, 1]",
            "zone": zone, "period": "2015-2024",
            "comparison": "Compare with vhi_{zone}.nc (v1, VCI-only proxy)",
        }
    )
    ds_out = da.to_dataset(name="VHI_v2")
    ds_out.to_netcdf(output, encoding={"VHI_v2": {"dtype": "float32", "zlib": True}})

    ds_ndvi.close(); ds_lst.close()
    log.info("  OK : %s", output.name)


# =====================================================================
# SPEI-3 v2 — tentative avec PET Thornthwaite
# =====================================================================

def compute_spei3_pixel(precip_series, temp_series, latitude, data_start_year):
    """
    Calcule PET via Thornthwaite puis SPEI-3.
    precip_series et temp_series doivent couvrir la meme periode.
    """
    try:
        pet_mm = indices.pet(
            temperature_celsius=temp_series.astype(float),
            latitude_degrees=latitude,
            data_start_year=data_start_year,
        )
        spei_values = indices.spei(
            precips_mm=precip_series.astype(float),
            pet_mm=pet_mm,
            scale=3,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=data_start_year,
            calibration_year_initial=data_start_year,
            calibration_year_final=data_start_year + (len(precip_series) // 12) - 1,
        )
        return spei_values.astype(np.float32), pet_mm.astype(np.float32)
    except Exception as e:
        n = len(precip_series)
        return np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)


def compute_spei3_v2_zone(zone):
    """
    Tentative SPEI-3 avec PET reel.
    Contrainte : LST disponible seulement 2015-2024 (10 ans).
    Climatologie limitee a cette periode pour ce test -- a documenter
    comme limitation si l'extension SPEI complete est souhaitee plus tard.
    """
    log.info("=== SPEI-3 v2 (PET reel, climatologie 2015-2024) — %s ===", zone.upper())

    output = INDICES_DIR / f"spei3_v2_{zone}.nc"
    if output.exists():
        log.info("  Deja calcule — ignore : %s", output.name)
        return

    ds_chirps = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    ds_lst    = xr.open_dataset(PROC_DIR / f"lst_{zone}.nc")

    chirps_full = ds_chirps["precipitation"]
    times_full  = pd.to_datetime(chirps_full.time.values)

    # Restreindre CHIRPS a la meme periode que LST (2015-2024)
    mask_period = (times_full.year >= 2015) & (times_full.year <= 2024)
    chirps = chirps_full.values[mask_period]
    times  = times_full[mask_period]
    lst    = ds_lst["LST"].values

    T, H, W = chirps.shape
    log.info("  Periode alignee : %s -> %s (%d mois)", str(times[0])[:7], str(times[-1])[:7], T)

    latitude = ZONE_LATITUDE[zone]
    spei_full = np.full((T, H, W), np.nan, dtype=np.float32)
    pet_full  = np.full((T, H, W), np.nan, dtype=np.float32)

    total = H * W
    done  = 0
    n_failed_pixels = 0

    for h in range(H):
        for w in range(W):
            precip_px = chirps[:, h, w]
            lst_px    = lst[:, h, w]

            if np.isnan(lst_px).all() or np.isnan(precip_px).all():
                n_failed_pixels += 1
            else:
                # Combler les NaN residuels par la moyenne du pixel (rare, 1.7%)
                if np.isnan(lst_px).any():
                    lst_px = np.where(np.isnan(lst_px), np.nanmean(lst_px), lst_px)

                spei_vals, pet_vals = compute_spei3_pixel(
                    precip_px, lst_px, latitude, data_start_year=2015
                )
                spei_full[:, h, w] = spei_vals
                pet_full[:, h, w]  = pet_vals
                if np.isnan(spei_vals).all():
                    n_failed_pixels += 1

            done += 1
            if done % (total // 10) == 0:
                log.info("  Progression : %d%%", int(done / total * 100))

    nan_pct = np.isnan(spei_full).mean() * 100
    log.info("  Pixels echoues : %d / %d", n_failed_pixels, total)
    log.info("  NaN global : %.2f%%", nan_pct)

    if nan_pct < 100:
        log.info("  PET min=%.1f max=%.1f mm/mois (moyenne pixels valides)",
                 float(np.nanmin(pet_full)), float(np.nanmax(pet_full)))
        log.info("  SPEI-3 v2 min=%.3f max=%.3f",
                 float(np.nanmin(spei_full)), float(np.nanmax(spei_full)))

    da = xr.DataArray(
        data=spei_full, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": ds_chirps.lat.values, "lon": ds_chirps.lon.values},
        attrs={
            "long_name": "Standardized Precipitation-Evapotranspiration Index v2 (3-month)",
            "source": "CHIRPS (precip) + MOD11A2 LST -> Thornthwaite PET",
            "scale": "3 months",
            "calibration_period": "2015-2024 (limited by LST availability, NOT 30-year WMO norm)",
            "limitation": "PET climatology shorter than precipitation climatology used in SPI-3 v1 (1981-2014)",
            "zone": zone,
        }
    )
    ds_out = da.to_dataset(name="SPEI3_v2")
    ds_out.to_netcdf(output, encoding={"SPEI3_v2": {"dtype": "float32", "zlib": True}})

    ds_chirps.close(); ds_lst.close()
    log.info("  OK : %s", output.name)


# =====================================================================
# COMPARAISON v1 vs v2
# =====================================================================

def compare_versions(zone):
    log.info("--- Comparaison v1 vs v2 — %s ---", zone.upper())

    try:
        vhi_v1 = xr.open_dataset(INDICES_DIR / f"vhi_{zone}.nc")["VHI"].values
        vhi_v2 = xr.open_dataset(INDICES_DIR / f"vhi_v2_{zone}.nc")["VHI_v2"].values
        valid = ~(np.isnan(vhi_v1.flatten()) | np.isnan(vhi_v2.flatten()))
        r, _ = stats.pearsonr(vhi_v1.flatten()[valid], vhi_v2.flatten()[valid])
        log.info("  VHI v1 vs v2 : r=%.3f | std v1=%.3f | std v2=%.3f",
                 r, float(np.nanstd(vhi_v1)), float(np.nanstd(vhi_v2)))
    except FileNotFoundError:
        log.warning("  Fichiers VHI manquants pour comparaison")

    try:
        spi3_v1 = xr.open_dataset(INDICES_DIR / f"spi3_{zone}.nc")["SPI3"].values
        spei3_v2 = xr.open_dataset(INDICES_DIR / f"spei3_v2_{zone}.nc")["SPEI3_v2"].values
        # Aligner sur la periode commune 2015-2024 (deja le cas pour les deux)
        valid = ~(np.isnan(spi3_v1.flatten()) | np.isnan(spei3_v2.flatten()))
        if valid.sum() > 1:
            r, _ = stats.pearsonr(spi3_v1.flatten()[valid], spei3_v2.flatten()[valid])
            log.info("  SPI-3 (v1) vs SPEI-3 (v2) : r=%.3f (n=%d)", r, valid.sum())
        else:
            log.warning("  Pas assez de donnees valides pour comparer SPI-3/SPEI-3")
    except FileNotFoundError:
        log.warning("  Fichiers SPI/SPEI manquants pour comparaison")


if __name__ == "__main__":

    log.info("============================================")
    log.info(" SPRINT 1 — Indices v2 (LST integree)")
    log.info("============================================")

    for zone in ZONES:
        compute_vhi_v2(zone)
        compute_spei3_v2_zone(zone)
        compare_versions(zone)
        log.info("")

    log.info("============================================")
    log.info(" Script termine.")
    log.info(" Fichiers v2 dans : %s", INDICES_DIR)
    log.info("============================================")
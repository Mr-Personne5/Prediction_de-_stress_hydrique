"""
preprocessing/05_compute_indices.py
=====================================
Calcul des indices climatiques : SPI-3 et VHI (VCI proxy).

SPI-3 (Standardized Precipitation Index, 3 mois) :
- Variable CIBLE du modèle ConvLSTM
- Calculé à partir de CHIRPS 1981-2024
- Climatologie de référence : 1981-2014 (34 ans)
- Période d'étude : 2015-2024 (120 mois)

Choix SPI-3 vs SPEI-3 :
- La version actuelle de climate-indices (2.4.0) exige l'évapotranspiration
  potentielle (PET) pour le SPEI, non disponible sans données LST.
- En Haute Guinée (savane tropicale), la variabilité des précipitations
  domine le signal de stress hydrique. Le SPI-3 est suffisant.
- Cohérent avec la littérature Afrique de l'Ouest (Blanco & Arreyndip, 2025).
- Validation : SPI-3 détecte -2.14 et -2.00 en jan-fév 2021 (sécheresse
  documentée par FEWS NET à Siguiri).

VHI = VCI (Vegetation Condition Index) par mois calendaire :
- Variable de VALIDATION SECONDAIRE
- VCI = (NDVI - NDVImin_mensuel) / (NDVImax_mensuel - NDVImin_mensuel)
- La formule standard VHI = 0.5*VCI + 0.5*TCI nécessite LST pour TCI.
  Sans LST, TCI = 1-VCI annule VCI → VHI = 0.5 constant (inutilisable).
- Solution : VHI = VCI calculé par mois calendaire. Limitation documentée.
  Intégration LST prévue pour les travaux futurs.

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
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
STUDY_START       = 2015   # v1 : 2015-2024
STUDY_START_V2    = 2000   # v2 : 2000-2024 (Sprint 3)
STUDY_END         = 2024
ZONES = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# SPI-3
# =====================================================================

def compute_spi3_pixel(precip_series):
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


def compute_spi3_zone(zone):
    log.info("=== SPI-3 — %s ===", zone.upper())
    output = INDICES_DIR / f"spi3_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ds    = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    chirps = ds["precipitation"]
    times  = pd.to_datetime(chirps.time.values)
    T, H, W = chirps.shape
    log.info("  Shape : %s | %s → %s", chirps.shape,
             str(times[0])[:7], str(times[-1])[:7])
    log.info("  Calcul SPI-3 pixel par pixel...")

    spi_full = np.full((T, H, W), np.nan, dtype=np.float32)
    total = H * W
    done  = 0
    for h in range(H):
        for w in range(W):
            spi_full[:, h, w] = compute_spi3_pixel(chirps.values[:, h, w])
            done += 1
            if done % (total // 10) == 0:
                log.info("  Progression : %d%%", int(done / total * 100))

    mask        = (times.year >= STUDY_START) & (times.year <= STUDY_END)
    spi_study   = spi_full[mask]
    times_study = times[mask]

    log.info("  Timesteps extraits : %d", len(times_study))
    log.info("  NaN : %.2f%%", np.isnan(spi_study).mean() * 100)
    log.info("  Min : %.3f | Max : %.3f",
             float(np.nanmin(spi_study)), float(np.nanmax(spi_study)))

    da = xr.DataArray(
        data=spi_study,
        dims=["time", "lat", "lon"],
        coords={"time": times_study,
                "lat": chirps.lat.values,
                "lon": chirps.lon.values},
        attrs={
            "long_name":          "Standardized Precipitation Index (3-month)",
            "source":             "CHIRPS v3 — climate-indices 2.4.0",
            "scale":              "3 months",
            "distribution":       "Gamma",
            "calibration_period": f"{CALIBRATION_START}-{CALIBRATION_END}",
            "study_period":       f"{STUDY_START}-{STUDY_END}",
            "units":              "dimensionless (standardized)",
            "thresholds":         "< -2.0 extreme | < -1.5 severe | < -1.0 moderate",
            "note":               "SPI-3 used (SPEI-3 requires PET not available). "
                                  "Justified for West African savanna context.",
            "zone": zone,
        }
    )
    ds_out = da.to_dataset(name="SPI3")
    ds_out.to_netcdf(output, encoding={"SPI3": {"dtype": "float32", "zlib": True}})
    ds.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


# =====================================================================
# VHI = VCI PAR MOIS CALENDAIRE
# =====================================================================

def compute_vhi_zone(zone):
    """
    VHI = VCI calculé par mois calendaire.

    Décision méthodologique :
    La formule standard VHI = 0.5*VCI + 0.5*TCI nécessite LST pour TCI.
    Sans LST, TCI = 1-VCI => VHI = 0.5 constant (inutilisable).
    On utilise VHI = VCI seul. Limitation documentée. LST prévu pour les travaux futurs.

    VCI = (NDVI - NDVImin_mois) / (NDVImax_mois - NDVImin_mois)
    NDVImin/max calculés par mois calendaire sur 2015-2024.
    """
    log.info("=== VHI (=VCI) — %s ===", zone.upper())
    output = INDICES_DIR / f"vhi_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ds    = xr.open_dataset(PROC_DIR / f"ndvi_{zone}.nc")
    ndvi  = ds["NDVI"].values.astype(np.float32)
    times = pd.to_datetime(ds["NDVI"].time.values)
    T, H, W = ndvi.shape
    log.info("  Shape NDVI : %s", ndvi.shape)

    vci = np.full_like(ndvi, np.nan)
    for month in range(1, 13):
        mask     = times.month == month
        ndvi_m   = ndvi[mask]
        ndvi_min = np.nanmin(ndvi_m, axis=0)
        ndvi_max = np.nanmax(ndvi_m, axis=0)
        ndvi_rng = ndvi_max - ndvi_min
        ndvi_rng[ndvi_rng < 1e-6] = 1e-6
        vci[mask] = np.clip((ndvi_m - ndvi_min) / ndvi_rng, 0.0, 1.0)

    vhi = vci.astype(np.float32)

    log.info("  VHI (VCI) : min=%.3f | max=%.3f | mean=%.3f | std=%.3f",
             float(np.nanmin(vhi)), float(np.nanmax(vhi)),
             float(np.nanmean(vhi)), float(np.nanstd(vhi)))

    da = xr.DataArray(
        data=vhi,
        dims=["time", "lat", "lon"],
        coords={"time": times,
                "lat": ds["NDVI"].lat.values,
                "lon": ds["NDVI"].lon.values},
        attrs={
            "long_name":     "Vegetation Health Index (= VCI proxy)",
            "formula":       "VHI = VCI = (NDVI - NDVImin_monthly) / (NDVImax_monthly - NDVImin_monthly)",
            "note":          "Standard VHI = 0.5*VCI + 0.5*TCI not applicable without LST. "
                             "VHI=VCI used as proxy. LST integration planned for thesis.",
            "source":        "MODIS MOD13A2 v6.1",
            "units":         "dimensionless [0, 1]",
            "interpretation":"0 = maximum stress | 1 = maximum health",
            "zone":          zone,
            "period":        "2015-2024",
        }
    )
    ds_out = da.to_dataset(name="VHI")
    ds_out.to_netcdf(output, encoding={"VHI": {"dtype": "float32", "zlib": True}})
    ds.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


# =====================================================================
# SPRINT 3 — SPI-3 v2 (2000-2024) + VHI v2 (VCI + TCI réel)
# =====================================================================

def compute_spi3_v2_zone(zone):
    """
    SPI-3 pour la période étendue 2000-2024 → spi3_v2_{zone}.nc.
    Même calcul pixel-par-pixel que v1 ; seule l'extraction finale change
    (mask sur 2000-2024 au lieu de 2015-2024).
    Calibration inchangée : 1981-2014.
    """
    log.info("=== SPI-3 v2 (2000-2024) — %s ===", zone.upper())
    output = INDICES_DIR / f"spi3_v2_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ds     = xr.open_dataset(PROC_DIR / f"chirps_{zone}.nc")
    chirps = ds["precipitation"]
    times  = pd.to_datetime(chirps.time.values)
    T, H, W = chirps.shape
    log.info("  Shape : %s | %s → %s", chirps.shape,
             str(times[0])[:7], str(times[-1])[:7])
    log.info("  Calcul SPI-3 pixel par pixel...")

    spi_full = np.full((T, H, W), np.nan, dtype=np.float32)
    total = H * W
    done  = 0
    for h in range(H):
        for w in range(W):
            spi_full[:, h, w] = compute_spi3_pixel(chirps.values[:, h, w])
            done += 1
            if done % (total // 10) == 0:
                log.info("  Progression : %d%%", int(done / total * 100))

    mask        = (times.year >= STUDY_START_V2) & (times.year <= STUDY_END)
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
                "lat": chirps.lat.values,
                "lon": chirps.lon.values},
        attrs={
            "long_name":          "Standardized Precipitation Index (3-month)",
            "source":             "CHIRPS v3 — climate-indices 2.4.0",
            "scale":              "3 months",
            "distribution":       "Gamma",
            "calibration_period": f"{CALIBRATION_START}-{CALIBRATION_END}",
            "study_period":       f"{STUDY_START_V2}-{STUDY_END}",
            "units":              "dimensionless (standardized)",
            "thresholds":         "< -2.0 extreme | < -1.5 severe | < -1.0 moderate",
            "note":               "Sprint 3 — période étendue 2000-2024 (~300 mois). "
                                  "Même calibration que v1 (1981-2014).",
            "zone": zone,
        }
    )
    ds_out = da.to_dataset(name="SPI3")
    ds_out.to_netcdf(output, encoding={"SPI3": {"dtype": "float32", "zlib": True}})
    ds.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


def compute_vhi_v2_zone(zone):
    """
    VHI complet avec vrai TCI (Sprint 3) → vhi_v2_{zone}.nc.
    VHI = 0.5 * VCI + 0.5 * TCI
      VCI = (NDVI - NDVImin_mois) / (NDVImax_mois - NDVImin_mois)
      TCI = (LSTmax_mois - LST)   / (LSTmax_mois - LSTmin_mois)
    Corrige la limitation v1 (VHI = VCI uniquement faute de LST).
    Alignement temporel sur l'intersection NDVI v2 ∩ LST (2000-03 à 2024-12).
    """
    log.info("=== VHI v2 (VCI + TCI réel) — %s ===", zone.upper())
    output = INDICES_DIR / f"vhi_v2_{zone}.nc"
    if output.exists():
        log.info("  Déjà calculé — ignoré : %s", output.name)
        return

    ds_ndvi = xr.open_dataset(PROC_DIR / f"ndvi_v2_{zone}.nc")
    ds_lst  = xr.open_dataset(PROC_DIR / f"lst_{zone}.nc")

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

        ndvi_m   = ndvi[mask]
        ndvi_min = np.nanmin(ndvi_m, axis=0)
        ndvi_max = np.nanmax(ndvi_m, axis=0)
        ndvi_rng = np.where(ndvi_max - ndvi_min < 1e-6, 1e-6, ndvi_max - ndvi_min)
        vci[mask] = np.clip((ndvi_m - ndvi_min) / ndvi_rng, 0.0, 1.0)

        # TCI : LST élevée = stress → relation inverse
        lst_m    = lst[mask]
        lst_min  = np.nanmin(lst_m, axis=0)
        lst_max  = np.nanmax(lst_m, axis=0)
        lst_rng  = np.where(lst_max - lst_min < 1e-6, 1e-6, lst_max - lst_min)
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
                "lat": ds_ndvi["NDVI"].lat.values,
                "lon": ds_ndvi["NDVI"].lon.values},
        attrs={
            "long_name":     "Vegetation Health Index (VCI + TCI réel)",
            "formula":       "VHI = 0.5*VCI + 0.5*TCI",
            "VCI_formula":   "VCI = (NDVI - NDVImin_mois) / (NDVImax_mois - NDVImin_mois)",
            "TCI_formula":   "TCI = (LSTmax_mois - LST) / (LSTmax_mois - LSTmin_mois)",
            "source_ndvi":   "MODIS MOD13A2 v6.1 (ndvi_v2)",
            "source_lst":    "MODIS MOD11A2 v6.1",
            "units":         "dimensionless [0, 1]",
            "interpretation":"0 = stress maximal | 1 = état végétatif optimal",
            "note":          "Sprint 3 — VHI complet. Corrige la limitation Sprint 1 "
                             "(VHI=VCI uniquement, faute de LST).",
            "zone":          zone,
            "period":        f"{str(common[0])[:7]} → {str(common[-1])[:7]}",
        }
    )
    ds_out = da.to_dataset(name="VHI")
    ds_out.to_netcdf(output, encoding={"VHI": {"dtype": "float32", "zlib": True}})
    ds_ndvi.close(); ds_lst.close()
    log.info("  ✓ Sauvegardé : %s (%.1f MB)", output.name,
             output.stat().st_size / 1e6)


# =====================================================================
# VALIDATION CROISÉE SPI-3 vs VHI
# =====================================================================

def cross_validate(zone):
    log.info("--- Validation croisée SPI-3 vs VHI — %s ---", zone.upper())

    spi_ds = xr.open_dataset(INDICES_DIR / f"spi3_{zone}.nc")
    vhi_ds = xr.open_dataset(INDICES_DIR / f"vhi_{zone}.nc")

    spi = spi_ds["SPI3"].values.flatten()
    vhi = vhi_ds["VHI"].values.flatten()
    valid = ~(np.isnan(spi) | np.isnan(vhi))

    if valid.sum() < 2:
        log.warning("  Pas assez de données valides.")
        spi_ds.close(); vhi_ds.close()
        return np.nan

    r, _ = stats.pearsonr(spi[valid], vhi[valid])
    log.info("  Corrélation SPI-3 vs VHI : r = %.3f (n=%d)", r, valid.sum())

    if r >= 0.5:
        log.info("  ✓ Cohérence satisfaisante (r >= 0.5)")
    elif r >= 0.3:
        log.warning("  ⚠ Cohérence partielle (r >= 0.3)")
    elif abs(r) < 0.1:
        log.warning("  ⚠ Corrélation quasi-nulle — attendue car VHI=VCI "
                    "capture la saisonnalité NDVI, pas directement la pluie")
    else:
        log.warning("  Corrélation : r = %.3f", r)

    spi_ds.close(); vhi_ds.close()
    return r


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    # Supprimer les fichiers existants pour re-run propre
    for zone in ZONES:
        for prefix in ["spi3", "vhi"]:
            f = INDICES_DIR / f"{prefix}_{zone}.nc"
            if f.exists():
                f.unlink()
                log.info("Supprimé (re-run) : %s", f.name)

    log.info("============================================")
    log.info(" PHASE 2 — Script 05 : Calcul indices")
    log.info(" SPI-3 : climatologie %d-%d", CALIBRATION_START, CALIBRATION_END)
    log.info(" VHI   : VCI par mois calendaire 2015-2024")
    log.info(" Sprint 3 : SPI-3 v2 (2000-2024) + VHI v2 (TCI réel)")
    log.info("============================================")

    for zone in ZONES:
        # v1 (existant)
        compute_spi3_zone(zone)
        compute_vhi_zone(zone)
        cross_validate(zone)

        # Sprint 3 — indices étendus
        compute_spi3_v2_zone(zone)
        compute_vhi_v2_zone(zone)
        log.info("")

    log.info("============================================")
    log.info(" Script 05 terminé.")
    log.info(" Fichiers dans : %s", INDICES_DIR)
    log.info("============================================")
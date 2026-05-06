"""
preprocessing/05_compute_indices.py
=====================================
Calcul des indices climatiques : SPI-3 et VHI (VCI proxy).

SPI-3 (Standardized Precipitation Index, 3 mois) :
- Variable CIBLE du modèle ConvLSTM
- Calculé à partir de CHIRPS 1981-2024
- Climatologie de référence : 1981-2014 (34 ans)
- Période d'étude : 2015-2024 (120 mois)

Choix SPI-3 vs SPEI-3 (à citer dans le mémoire) :
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
  Intégration LST prévue pour la thèse.

Auteur : Djiba Kaba — Master IASD UKAG
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
STUDY_START       = 2015
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
    On utilise VHI = VCI seul. Limitation documentée. LST prévu pour thèse.

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
    log.info("============================================")

    for zone in ZONES:
        compute_spi3_zone(zone)
        compute_vhi_zone(zone)
        cross_validate(zone)
        log.info("")

    log.info("============================================")
    log.info(" Script 05 terminé.")
    log.info(" Fichiers dans : %s", INDICES_DIR)
    log.info("============================================")
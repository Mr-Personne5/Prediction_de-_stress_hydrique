"""
preprocessing/11_reproject_align_v3.py
========================================
Sprint 3 — Reprojection et alignement : NDVI 2000-2024 + LST 2000-2024.

Produit 4 fichiers NetCDF dans data/processed/ :
  ndvi_v3_haute_guinee.nc    — 299 timesteps (2000-02 → 2024-12)
  ndvi_v3_moyenne_guinee.nc  — 299 timesteps
  lst_v3_haute_guinee.nc     — 298 timesteps (2000-03 → 2024-12)
  lst_v3_moyenne_guinee.nc   — 298 timesteps

Les fichiers v1 existants (ndvi_haute_guinee.nc, etc.) ne sont pas touchés.
Grille de référence : EPSG:4326, 0.045°, 90×90 px (HG) / 46×68 px (MG).

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 3 — Juillet 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.warp
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = CRS.from_epsg(4326)
ZONES      = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# UTILITAIRES (identiques à 03_reproject_align.py)
# =====================================================================

def get_reference_grid(zone):
    """Établit la grille de référence depuis le premier fichier NDVI trié."""
    ndvi_dir = RAW_DIR / zone / "ndvi"
    ref_file = sorted(ndvi_dir.glob("ndvi_*.tif"))[0]
    with rasterio.open(ref_file) as src:
        return {
            "width":     src.width,
            "height":    src.height,
            "transform": src.transform,
            "crs":       TARGET_CRS,
            "bounds":    src.bounds,
            "res":       src.res[0],
        }


def read_and_align(filepath, ref_grid):
    """Lit un GeoTIFF et l'aligne sur la grille de référence."""
    with rasterio.open(filepath) as src:
        src_crs = src.crs if src.crs else TARGET_CRS

        if (src.width == ref_grid["width"] and
                src.height == ref_grid["height"] and
                src.crs == TARGET_CRS):
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data

        data = np.empty((ref_grid["height"], ref_grid["width"]), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=ref_grid["transform"],
            dst_crs=ref_grid["crs"],
            resampling=Resampling.bilinear,
        )
        if src.nodata is not None:
            data[np.abs(data - src.nodata) < 1e-5] = np.nan

    return data.astype(np.float32)


def build_coords(ref_grid):
    """Construit les vecteurs lat/lon depuis la grille de référence."""
    t = ref_grid["transform"]
    lons = np.array([t.c + (i + 0.5) * t.a for i in range(ref_grid["width"])])
    lats = np.array([t.f + (j + 0.5) * t.e for j in range(ref_grid["height"])])
    return lats, lons


# =====================================================================
# SCRIPT 1 — NDVI v3 (2000-02 → 2024-12)
# =====================================================================

def build_ndvi_v3(zone, ref_grid):
    """
    Empile les 299 fichiers NDVI mensuels (2000-02 → 2024-12).
    MOD13A2 disponible depuis 2000-02-18 → pas de 2000-01.
    """
    log.info("=== NDVI v3 — %s ===", zone.upper())
    output = PROC_DIR / f"ndvi_v3_{zone}.nc"
    if output.exists():
        log.info("  Déjà construit — ignoré : %s", output.name)
        return

    ndvi_dir = RAW_DIR / zone / "ndvi"
    times, arrays = [], []

    for year in range(2000, 2025):
        start_month = 2 if year == 2000 else 1
        for month in range(start_month, 13):
            f = ndvi_dir / f"ndvi_{year}_{month:02d}.tif"
            if not f.exists():
                log.warning("  Manquant : %s", f.name)
                continue
            arrays.append(read_and_align(f, ref_grid))
            times.append(pd.Timestamp(year=year, month=month, day=1))

    if not arrays:
        log.error("  Aucun fichier NDVI trouvé pour %s — téléchargement terminé ?", zone)
        return

    lats, lons = build_coords(ref_grid)
    da = xr.DataArray(
        data=np.stack(arrays, axis=0),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "long_name":  "Normalized Difference Vegetation Index",
            "source":     "MODIS MOD13A2 v6.1",
            "units":      "dimensionless [-0.2, 1.0]",
            "scale":      "0.0001 applied",
            "resolution": f"{round(ref_grid['res'], 4)} degrees (~5 km)",
            "zone":       zone,
            "period":     "2000-02 to 2024-12",
            "sprint":     "Sprint 3",
        }
    )
    da.to_dataset(name="NDVI").to_netcdf(
        output, encoding={"NDVI": {"dtype": "float32", "zlib": True}}
    )
    log.info("  ✓ %s — %d timesteps | %.1f MB",
             output.name, len(times), output.stat().st_size / 1e6)


# =====================================================================
# LST v3 (2000-03 → 2024-12)
# =====================================================================

def build_lst_v3(zone, ref_grid):
    """
    Empile les 298 fichiers LST mensuels (2000-03 → 2024-12).
    MOD11A2 disponible depuis 2000-03-05 → pas de 2000-01 ni 2000-02.
    Valeurs en °C (conversion DN × 0.02 − 273.15 appliquée à la source).
    """
    log.info("=== LST v3 — %s ===", zone.upper())
    output = PROC_DIR / f"lst_v3_{zone}.nc"
    if output.exists():
        log.info("  Déjà construit — ignoré : %s", output.name)
        return

    lst_dir = RAW_DIR / zone / "lst"
    if not lst_dir.exists():
        log.error("  Dossier LST introuvable : %s", lst_dir)
        return

    times, arrays = [], []

    for year in range(2000, 2025):
        start_month = 3 if year == 2000 else 1
        for month in range(start_month, 13):
            f = lst_dir / f"lst_{year}_{month:02d}.tif"
            if not f.exists():
                log.warning("  Manquant : %s", f.name)
                continue
            arrays.append(read_and_align(f, ref_grid))
            times.append(pd.Timestamp(year=year, month=month, day=1))

    if not arrays:
        log.error("  Aucun fichier LST trouvé pour %s", zone)
        return

    lats, lons = build_coords(ref_grid)
    da = xr.DataArray(
        data=np.stack(arrays, axis=0),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "long_name":  "Land Surface Temperature (Day)",
            "source":     "MODIS MOD11A2 v6.1",
            "units":      "degrees Celsius",
            "processing": "Médiane composites 8-jours/mois. DN < 7500 masqués. "
                          "Conversion : DN × 0.02 − 273.15 appliquée à la source.",
            "resolution": f"{round(ref_grid['res'], 4)} degrees (~5 km) — dégradée depuis 1 km",
            "zone":       zone,
            "period":     "2000-03 to 2024-12",
            "sprint":     "Sprint 3",
        }
    )
    da.to_dataset(name="LST").to_netcdf(
        output, encoding={"LST": {"dtype": "float32", "zlib": True}}
    )
    log.info("  ✓ %s — %d timesteps | %.1f MB",
             output.name, len(times), output.stat().st_size / 1e6)


# =====================================================================
# VÉRIFICATION FINALE
# =====================================================================

def verify(zone):
    log.info("--- Vérification %s ---", zone.upper())
    for varname, fname in [("NDVI", f"ndvi_v3_{zone}.nc"),
                            ("LST",  f"lst_v3_{zone}.nc")]:
        f = PROC_DIR / fname
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


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" SPRINT 3 — Script 11 : Reprojection + NetCDF v3")
    log.info(" NDVI 2000-2024 + LST 2000-2024")
    log.info("=" * 55)

    for zone in ZONES:
        log.info(">>> Zone : %s", zone.upper())
        ref_grid = get_reference_grid(zone)
        log.info("  Grille : %d×%d px | res=%.4f° | CRS=EPSG:4326",
                 ref_grid["width"], ref_grid["height"], ref_grid["res"])
        build_ndvi_v3(zone, ref_grid)
        build_lst_v3(zone, ref_grid)
        verify(zone)
        log.info("")

    log.info("=" * 55)
    log.info(" Script 11 terminé. Fichiers dans : %s", PROC_DIR)
    log.info("=" * 55)

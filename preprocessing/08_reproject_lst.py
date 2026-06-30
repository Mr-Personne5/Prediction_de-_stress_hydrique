"""
preprocessing/08_reproject_lst.py
====================================
SPRINT 1 — Reprojection LST et conversion NetCDF.

Même logique que le script 03 (NDVI/CHIRPS) mais appliquée à LST.
Produit un fichier NetCDF aligné sur la grille de référence déjà
utilisée pour NDVI/CHIRPS — garantit l'alignement pixel-à-pixel
entre toutes les sources.

Sorties :
- data/processed/lst_haute_guinee.nc    (120, 90, 90)
- data/processed/lst_moyenne_guinee.nc  (120, 46, 68)

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 1 - 30/06/26
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

BASE_DIR = Path(__file__).parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = CRS.from_epsg(4326)
ZONES = ["haute_guinee", "moyenne_guinee"]


def get_reference_grid(zone):
    """Utilise le NDVI déjà traité comme grille de référence — garantit l'alignement."""
    ndvi_dir = RAW_DIR / zone / "ndvi"
    ref_file = sorted(ndvi_dir.glob("ndvi_*.tif"))[0]
    with rasterio.open(ref_file) as src:
        return {
            "width": src.width, "height": src.height,
            "transform": src.transform, "crs": TARGET_CRS,
            "res": src.res[0]
        }


def read_and_align(filepath, ref_grid):
    with rasterio.open(filepath) as src:
        src_crs = src.crs if src.crs else TARGET_CRS

        if (src.width == ref_grid["width"] and
            src.height == ref_grid["height"] and
            src.crs == TARGET_CRS):
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            return data

        data = np.empty((ref_grid["height"], ref_grid["width"]), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1), destination=data,
            src_transform=src.transform, src_crs=src_crs,
            dst_transform=ref_grid["transform"], dst_crs=ref_grid["crs"],
            resampling=Resampling.bilinear
        )
        data = data.astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[np.abs(data - nodata) < 1e-5] = np.nan
    return data


def build_lst_netcdf(zone, ref_grid):
    log.info("=== LST -> NetCDF — %s ===", zone.upper())

    lst_dir = RAW_DIR / zone / "lst"
    output  = PROC_DIR / f"lst_{zone}.nc"

    if output.exists():
        log.info("  Déjà traité — ignoré : %s", output.name)
        return

    times, arrays = [], []

    for year in range(2015, 2025):
        for month in range(1, 13):
            f = lst_dir / f"lst_{year}_{month:02d}.tif"
            if not f.exists():
                log.warning("  Manquant : %s", f.name)
                continue
            data = read_and_align(f, ref_grid)
            times.append(pd.Timestamp(year=year, month=month, day=1))
            arrays.append(data)

    transform = ref_grid["transform"]
    lons = np.array([transform.c + (i + 0.5) * transform.a for i in range(ref_grid["width"])])
    lats = np.array([transform.f + (j + 0.5) * transform.e for j in range(ref_grid["height"])])

    da = xr.DataArray(
        data=np.stack(arrays, axis=0),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={
            "long_name": "Land Surface Temperature (Day)",
            "source": "MODIS MOD11A2 v6.1, band LST_Day_1km",
            "units": "degrees Celsius",
            "scale_applied": "0.02 (Kelvin) - 273.15 (Celsius)",
            "resolution": f"{round(ref_grid['res'], 4)} degrees (~5 km)",
            "zone": zone, "period": "2015-2024",
        }
    )

    ds = da.to_dataset(name="LST")
    ds.to_netcdf(output, encoding={"LST": {"dtype": "float32", "zlib": True}})

    log.info("  OK : %s (%d timesteps, %.1f MB)",
             output.name, len(times), output.stat().st_size / 1e6)


def verify_output(zone):
    log.info("--- Vérification %s ---", zone.upper())
    f = PROC_DIR / f"lst_{zone}.nc"
    if not f.exists():
        log.error("  Manquant : lst_%s.nc", zone)
        return
    ds = xr.open_dataset(f)
    var = ds["LST"]
    log.info("  shape=%s | NaN=%.1f%% | min=%.1f°C | max=%.1f°C | mean=%.1f°C",
             var.shape, float(np.isnan(var.values).mean() * 100),
             float(np.nanmin(var.values)), float(np.nanmax(var.values)),
             float(np.nanmean(var.values)))
    ds.close()


if __name__ == "__main__":

    log.info("============================================")
    log.info(" SPRINT 1 — Reprojection LST")
    log.info("============================================")

    for zone in ZONES:
        log.info("=== Zone : %s ===", zone.upper())
        ref_grid = get_reference_grid(zone)
        log.info("  Grille de référence : %dx%d px | res=%.4f°",
                 ref_grid["width"], ref_grid["height"], ref_grid["res"])
        build_lst_netcdf(zone, ref_grid)
        verify_output(zone)

    log.info("============================================")
    log.info(" Script terminé.")
    log.info("============================================")
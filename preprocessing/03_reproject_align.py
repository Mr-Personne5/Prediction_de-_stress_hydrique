"""
preprocessing/03_reproject_align.py
=====================================
Reprojection, alignement spatial et conversion au format NetCDF.

Contexte méthodologique :
- Les données brutes NDVI et CHIRPS sont toutes deux à ~0.045°/pixel (90×90 pixels)
- Elles partagent déjà la même emprise géographique et la même résolution
- La reprojection consiste principalement à :
  (1) Assigner explicitement le CRS WGS84 (EPSG:4326) manquant sur CHIRPS
  (2) Aligner pixel-à-pixel les deux grilles sur une grille de référence commune
  (3) Convertir en format NetCDF pour manipulation efficace avec xarray

Choix de résolution (décision méthodologique) :
- Résolution retenue : 0.045° (~5 km) — résolution native des données téléchargées
- Alternative envisagée : 0.1° (~11 km) — rejetée car elle réduirait la grille
  de 90×90 à ~40×40 pixels, perdant 75% de l'information spatiale
- La grille 90×90 est parfaitement gérable sur GPU RTX 4060 (mémoire suffisante)
- Ce choix est documenté dans le journal de recherche

Sorties :
- data/processed/ndvi_haute_guinee.nc    ← NDVI mensuel 2015-2024 (120 timesteps)
- data/processed/chirps_haute_guinee.nc  ← Précipitations mensuelles 1981-2024
- data/processed/ndvi_moyenne_guinee.nc
- data/processed/chirps_moyenne_guinee.nc

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import numpy as np
import xarray as xr
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import rasterio.warp
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Chemins ---
BASE_DIR  = Path(__file__).parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- CRS cible ---
TARGET_CRS = CRS.from_epsg(4326)  # WGS84 — système GPS standard

# --- Zones ---
ZONES = ["haute_guinee", "moyenne_guinee"]


# =====================================================================
# FONCTIONS
# =====================================================================

def get_reference_grid(zone):
    """
    Lit le premier fichier NDVI de la zone pour établir la grille de référence.
    Tous les autres fichiers seront alignés sur cette grille.

    Returns:
        dict avec width, height, transform, crs, bounds
    """
    ndvi_dir = RAW_DIR / zone / "ndvi"
    ref_file = sorted(ndvi_dir.glob("ndvi_*.tif"))[0]

    with rasterio.open(ref_file) as src:
        return {
            "width":     src.width,
            "height":    src.height,
            "transform": src.transform,
            "crs":       TARGET_CRS,
            "bounds":    src.bounds,
            "res":       src.res[0]
        }


def read_and_align(filepath, ref_grid):
    """
    Lit un fichier GeoTIFF et l'aligne sur la grille de référence.
    Assigne le CRS WGS84 si absent (cas CHIRPS).

    Args:
        filepath : Path — fichier .tif à lire
        ref_grid : dict — grille de référence

    Returns:
        np.ndarray 2D aligné sur la grille de référence
    """
    with rasterio.open(filepath) as src:
        # Assigner CRS si absent (cas CHIRPS)
        src_crs = src.crs if src.crs else TARGET_CRS

        # Si les dimensions et la résolution correspondent déjà — lecture directe
        if (src.width == ref_grid["width"] and
            src.height == ref_grid["height"] and
            src.crs == TARGET_CRS):
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            return data

        # Sinon — reprojection sur la grille de référence
        data = np.empty(
            (ref_grid["height"], ref_grid["width"]),
            dtype=np.float32
        )
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=ref_grid["transform"],
            dst_crs=ref_grid["crs"],
            resampling=Resampling.bilinear
        )
        data = data.astype(np.float32)

        # Masquer les valeurs nodata
        nodata = src.nodata
        if nodata is not None:
            data[np.abs(data - nodata) < 1e-5] = np.nan

    return data


def build_ndvi_netcdf(zone, ref_grid):
    """
    Empile les 120 fichiers NDVI mensuels (2015-2024) en un seul fichier NetCDF.
    Dimensions : (time=120, lat=90, lon=90)

    Args:
        zone     : str — 'haute_guinee' ou 'moyenne_guinee'
        ref_grid : dict — grille de référence
    """
    log.info("=== NDVI → NetCDF — %s ===", zone.upper())

    ndvi_dir = RAW_DIR / zone / "ndvi"
    output   = PROC_DIR / f"ndvi_{zone}.nc"

    if output.exists():
        log.info("  Déjà traité — ignoré : %s", output.name)
        return

    # Construire les coordonnées
    times, arrays = [], []

    for year in range(2015, 2025):
        for month in range(1, 13):
            f = ndvi_dir / f"ndvi_{year}_{month:02d}.tif"
            if not f.exists():
                log.warning("  Manquant : %s", f.name)
                continue
            data = read_and_align(f, ref_grid)
            times.append(pd.Timestamp(year=year, month=month, day=1))
            arrays.append(data)

    # Construire les coordonnées lat/lon depuis la grille de référence
    transform = ref_grid["transform"]
    lons = np.array([transform.c + (i + 0.5) * transform.a for i in range(ref_grid["width"])])
    lats = np.array([transform.f + (j + 0.5) * transform.e for j in range(ref_grid["height"])])

    # Créer le DataArray xarray
    da = xr.DataArray(
        data=np.stack(arrays, axis=0),
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat":  lats,
            "lon":  lons
        },
        attrs={
            "long_name":  "Normalized Difference Vegetation Index",
            "source":     "MODIS MOD13A2 v6.1",
            "units":      "dimensionless [-0.2, 1.0]",
            "scale":      "0.0001 applied",
            "resolution": f"{round(ref_grid['res'], 4)} degrees (~5 km)",
            "zone":       zone,
            "period":     "2015-2024",
        }
    )

    # Sauvegarder en NetCDF
    ds = da.to_dataset(name="NDVI")
    ds.to_netcdf(output, encoding={"NDVI": {"dtype": "float32", "zlib": True}})

    log.info("  ✓ Sauvegardé : %s (%d timesteps, %.1f MB)",
             output.name, len(times), output.stat().st_size / 1e6)


def build_chirps_netcdf(zone, ref_grid):
    """
    Empile les fichiers CHIRPS mensuels (1981-2024) en un seul fichier NetCDF.
    Dimensions : (time=528, lat=90, lon=90)

    Note : 1981-2014 = climatologie de référence pour SPEI-3
           2015-2024 = période d'étude (entraînement + évaluation)

    Args:
        zone     : str — 'haute_guinee' ou 'moyenne_guinee'
        ref_grid : dict — grille de référence
    """
    log.info("=== CHIRPS → NetCDF — %s ===", zone.upper())

    chirps_dir = RAW_DIR / zone / "chirps"
    output     = PROC_DIR / f"chirps_{zone}.nc"

    if output.exists():
        log.info("  Déjà traité — ignoré : %s", output.name)
        return

    times, arrays = [], []

    for year in range(1981, 2025):
        for month in range(1, 13):
            f = chirps_dir / f"chirps_{year}_{month:02d}.tif"
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
        coords={
            "time": times,
            "lat":  lats,
            "lon":  lons
        },
        attrs={
            "long_name":      "Monthly Total Precipitation",
            "source":         "CHIRPS v3 (UCSB-CHG/CHIRPS/DAILY aggregated monthly)",
            "units":          "mm/month",
            "resolution":     f"{round(ref_grid['res'], 4)} degrees (~5 km)",
            "zone":           zone,
            "period":         "1981-2024",
            "note_spei":      "1981-2014 used as climatology reference for SPEI-3 calculation",
            "note_training":  "2015-2024 used for model training and evaluation",
            "validation":     "Validated vs ERA5-Land: r=0.978 (HG), r=0.970 (MG), bias<10%"
        }
    )

    ds = da.to_dataset(name="precipitation")
    ds.to_netcdf(output, encoding={"precipitation": {"dtype": "float32", "zlib": True}})

    log.info("  ✓ Sauvegardé : %s (%d timesteps, %.1f MB)",
             output.name, len(times), output.stat().st_size / 1e6)


# =====================================================================
# VÉRIFICATION FINALE
# =====================================================================

def verify_output(zone):
    """Vérifie la cohérence des fichiers NetCDF produits."""
    log.info("--- Vérification %s ---", zone.upper())

    for varname, filename in [("NDVI", f"ndvi_{zone}.nc"),
                               ("precipitation", f"chirps_{zone}.nc")]:
        f = PROC_DIR / filename
        if not f.exists():
            log.error("  Manquant : %s", filename)
            continue

        ds = xr.open_dataset(f)
        var = ds[varname]
        log.info("  %s : shape=%s | NaN=%.1f%% | min=%.3f | max=%.3f",
                 filename,
                 var.shape,
                 float(np.isnan(var.values).mean() * 100),
                 float(np.nanmin(var.values)),
                 float(np.nanmax(var.values)))
        ds.close()


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("============================================")
    log.info(" PHASE 2 — Script 03 : Reprojection + NetCDF")
    log.info(" Résolution conservée : ~0.045° (~5 km)")
    log.info(" Grille : 90 × 90 pixels par zone")
    log.info("============================================")

    for zone in ZONES:
        log.info("=== Zone : %s ===", zone.upper())
        ref_grid = get_reference_grid(zone)
        log.info("  Grille de référence : %dx%d px | res=%.4f° | CRS=EPSG:4326",
                 ref_grid["width"], ref_grid["height"], ref_grid["res"])

        build_ndvi_netcdf(zone, ref_grid)
        build_chirps_netcdf(zone, ref_grid)
        verify_output(zone)

    log.info("============================================")
    log.info(" Script 03 terminé.")
    log.info(" Fichiers dans : %s", PROC_DIR)
    log.info("============================================")
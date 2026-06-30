"""
preprocessing/07_download_lst.py
==================================
SPRINT 1 — Téléchargement LST MODIS MOD11A2.

Objectif : obtenir la température de surface (Land Surface Temperature)
pour calculer un vrai TCI (Temperature Condition Index) et permettre
le calcul du vrai SPEI-3 (via PET).

Source : MODIS/061/MOD11A2 — composite 8 jours, ~1 km, bande LST_Day_1km.
Facteur d'échelle MODIS standard : 0.02 (valeur brute -> Kelvin).
Conversion : Celsius = (brute * 0.02) - 273.15

Agrégation mensuelle par moyenne (cohérent avec NDVI déjà traité).
Période : 2015-2024 (alignée sur NDVI).

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Sprint 1 — LST + SPEI-3 - 30/06/26
"""

import ee
import time
import logging
import requests
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

PROJECT_ID = "master-iasd-guinee"
ee.Initialize(project=PROJECT_ID)
log.info("GEE initialisé — projet : %s", PROJECT_ID)

SCALE_M = 5000
CRS = "EPSG:4326"

BASE_DIR = Path(__file__).parent.parent / "data" / "raw"
LST_HG = BASE_DIR / "haute_guinee" / "lst"
LST_MG = BASE_DIR / "moyenne_guinee" / "lst"
for d in [LST_HG, LST_MG]:
    d.mkdir(parents=True, exist_ok=True)

BBOX_HG = ee.Geometry.Rectangle([-12.0, 9.0, -8.0, 13.0])
BBOX_MG = ee.Geometry.Rectangle([-13.0, 10.0, -10.0, 12.0])

ZONES = {
    "haute_guinee":   {"region": BBOX_HG, "out": LST_HG},
    "moyenne_guinee": {"region": BBOX_MG, "out": LST_MG},
}

# Facteur d'échelle MODIS LST officiel
LST_SCALE_FACTOR = 0.02
KELVIN_TO_CELSIUS = 273.15


def download_image(image, region, filename, scale=SCALE_M, crs=CRS):
    if filename.exists():
        return True
    try:
        url = image.getDownloadUrl({
            "scale": scale, "crs": crs, "region": region,
            "format": "GEO_TIFF", "filePerBand": False,
        })
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        size_kb = filename.stat().st_size / 1e3
        log.info("  OK : %s (%.1f KB)", filename.name, size_kb)
        return True
    except Exception as e:
        log.error("  ECHEC %s : %s", filename.name, str(e)[:100])
        return False


def download_lst(zone_name, region, output_dir):
    """
    Télécharge LST_Day_1km MOD11A2, agrégé mensuellement par moyenne.
    Conversion Kelvin -> Celsius appliquée avant sauvegarde.
    """
    log.info("=== LST MODIS — %s ===", zone_name.upper())

    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate("2015-01-01", "2024-12-31")
        .filterBounds(region)
        .select("LST_Day_1km")
    )

    ok, fail = 0, 0

    for year in range(2015, 2025):
        for month in range(1, 13):
            filename = output_dir / f"lst_{year}_{month:02d}.tif"

            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            start = f"{year}-{month:02d}-01"
            end   = f"{end_year}-{end_month:02d}-01"

            monthly = (
                collection
                .filterDate(start, end)
                .mean()
                .multiply(LST_SCALE_FACTOR)
                .subtract(KELVIN_TO_CELSIUS)
                .rename("LST_Celsius")
                .clip(region)
            )

            success = download_image(monthly, region, filename)
            ok   += 1 if success else 0
            fail += 0 if success else 1
            time.sleep(0.4)

    log.info("LST %s — %d OK / %d erreurs", zone_name, ok, fail)
    return ok, fail


if __name__ == "__main__":

    log.info("============================================")
    log.info(" SPRINT 1 — Téléchargement LST MODIS MOD11A2")
    log.info(" Résolution : 5000 m (~0.05°) — cohérent avec NDVI/CHIRPS")
    log.info(" Période : 2015-2024")
    log.info("============================================")

    total_ok, total_fail = 0, 0
    for zone_name, cfg in ZONES.items():
        ok, fail = download_lst(zone_name, cfg["region"], cfg["out"])
        total_ok += ok
        total_fail += fail

    log.info("============================================")
    log.info(" TERMINE — %d fichiers OK / %d erreurs", total_ok, total_fail)
    log.info("============================================")
"""
preprocessing/01b_download_sprint3.py
======================================
Sprint 3 — Extension temporelle : NDVI 2000-2024 + LST 2000-2024.

Ce script complète le téléchargement initial (01_download_gee.py) avec :
- NDVI MODIS MOD13A2 v6.1 étendu à 2000-2024
  (les mois 2015-2024 déjà présents sont ignorés par vérification d'existence)
- LST MODIS MOD11A2 v6.1 — 2000-2024 — agrégé mensuellement
  (Land Surface Temperature de jour, converti en °C)

Motivation : 120 mois (2015-2024) insuffisants pour ConvLSTM → extension à
~299 mois (2000-02 à 2024-12) pour tripler le volume temporel d'entraînement.

CHIRPS 1981-2024 déjà disponible — non retéléchargé ici.

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Juillet 2026
"""

import ee
import time
import logging
import requests
import zipfile
import io
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

# =====================================================================
# CONFIGURATION
# =====================================================================

SCALE_M = 5000   # 5000 m ≈ 0.05° — cohérent avec NDVI et CHIRPS existants
CRS     = "EPSG:4326"

BASE_DIR = Path(__file__).parent.parent / "data" / "raw"
NDVI_HG  = BASE_DIR / "haute_guinee"  / "ndvi"
NDVI_MG  = BASE_DIR / "moyenne_guinee" / "ndvi"
LST_HG   = BASE_DIR / "haute_guinee"  / "lst"
LST_MG   = BASE_DIR / "moyenne_guinee" / "lst"

for d in [NDVI_HG, NDVI_MG, LST_HG, LST_MG]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ZONES D'ÉTUDE
# =====================================================================

BBOX_HG = ee.Geometry.Rectangle([-12.0, 9.0, -8.0, 13.0])
BBOX_MG = ee.Geometry.Rectangle([-13.0, 10.0, -10.0, 12.0])

ZONES = {
    "haute_guinee":   (BBOX_HG, NDVI_HG, LST_HG),
    "moyenne_guinee": (BBOX_MG, NDVI_MG, LST_MG),
}

# =====================================================================
# UTILITAIRE DE TÉLÉCHARGEMENT
# =====================================================================

def download_image(image, region, filename):
    """Télécharge une ee.Image en GeoTIFF local. Ignore si déjà présent."""
    if filename.exists():
        log.info("  Ignoré (déjà présent) : %s", filename.name)
        return True

    try:
        url = image.getDownloadUrl({
            "scale":       SCALE_M,
            "crs":         CRS,
            "region":      region,
            "format":      "GEO_TIFF",
            "filePerBand": False,
        })

        log.info("  Téléchargement : %s", filename.name)
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "zip" in content_type:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                tif_files = [f for f in z.namelist() if f.endswith(".tif")]
                if tif_files:
                    with z.open(tif_files[0]) as src, open(filename, "wb") as dst:
                        dst.write(src.read())
        else:
            with open(filename, "wb") as f:
                f.write(response.content)

        size_kb = filename.stat().st_size / 1e3
        log.info("  ✓ %s (%.1f KB)", filename.name, size_kb)
        return True

    except Exception as e:
        log.error("  ✗ Erreur %s : %s", filename.name, str(e))
        return False


# =====================================================================
# NDVI MODIS MOD13A2 — 2000-2024
# =====================================================================

def download_ndvi(zone_name, region, output_dir):
    """
    NDVI MODIS MOD13A2 v6.1, agrégé mensuellement (médiane + facteur 0.0001).

    MOD13A2 disponible depuis 2000-02-18 → on commence à 2000-02.
    Les fichiers 2015-2024 déjà présents sont automatiquement ignorés.
    Période effective : 2000-02 à 2024-12 = 299 mois.
    """
    log.info("=== NDVI MOD13A2 2000-2024 — %s ===", zone_name.upper())

    collection = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate("2000-01-01", "2024-12-31")
        .filterBounds(region)
        .select("NDVI")
    )

    ok, fail = 0, 0

    for year in range(2000, 2025):
        # MOD13A2 n'a pas d'image complète en janvier 2000 (premier composite : 2000-02-18)
        start_month = 2 if year == 2000 else 1
        for month in range(start_month, 13):
            filename = output_dir / f"ndvi_{year}_{month:02d}.tif"

            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            t_start   = f"{year}-{month:02d}-01"
            t_end     = f"{end_year}-{end_month:02d}-01"

            monthly = (
                collection
                .filterDate(t_start, t_end)
                .median()
                .multiply(0.0001)
                .rename("NDVI")
                .clip(region)
            )

            success = download_image(monthly, region, filename)
            ok   += success
            fail += not success
            time.sleep(0.5)

    log.info("NDVI %s — %d OK / %d erreurs", zone_name, ok, fail)
    return ok, fail


# =====================================================================
# LST MODIS MOD11A2 — 2000-2024
# =====================================================================

def _mask_lst_invalid(image):
    """
    Masque les pixels LST invalides (fill value = 0 DN, valeurs DN < 7500).
    Plage valide MOD11A2 LST_Day_1km : 7500–65535 DN (soit ≈ -23°C à 1037°C).
    """
    valid = image.select("LST_Day_1km").gte(7500)
    return image.updateMask(valid)


def download_lst(zone_name, region, output_dir):
    """
    LST MODIS MOD11A2 v6.1, agrégé mensuellement (médiane des composites 8-jours).
    Conversion : DN × 0.02 − 273.15 → °C.
    Pixels invalides masqués avant agrégation.

    MOD11A2 disponible depuis 2000-03-05 → on commence à 2000-03.
    Période effective : 2000-03 à 2024-12 = 298 mois.
    """
    log.info("=== LST MOD11A2 2000-2024 — %s ===", zone_name.upper())

    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate("2000-01-01", "2024-12-31")
        .filterBounds(region)
        .select("LST_Day_1km")
        .map(_mask_lst_invalid)
    )

    ok, fail = 0, 0

    for year in range(2000, 2025):
        # MOD11A2 commence le 2000-03-05 → pas de données jan-fév 2000
        start_month = 3 if year == 2000 else 1
        for month in range(start_month, 13):
            filename = output_dir / f"lst_{year}_{month:02d}.tif"

            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            t_start   = f"{year}-{month:02d}-01"
            t_end     = f"{end_year}-{end_month:02d}-01"

            monthly = (
                collection
                .filterDate(t_start, t_end)
                .median()
                .multiply(0.02)
                .subtract(273.15)
                .rename("LST_C")
                .clip(region)
            )

            success = download_image(monthly, region, filename)
            ok   += success
            fail += not success
            time.sleep(0.5)

    log.info("LST %s — %d OK / %d erreurs", zone_name, ok, fail)
    return ok, fail


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("=" * 50)
    log.info(" SPRINT 3 — Extension temporelle 2000-2024")
    log.info(" NDVI MOD13A2 v6.1 + LST MOD11A2 v6.1")
    log.info(" Résolution : %d m (~0.05°)", SCALE_M)
    log.info(" Projet GEE : %s", PROJECT_ID)
    log.info("=" * 50)

    # Estimation : ~179 nouveaux fichiers NDVI × 2 zones + ~298 LST × 2 zones
    # ≈ 954 fichiers au total — durée estimée : 3-6 heures selon latence GEE
    log.info(" Estimation : ~954 fichiers (NDVI nouveaux + LST complet)")
    log.info(" Les fichiers NDVI 2015-2024 existants seront ignorés.")
    log.info("=" * 50)

    total_ok, total_fail = 0, 0

    for zone_name, (region, ndvi_dir, lst_dir) in ZONES.items():
        ok, fail = download_ndvi(zone_name, region, ndvi_dir)
        total_ok += ok; total_fail += fail

    for zone_name, (region, ndvi_dir, lst_dir) in ZONES.items():
        ok, fail = download_lst(zone_name, region, lst_dir)
        total_ok += ok; total_fail += fail

    log.info("=" * 50)
    log.info(" TERMINÉ — %d fichiers OK / %d erreurs", total_ok, total_fail)
    log.info(" Données dans : %s", BASE_DIR)
    log.info("=" * 50)

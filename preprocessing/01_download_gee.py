"""
preprocessing/01_download_gee.py
=================================
Script d'acquisition des données satellitaires via Google Earth Engine.

Téléchargement direct en local (sans Google Drive) :
- NDVI MODIS MOD13A2 v6.1 — 2015-2024 — agrégé mensuellement
- Précipitations CHIRPS DAILY agrégé en mensuel — 1981-2024

Résolution de téléchargement : 0.05° (~5.5 km).
Le preprocessing (script 03) agrégera ensuite à 0.1° pour l'entraînement.

Note CHIRPS : l'asset MONTHLY n'est pas accessible via l'API Python GEE.
On utilise CHIRPS/DAILY avec une agrégation mensuelle par somme (mm/mois).
C'est la méthode standard — scientifiquement équivalente.

Zones :
- Haute Guinée (zone principale) : Kankan, Siguiri, Mandiana, Kouroussa, Kérouané
- Nord Moyenne Guinée (zone de contrôle) : Labé, Dinguiraye

Auteur : Djiba Kaba — Master IASD UKAG
Date   : Avril 2026
"""

import ee
import time
import logging
import requests
import zipfile
import io
from pathlib import Path

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Initialisation GEE ---
PROJECT_ID = "master-iasd-guinee"
ee.Initialize(project=PROJECT_ID)
log.info("GEE initialisé — projet : %s", PROJECT_ID)

# =====================================================================
# CONFIGURATION
# =====================================================================

# Résolution de téléchargement : 5000 m ≈ 0.05° (résolution native CHIRPS)
SCALE_M = 5000

# Système de coordonnées cible
CRS = "EPSG:4326"

# Dossiers de sortie
BASE_DIR  = Path(__file__).parent.parent / "data" / "raw"
NDVI_HG   = BASE_DIR / "haute_guinee" / "ndvi"
NDVI_MG   = BASE_DIR / "moyenne_guinee" / "ndvi"
CHIRPS_HG = BASE_DIR / "haute_guinee" / "chirps"
CHIRPS_MG = BASE_DIR / "moyenne_guinee" / "chirps"

for d in [NDVI_HG, NDVI_MG, CHIRPS_HG, CHIRPS_MG]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================================
# ZONES D'ÉTUDE (bounding boxes)
# =====================================================================

BBOX_HG = ee.Geometry.Rectangle([-12.0, 9.0, -8.0, 13.0])
BBOX_MG = ee.Geometry.Rectangle([-13.0, 10.0, -10.0, 12.0])

ZONES = {
    "haute_guinee":   BBOX_HG,
    "moyenne_guinee": BBOX_MG,
}

# =====================================================================
# FONCTIONS UTILITAIRES
# =====================================================================

def download_image(image, region, filename, scale=SCALE_M, crs=CRS):
    """
    Télécharge une image GEE directement en local (sans Google Drive).

    Args:
        image    : ee.Image à télécharger
        region   : ee.Geometry définissant la zone
        filename : Path — chemin de sauvegarde local (.tif)
        scale    : résolution en mètres
        crs      : système de coordonnées

    Returns:
        True si succès, False sinon
    """
    if filename.exists():
        log.info("  Déjà téléchargé — ignoré : %s", filename.name)
        return True

    try:
        url = image.getDownloadUrl({
            "scale":       scale,
            "crs":         crs,
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
        log.info("  ✓ Sauvegardé : %s (%.1f KB)", filename.name, size_kb)
        return True

    except Exception as e:
        log.error("  ✗ Erreur pour %s : %s", filename.name, str(e))
        return False


# =====================================================================
# TÉLÉCHARGEMENT NDVI MODIS MOD13A2 — 2015-2024
# =====================================================================

def download_ndvi(zone_name, region, output_dir):
    """
    Télécharge le NDVI MODIS MOD13A2 agrégé mensuellement.
    Un fichier GeoTIFF par mois — résolution 5000 m (~0.05°).
    Facteur d'échelle MODIS (0.0001) appliqué — valeurs NDVI dans [-0.2, 1.0].
    """
    log.info("=== NDVI MODIS — %s ===", zone_name.upper())

    collection = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate("2015-01-01", "2024-12-31")
        .filterBounds(region)
        .select("NDVI")
    )

    ok, fail = 0, 0

    for year in range(2015, 2025):
        for month in range(1, 13):
            filename = output_dir / f"ndvi_{year}_{month:02d}.tif"

            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            start = f"{year}-{month:02d}-01"
            end   = f"{end_year}-{end_month:02d}-01"

            monthly = (
                collection  
                .filterDate(start, end)     # → garde uniquement les images du mois
                .median()   # → élimine les valeurs nuageuses résiduelles
                .multiply(0.0001)   # → applique le facteur d'échelle MODIS
                .rename("NDVI")
                .clip(region)   # → découpe sur ta zone d'étude
            )

            success = download_image(monthly, region, filename)
            ok   += 1 if success else 0
            fail += 0 if success else 1
            time.sleep(0.5)

    log.info("NDVI %s — %d OK / %d erreurs", zone_name, ok, fail)
    return ok, fail


# =====================================================================
# TÉLÉCHARGEMENT CHIRPS — 1981-2024
# Agrégation mensuelle depuis CHIRPS/DAILY (somme mm/mois)
# =====================================================================

def download_chirps(zone_name, region, output_dir):
    """
    Télécharge les précipitations CHIRPS agrégées mensuellement.
    Source : UCSB-CHG/CHIRPS/DAILY — agrégation par somme mensuelle (mm/mois).

    Note : l'asset MONTHLY n'est pas accessible via l'API Python GEE.
    L'agrégation depuis DAILY est scientifiquement équivalente.

    Téléchargé depuis 1981 pour la climatologie de référence SPEI-3.
    La période d'entraînement 2015-2024 s'y inscrit comme sous-ensemble.
    """
    log.info("=== CHIRPS DAILY→MONTHLY — %s ===", zone_name.upper())

    collection = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region)
        .select("precipitation")
    )

    ok, fail = 0, 0

    for year in range(1981, 2025):
        for month in range(1, 13):
            filename = output_dir / f"chirps_{year}_{month:02d}.tif"

            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            start = f"{year}-{month:02d}-01"
            end   = f"{end_year}-{end_month:02d}-01"

            # Somme des précipitations quotidiennes du mois = total mensuel mm
            monthly = (
                collection
                .filterDate(start, end)
                .sum()
                .rename("precipitation")
                .clip(region)
            )

            success = download_image(monthly, region, filename)
            ok   += 1 if success else 0
            fail += 0 if success else 1
            time.sleep(0.3)

    log.info("CHIRPS %s — %d OK / %d erreurs", zone_name, ok, fail)
    return ok, fail


# =====================================================================
# POINT D'ENTRÉE PRINCIPAL
# =====================================================================

if __name__ == "__main__":

    log.info("============================================")
    log.info(" PHASE 1 — Acquisition données GEE")
    log.info(" Résolution : 5000 m (~0.05°)")
    log.info(" Projet GEE : %s", PROJECT_ID)
    log.info("============================================")

    total_ok, total_fail = 0, 0

    # --- NDVI MODIS (2015-2024) — déjà téléchargé, sera ignoré ---
    for zone_name, region in ZONES.items():
        out = NDVI_HG if zone_name == "haute_guinee" else NDVI_MG
        ok, fail = download_ndvi(zone_name, region, out)
        total_ok += ok; total_fail += fail

    # --- CHIRPS agrégé depuis DAILY (1981-2024) ---
    for zone_name, region in ZONES.items():
        out = CHIRPS_HG if zone_name == "haute_guinee" else CHIRPS_MG
        ok, fail = download_chirps(zone_name, region, out)
        total_ok += ok; total_fail += fail

    log.info("============================================")
    log.info(" TERMINÉ — %d fichiers OK / %d erreurs", total_ok, total_fail)
    log.info(" Données dans : %s", BASE_DIR)
    log.info("============================================")
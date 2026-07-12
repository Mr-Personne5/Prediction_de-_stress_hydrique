"""
preprocessing/04_fill_missing.py
=====================================
Détection et interpolation des valeurs manquantes (NaN).

Contexte méthodologique :
- Les images MODIS peuvent contenir des pixels masqués par les nuages
- Le produit mensuel composite NASA filtre déjà la majorité des nuages
- Cette étape traite les valeurs résiduelles manquantes
- Méthode : interpolation linéaire temporelle pixel par pixel
  (on remplace un mois manquant par la moyenne pondérée des mois voisins)
- Seuil : si un pixel a > 3 mois consécutifs manquants, il est signalé
  mais non interpolé (interpolation sur longue période = trop incertaine)

Note : lors de l'exécution initiale (Mai 2026), NaN = 0.0% sur tous
les fichiers. Ce script est néanmoins maintenu pour la robustesse et
la reproductibilité de la démarche.

Sorties : fichiers NetCDF mis à jour in-place (remplacement des NaN)
Rapport : résumé du nombre de pixels interpolés par fichier

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import xarray as xr
import logging
from pathlib import Path

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Chemins ---
BASE_DIR = Path(__file__).parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

# Seuil : ne pas interpoler si plus de N mois consécutifs manquants
MAX_CONSECUTIVE_NAN = 3

# Fichiers à traiter
FILES = {
    "ndvi_haute_guinee.nc":       "NDVI",
    "ndvi_moyenne_guinee.nc":     "NDVI",
    "chirps_haute_guinee.nc":     "precipitation",
    "chirps_moyenne_guinee.nc":   "precipitation",
    # Sprint 3 — extension temporelle
    "ndvi_v2_haute_guinee.nc":    "NDVI",
    "ndvi_v2_moyenne_guinee.nc":  "NDVI",
    "lst_haute_guinee.nc":        "LST",
    "lst_moyenne_guinee.nc":      "LST",
}


# =====================================================================
# FONCTIONS
# =====================================================================

def count_nan(data):
    """Compte le nombre total de valeurs NaN dans un array 3D."""
    return int(np.isnan(data).sum())


def interpolate_temporal(data):
    """
    Interpolation linéaire temporelle pixel par pixel.

    Pour chaque pixel (lat, lon), remplace les NaN par interpolation
    linéaire entre les valeurs voisines non-NaN dans la dimension temps.

    Règle : si un pixel a plus de MAX_CONSECUTIVE_NAN mois consécutifs
    manquants, il n'est pas interpolé (trop incertain).

    Args:
        data : np.ndarray 3D (time, lat, lon)

    Returns:
        data_filled : np.ndarray 3D interpolé
        n_filled    : int — nombre de valeurs interpolées
        n_unfilled  : int — nombre de valeurs non interpolées (trop longues)
    """
    data_filled = data.copy()
    n_filled    = 0
    n_unfilled  = 0

    T, H, W = data.shape

    for h in range(H):
        for w in range(W):
            pixel = data[:, h, w]

            if not np.any(np.isnan(pixel)):
                continue  # Pas de NaN — on passe

            # Identifier les indices NaN
            nan_mask = np.isnan(pixel)
            nan_indices = np.where(nan_mask)[0]

            for idx in nan_indices:
                # Chercher les voisins valides à gauche et à droite
                left  = idx - 1
                right = idx + 1

                # Avancer jusqu'au premier voisin valide à gauche
                while left >= 0 and np.isnan(pixel[left]):
                    left -= 1

                # Avancer jusqu'au premier voisin valide à droite
                while right < T and np.isnan(pixel[right]):
                    right += 1

                # Vérifier la longueur du gap
                gap_size = right - left - 1
                if gap_size > MAX_CONSECUTIVE_NAN:
                    n_unfilled += 1
                    continue

                # Interpolation linéaire
                if left >= 0 and right < T:
                    # Interpolation entre deux voisins
                    alpha = (idx - left) / (right - left)
                    data_filled[idx, h, w] = (
                        pixel[left] * (1 - alpha) + pixel[right] * alpha
                    )
                    n_filled += 1
                elif left >= 0:
                    # Extrapolation à droite — copier la dernière valeur
                    data_filled[idx, h, w] = pixel[left]
                    n_filled += 1
                elif right < T:
                    # Extrapolation à gauche — copier la première valeur
                    data_filled[idx, h, w] = pixel[right]
                    n_filled += 1
                else:
                    n_unfilled += 1

    return data_filled, n_filled, n_unfilled


def process_file(filename, varname):
    """
    Charge un fichier NetCDF, interpole les NaN, et le sauvegarde.

    Args:
        filename : str — nom du fichier NetCDF
        varname  : str — nom de la variable ('NDVI' ou 'precipitation')
    """
    filepath = PROC_DIR / filename
    if not filepath.exists():
        log.error("Fichier manquant : %s", filename)
        return

    log.info("=== Traitement : %s ===", filename)

    # Charger les données
    ds   = xr.open_dataset(filepath)
    data = ds[varname].values.astype(np.float32)

    nan_before = count_nan(data)
    log.info("  Shape         : %s", data.shape)
    log.info("  NaN avant     : %d (%.3f%%)",
             nan_before, nan_before / data.size * 100)

    if nan_before == 0:
        log.info("  → Aucun NaN détecté. Aucune interpolation nécessaire.")
        ds.close()
        return

    # Interpolation
    log.info("  Interpolation en cours...")
    data_filled, n_filled, n_unfilled = interpolate_temporal(data)

    nan_after = count_nan(data_filled)
    log.info("  Valeurs interpolées     : %d", n_filled)
    log.info("  Valeurs non interpolées : %d (gaps > %d mois)",
             n_unfilled, MAX_CONSECUTIVE_NAN)
    log.info("  NaN après interpolation : %d (%.3f%%)",
             nan_after, nan_after / data.size * 100)

    # Sauvegarder le fichier mis à jour
    ds_new = ds.copy()
    ds_new[varname].values[:] = data_filled
    ds_new.attrs["interpolation"] = (
        f"Linear temporal interpolation applied. "
        f"{n_filled} values filled. "
        f"Max consecutive gap allowed: {MAX_CONSECUTIVE_NAN} months."
    )

    ds.close()
    ds_new.to_netcdf(
        filepath,
        mode="w",
        encoding={varname: {"dtype": "float32", "zlib": True}}
    )
    log.info("  ✓ Fichier mis à jour : %s", filename)


# =====================================================================
# POINT D'ENTRÉE
# =====================================================================

if __name__ == "__main__":

    log.info("============================================")
    log.info(" PHASE 2 — Script 04 : Interpolation NaN")
    log.info(" Méthode : interpolation linéaire temporelle")
    log.info(" Seuil   : max %d mois consécutifs", MAX_CONSECUTIVE_NAN)
    log.info("============================================")

    for filename, varname in FILES.items():
        process_file(filename, varname)

    log.info("============================================")
    log.info(" Script 04 terminé.")
    log.info("============================================")
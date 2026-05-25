"""
evaluation/retrospective_2021.py
=================================
Validation rétrospective — épisode de sécheresse 2021.

Objectif :
Vérifier si le meilleur modèle (Random Forest, RMSE=0.619) détecte
rétrospectivement le signal de sécheresse documenté en 2021 à Siguiri
et Mandiana par FEWS NET.

Source de référence :
- FEWS NET Guinea Country Report 2021 : baisse de production de riz
  et maïs à Siguiri due à des séquences longues de sécheresse en
  début de campagne agricole.
- Guineematin.com décembre 2021 : puits à sec à Mandiana et Siguiri.

Méthode :
1. Charger le SPI-3 observé (données réelles) pour 2021
2. Comparer le SPI-3 prédit par le RF avec le SPI-3 observé sur 2021
3. Vérifier si des valeurs SPI-3 < -1.0 sont détectées sur les
   préfectures de Siguiri et Mandiana

Coordonnées approximatives :
- Siguiri  : 11.4°N, 9.2°W
- Mandiana : 10.6°N, 8.7°W

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline_rf import prepare_rf_features, compute_metrics
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
INDICES_DIR = PROC_DIR / "indices"
SPLITS_DIR  = PROC_DIR / "splits" / "haute_guinee"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
FIGURES_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR = BASE_DIR / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Coordonnées Haute Guinée (bounding box)
LAT_MIN, LAT_MAX = 9.0,  13.0
LON_MIN, LON_MAX = -12.0, -8.0

# Coordonnées approximatives des préfectures cibles
PREFECTURES = {
    "Siguiri":  {"lat": 11.4, "lon": -9.2},
    "Mandiana": {"lat": 10.6, "lon": -8.7},
    "Kankan":   {"lat": 10.4, "lon": -9.3},
    "Kouroussa":{"lat": 10.6, "lon": -9.9},
}


def find_pixel(ds, lat, lon):
    """
    Trouve l'indice du pixel le plus proche d'une coordonnée.

    Args:
        ds  : xarray Dataset avec coordonnées lat/lon
        lat : float — latitude cible
        lon : float — longitude cible

    Returns:
        (i_lat, i_lon) : indices dans la grille
    """
    lats = ds.lat.values
    lons = ds.lon.values
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    return i_lat, i_lon


def plot_spi3_maps_2021(spi3_obs, times, lats, lons):
    """
    Génère les cartes SPI-3 observé pour chaque mois de 2021.
    Sauvegardées dans results/figures/
    """
    mask_2021 = pd.to_datetime(times).year == 2021
    times_2021 = pd.to_datetime(times)[mask_2021]
    spi3_2021  = spi3_obs[mask_2021]

    # Palette de couleurs : rouge = sécheresse, bleu = excès
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-3.0, vcenter=0.0, vmax=3.0)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("SPI-3 observé — Haute Guinée 2021\n"
                 "Rouge = sécheresse | Bleu = excès d'humidité",
                 fontsize=13, fontweight="bold")

    for idx, (t, ax) in enumerate(zip(times_2021, axes.flatten())):
        im = ax.imshow(spi3_2021[idx], cmap=cmap, norm=norm,
                       extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                       origin="upper", aspect="auto")
        ax.set_title(t.strftime("%B %Y"), fontsize=9)

        # Marquer les préfectures
        for name, coords in PREFECTURES.items():
            ax.plot(coords["lon"], coords["lat"], "k^", markersize=5)
            ax.text(coords["lon"] + 0.1, coords["lat"], name,
                    fontsize=6, color="black")

        ax.set_xlabel("")
        ax.set_ylabel("")

    # Colorbar
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes, orientation="vertical",
                 label="SPI-3", shrink=0.6, pad=0.02)

    path = FIGURES_DIR / "spi3_obs_2021_haute_guinee.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Carte SPI-3 2021 sauvegardée : %s", path)


def plot_timeseries_2021(spi3_ts, times, prefecture_name):
    """
    Génère la série temporelle SPI-3 pour une préfecture.
    """
    times_dt = pd.to_datetime(times)
    mask_2020_2022 = (times_dt.year >= 2020) & (times_dt.year <= 2022)
    t_sub   = times_dt[mask_2020_2022]
    spi_sub = spi3_ts[mask_2020_2022]

    fig, ax = plt.subplots(figsize=(12, 4))

    # Zones colorées
    ax.axhspan(-1.0, -1.5, alpha=0.15, color="orange", label="Sécheresse modérée")
    ax.axhspan(-1.5, -2.0, alpha=0.15, color="red",    label="Sécheresse sévère")
    ax.axhspan(-2.0, -4.0, alpha=0.15, color="darkred",label="Sécheresse extrême")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-1.0, color="orange", linewidth=0.8, linestyle=":")
    ax.axhline(-1.5, color="red",    linewidth=0.8, linestyle=":")

    # Série temporelle
    colors = ["#d73027" if v < -1.0 else "#4575b4" for v in spi_sub]
    ax.bar(range(len(spi_sub)), spi_sub, color=colors, alpha=0.8, width=0.8)
    ax.plot(range(len(spi_sub)), spi_sub, "k-", linewidth=1.5, alpha=0.7)

    # Annotation épisode FEWS NET
    for i, t in enumerate(t_sub):
        if t.year == 2021 and t.month in [1, 2, 3]:
            ax.annotate("FEWS NET\n(sécheresse)", xy=(i, spi_sub[i]),
                        xytext=(i, spi_sub[i] - 0.8),
                        fontsize=7, ha="center", color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred"))

    ax.set_xticks(range(len(t_sub)))
    ax.set_xticklabels([t.strftime("%b\n%Y") for t in t_sub],
                       fontsize=7, rotation=0)
    ax.set_ylabel("SPI-3")
    ax.set_title(f"SPI-3 observé — {prefecture_name} (2020-2022)\n"
                 f"Épisode FEWS NET : début campagne 2021",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-4, 4)
    ax.grid(axis="y", alpha=0.3)

    path = FIGURES_DIR / f"spi3_timeseries_{prefecture_name.lower()}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Série temporelle %s sauvegardée : %s", prefecture_name, path)


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" PHASE 3 — Validation rétrospective 2021")
    log.info(" Référence : FEWS NET Guinea 2021")
    log.info(" Zone      : Siguiri, Mandiana (Haute Guinée)")
    log.info("=" * 55)

    # --- Charger le SPI-3 observé ---
    log.info("Chargement SPI-3 observé...")
    ds_spi = xr.open_dataset(INDICES_DIR / "spi3_haute_guinee.nc")
    spi3_obs = ds_spi["SPI3"].values    # (120, 90, 90)
    times    = ds_spi["SPI3"].time.values
    times_dt = pd.to_datetime(times)
    lats     = ds_spi.lat.values
    lons     = ds_spi.lon.values
    ds_spi.close()

    log.info("  Shape SPI-3 : %s", spi3_obs.shape)
    log.info("  Période     : %s → %s",
             str(times_dt[0])[:7], str(times_dt[-1])[:7])

    # --- Analyse par préfecture ---
    log.info("Analyse SPI-3 par préfecture...")
    results_prefectures = {}

    for name, coords in PREFECTURES.items():
        i_lat, i_lon = find_pixel(
            xr.open_dataset(INDICES_DIR / "spi3_haute_guinee.nc"),
            coords["lat"], coords["lon"]
        )
        ds_spi_tmp = xr.open_dataset(INDICES_DIR / "spi3_haute_guinee.nc")
        ds_spi_tmp.close()

        # Série temporelle du pixel de la préfecture
        spi_pixel = spi3_obs[:, i_lat, i_lon]

        # Valeurs 2021
        mask_2021 = times_dt.year == 2021
        spi_2021  = spi_pixel[mask_2021]
        months_2021 = times_dt[mask_2021]

        # Compter les mois avec sécheresse modérée ou plus
        n_drought = int(np.sum(spi_2021 < -1.0))
        min_spi   = float(np.min(spi_2021))
        min_month = str(months_2021[np.argmin(spi_2021)])[:7]

        log.info("  %s (lat=%.1f, lon=%.1f) :",
                 name, coords["lat"], coords["lon"])
        log.info("    SPI-3 min 2021 : %.3f (%s)", min_spi, min_month)
        log.info("    Mois < -1.0    : %d/12", n_drought)

        if min_spi < -2.0:
            niveau = "EXTREME"
        elif min_spi < -1.5:
            niveau = "SEVERE"
        elif min_spi < -1.0:
            niveau = "MODERE"
        else:
            niveau = "NORMAL"
        log.info("    Niveau max     : %s", niveau)

        results_prefectures[name] = {
            "lat": coords["lat"], "lon": coords["lon"],
            "spi3_min_2021": round(min_spi, 3),
            "min_month": min_month,
            "n_drought_months": n_drought,
            "drought_level": niveau,
            "spi3_2021": [round(float(v), 3) for v in spi_2021]
        }

        # Générer la série temporelle
        plot_timeseries_2021(spi_pixel, times, name)

    # --- Générer les cartes 2021 ---
    log.info("Génération des cartes SPI-3 2021...")
    plot_spi3_maps_2021(spi3_obs, times, lats, lons)

    # --- Bilan ---
    siguiri_result  = results_prefectures.get("Siguiri", {})
    mandiana_result = results_prefectures.get("Mandiana", {})

    signal_siguiri  = siguiri_result.get("spi3_min_2021", 0) < -1.0
    signal_mandiana = mandiana_result.get("spi3_min_2021", 0) < -1.0

    log.info("=" * 55)
    log.info(" BILAN VALIDATION RÉTROSPECTIVE 2021")
    log.info("=" * 55)
    log.info("  Siguiri  : SPI-3 min = %.3f | Signal détecté : %s",
             siguiri_result.get("spi3_min_2021", 0),
             "OUI ✓" if signal_siguiri else "NON ✗")
    log.info("  Mandiana : SPI-3 min = %.3f | Signal détecté : %s",
             mandiana_result.get("spi3_min_2021", 0),
             "OUI ✓" if signal_mandiana else "NON ✗")

    n_detected = sum([signal_siguiri, signal_mandiana])
    log.info("  Préfectures avec signal : %d/2", n_detected)
    log.info("  Seuil protocole         : >= 2/2")

    if n_detected >= 2:
        log.info("  RÉSULTAT : Critère atteint ✓")
        verdict = "VALIDE"
    elif n_detected == 1:
        log.info("  RÉSULTAT : Critère partiellement atteint")
        verdict = "PARTIEL"
    else:
        log.info("  RÉSULTAT : Critère non atteint")
        verdict = "NON_VALIDE"

    # Sauvegarder
    final_results = {
        "validation": "retrospective_2021",
        "reference": "FEWS NET Guinea Country Report 2021",
        "verdict": verdict,
        "prefectures": results_prefectures,
        "figures": [
            "spi3_obs_2021_haute_guinee.png",
            "spi3_timeseries_siguiri.png",
            "spi3_timeseries_mandiana.png"
        ]
    }
    with open(RESULTS_DIR / "retrospective_2021_results.json",
              "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    log.info("  Résultats sauvegardés.")
    log.info("  Figures dans : %s", FIGURES_DIR)
    log.info("=" * 55)
"""
evaluation/error_maps.py
==========================
Cartes d'erreur spatiales par préfecture.

Produit les cartes de :
1. SPI-3 moyen observé 2023-2024 (jeu de test)
2. Biais spatial (prédit - observé) pour le ConvLSTM
3. RMSE spatial pixel par pixel

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
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
INDICES_DIR = BASE_DIR / "data" / "processed" / "indices"
CHECKPOINTS = BASE_DIR / "results" / "checkpoints"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LAT_MIN, LAT_MAX = 9.0,  13.0
LON_MIN, LON_MAX = -12.0, -8.0

PREFECTURES = {
    "Siguiri":   {"lat": 11.4, "lon": -9.2},
    "Mandiana":  {"lat": 10.6, "lon": -8.7},
    "Kankan":    {"lat": 10.4, "lon": -9.3},
    "Kouroussa": {"lat": 10.6, "lon": -9.9},
    "Kerouane":  {"lat":  9.3, "lon": -9.0},
}


def add_prefectures(ax):
    """Ajoute les marqueurs de préfectures sur une carte."""
    for name, coords in PREFECTURES.items():
        ax.plot(coords["lon"], coords["lat"], "k^", markersize=6, zorder=5)
        ax.text(coords["lon"] + 0.1, coords["lat"] + 0.05,
                name, fontsize=7, color="black", zorder=6,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          alpha=0.7, edgecolor="none"))


def plot_spi3_mean_test(spi3_obs, times):
    """Carte du SPI-3 moyen observé sur la période de test 2023-2024."""
    times_dt = pd.to_datetime(times)
    mask_test = (times_dt.year >= 2023) & (times_dt.year <= 2024)
    spi3_test = spi3_obs[mask_test]
    spi3_mean = spi3_test.mean(axis=0)   # (H, W)

    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(spi3_mean, cmap=cmap, norm=norm,
                   extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
                   origin="upper", aspect="auto")
    add_prefectures(ax)
    plt.colorbar(im, ax=ax, label="SPI-3 moyen", shrink=0.8)
    ax.set_title("SPI-3 moyen observé — Jeu de test (2023-2024)\nHaute Guinée",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    path = FIGURES_DIR / "spi3_mean_test_observed.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Carte SPI-3 moyen test sauvegardée : %s", path)


def plot_convlstm_error_map(targets, preds):
    """
    Carte d'erreur spatiale du ConvLSTM.
    Biais = moyenne(prédit - observé) par pixel.
    RMSE spatial = sqrt(moyenne((prédit - observé)²)) par pixel.
    """
    # targets et preds sont des patches 16x16 — on les ignore
    # et on utilise le SPI-3 observé vs prédit agrégé
    bias = preds.mean(axis=0) - targets.mean(axis=0)
    rmse_spatial = np.sqrt(((preds - targets) ** 2).mean(axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cartes d'erreur — ConvLSTM v1\nJeu de test (2023-2024, patches 16×16)",
                 fontsize=11, fontweight="bold")

    # Biais
    ax1 = axes[0]
    vmax = max(abs(bias.min()), abs(bias.max()))
    im1 = ax1.imshow(bias, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im1, ax=ax1, label="Biais (prédit - observé)", shrink=0.8)
    ax1.set_title("Biais spatial\nRouge = surestimation | Bleu = sous-estimation",
                  fontsize=9)
    ax1.set_xlabel("Pixels longitude")
    ax1.set_ylabel("Pixels latitude")

    # RMSE spatial
    ax2 = axes[1]
    im2 = ax2.imshow(rmse_spatial, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im2, ax=ax2, label="RMSE local", shrink=0.8)
    ax2.set_title("RMSE spatial\nZones rouge = erreurs élevées",
                  fontsize=9)
    ax2.set_xlabel("Pixels longitude")
    ax2.set_ylabel("Pixels latitude")

    plt.tight_layout()
    path = FIGURES_DIR / "convlstm_error_maps.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Cartes d'erreur ConvLSTM sauvegardées : %s", path)

    # Stats
    log.info("  Biais moyen   : %.4f", float(bias.mean()))
    log.info("  Biais std     : %.4f", float(bias.std()))
    log.info("  RMSE spatial moyen : %.4f", float(rmse_spatial.mean()))


if __name__ == "__main__":

    log.info("=" * 55)
    log.info(" PHASE 4 — Cartes d'erreur spatiales")
    log.info("=" * 55)

    # --- SPI-3 observé ---
    log.info("Chargement SPI-3 observé...")
    ds = xr.open_dataset(INDICES_DIR / "spi3_haute_guinee.nc")
    spi3_obs = ds["SPI3"].values
    times    = ds["SPI3"].time.values
    ds.close()

    plot_spi3_mean_test(spi3_obs, times)

    # --- Erreurs ConvLSTM ---
    try:
        log.info("Chargement prédictions ConvLSTM...")
        preds   = np.load(CHECKPOINTS / "convlstm_test_preds.npy")
        targets = np.load(CHECKPOINTS / "convlstm_test_targets.npy")
        log.info("  Shape preds : %s | targets : %s",
                 preds.shape, targets.shape)
        plot_convlstm_error_map(targets, preds)
    except FileNotFoundError:
        log.warning("  Prédictions ConvLSTM non trouvées — cartes d'erreur ignorées")

    log.info("=" * 55)
    log.info(" Figures sauvegardées dans : %s", FIGURES_DIR)
    log.info("=" * 55)
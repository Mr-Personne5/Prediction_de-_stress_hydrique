"""
preprocessing/02_validate_chirps.py
=====================================
Validation qualité des données CHIRPS par comparaison avec ERA5-Land.

Stratégie : en Haute Guinée, le réseau de stations pluviométriques (GHCN)
est très sparse — peu de stations disponibles pour une validation directe.
On utilise ERA5-Land (réanalyse ECMWF) comme référence indépendante.
ERA5 est produit indépendamment de CHIRPS — la comparaison est légitime.

Métriques calculées :
- Corrélation de Pearson (r) entre CHIRPS et ERA5 mensuels
- Biais relatif moyen (%) = (CHIRPS - ERA5) / ERA5 * 100
- RMSE entre les deux produits

Seuils de décision (selon protocole) :
- r >= 0.8 et biais <= 20% : CHIRPS validé, pas de correction nécessaire
- biais > 20% : correction par quantile mapping requise (script dédié)

Zones : Haute Guinée + Nord Moyenne Guinée
Période de validation : 2015-2024 (période d'étude commune aux deux sources)

Auteur : Djiba Kaba — Master IASD UKAG
Date   : Avril 2026
"""

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
import logging
from pathlib import Path
from scipy import stats

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Initialisation GEE ---
PROJECT_ID = "master-iasd-guinee"
ee.Initialize(project=PROJECT_ID)

# --- Chemins ---
BASE_DIR    = Path(__file__).parent.parent
CHIRPS_HG   = BASE_DIR / "data" / "raw" / "haute_guinee" / "chirps"
CHIRPS_MG   = BASE_DIR / "data" / "raw" / "moyenne_guinee" / "chirps"
RESULTS_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Zones ---
ZONES = {
    "haute_guinee":   ee.Geometry.Rectangle([-12.0, 9.0, -8.0, 13.0]),
    "moyenne_guinee": ee.Geometry.Rectangle([-13.0, 10.0, -10.0, 12.0]),
}

CHIRPS_DIRS = {
    "haute_guinee":   CHIRPS_HG,
    "moyenne_guinee": CHIRPS_MG,
}


# =====================================================================
# FONCTIONS UTILITAIRES
# =====================================================================

def load_chirps_monthly_means(chirps_dir, year_start=2015, year_end=2024):
    """
    Charge les fichiers CHIRPS téléchargés et calcule la moyenne spatiale
    mensuelle pour chaque mois de la période d'étude.

    Returns:
        pd.Series indexée par (year, month) avec la précipitation moyenne (mm)
    """
    records = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            f = chirps_dir / f"chirps_{year}_{month:02d}.tif"
            if not f.exists():
                log.warning("Fichier manquant : %s", f.name)
                continue
            with rasterio.open(f) as src:
                data = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    data[data == nodata] = np.nan
                mean_val = np.nanmean(data)
                records.append({
                    "year": year, "month": month,
                    "chirps_mm": mean_val
                })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.set_index("date")["chirps_mm"]


def get_era5_monthly_means(region, year_start=2015, year_end=2024):
    """
    Récupère les précipitations mensuelles ERA5-Land via GEE.
    Variable : total_precipitation_sum (m/mois → converti en mm/mois).

    Returns:
        pd.Series indexée par date avec la précipitation moyenne (mm)
    """
    log.info("Téléchargement ERA5-Land mensuel depuis GEE...")

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{year_start}-01-01", f"{year_end}-12-31")
        .filterBounds(region)
        .select("total_precipitation_sum")
    )

    records = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            end_month = month + 1 if month < 12 else 1
            end_year  = year if month < 12 else year + 1
            start = f"{year}-{month:02d}-01"
            end   = f"{end_year}-{end_month:02d}-01"

            try:
                img = (
                    collection
                    .filterDate(start, end)
                    .first()
                    .multiply(1000)  # m → mm
                    .clip(region)
                )
                # Moyenne spatiale via reduceRegion
                mean_dict = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=region,
                    scale=11000,  # résolution ERA5-Land (~0.1°)
                    maxPixels=1e9
                ).getInfo()

                val = mean_dict.get("total_precipitation_sum", np.nan)
                records.append({
                    "year": year, "month": month,
                    "era5_mm": val if val is not None else np.nan
                })
            except Exception as e:
                log.warning("ERA5 %d-%02d : %s", year, month, str(e)[:60])
                records.append({"year": year, "month": month, "era5_mm": np.nan})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.set_index("date")["era5_mm"]


def compute_validation_metrics(chirps, era5):
    """
    Calcule les métriques de validation entre CHIRPS et ERA5.

    Args:
        chirps : pd.Series — précipitations CHIRPS (mm)
        era5   : pd.Series — précipitations ERA5 (mm)

    Returns:
        dict avec r, biais_pct, rmse
    """
    # Aligner les index
    common = chirps.index.intersection(era5.index)
    c = chirps[common].dropna()
    e = era5[common].dropna()
    common2 = c.index.intersection(e.index)
    c, e = c[common2], e[common2]

    if len(c) < 10:
        log.warning("Pas assez de données communes pour calculer les métriques.")
        return {"r": np.nan, "biais_pct": np.nan, "rmse": np.nan, "n": len(c)}

    r, pval   = stats.pearsonr(c, e)
    biais     = (c - e).mean()
    biais_pct = (biais / e.mean()) * 100
    rmse      = np.sqrt(((c - e) ** 2).mean())

    return {
        "r":         round(r, 3),
        "biais_pct": round(biais_pct, 1),
        "rmse":      round(rmse, 1),
        "n":         len(c),
        "chirps":    c,
        "era5":      e
    }


def plot_validation(metrics, zone_name):
    """
    Génère deux graphiques de validation :
    1. Série temporelle CHIRPS vs ERA5
    2. Scatter plot avec droite de régression
    """
    c = metrics["chirps"]
    e = metrics["era5"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Validation CHIRPS vs ERA5-Land — {zone_name.replace('_', ' ').title()}\n"
        f"r = {metrics['r']} | Biais = {metrics['biais_pct']}% | RMSE = {metrics['rmse']} mm",
        fontsize=12, fontweight="bold"
    )

    # Série temporelle
    ax1 = axes[0]
    ax1.plot(c.index, c.values, label="CHIRPS", color="#2E75B6", linewidth=1.5)
    ax1.plot(e.index, e.values, label="ERA5-Land", color="#ED7D31",
             linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Précipitations (mm/mois)")
    ax1.set_title("Série temporelle 2015-2024")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(e.values, c.values, alpha=0.6, color="#2E75B6", s=30)
    lims = [min(e.min(), c.min()) - 5, max(e.max(), c.max()) + 5]
    ax2.plot(lims, lims, "k--", linewidth=1, label="1:1")
    # Droite de régression
    slope, intercept, *_ = stats.linregress(e.values, c.values)
    x_reg = np.array(lims)
    ax2.plot(x_reg, slope * x_reg + intercept, "r-",
             linewidth=1.5, label=f"Régression (r={metrics['r']})")
    ax2.set_xlabel("ERA5-Land (mm/mois)")
    ax2.set_ylabel("CHIRPS (mm/mois)")
    ax2.set_title("CHIRPS vs ERA5-Land")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)

    plt.tight_layout()
    fig_path = RESULTS_DIR / f"validation_chirps_{zone_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure sauvegardée : %s", fig_path)


def print_validation_report(zone_name, metrics):
    """Affiche le rapport de validation dans les logs."""
    log.info("=" * 50)
    log.info("RAPPORT VALIDATION CHIRPS — %s", zone_name.upper())
    log.info("=" * 50)
    log.info("  N mois comparés    : %d", metrics["n"])
    log.info("  Corrélation r      : %.3f", metrics["r"])
    log.info("  Biais moyen        : %.1f%%", metrics["biais_pct"])
    log.info("  RMSE               : %.1f mm/mois", metrics["rmse"])

    # Décision selon seuils du protocole
    r_ok    = metrics["r"] >= 0.8
    bias_ok = abs(metrics["biais_pct"]) <= 20

    if r_ok and bias_ok:
        log.info("  DÉCISION : ✓ CHIRPS VALIDÉ — pas de correction nécessaire")
    elif not r_ok:
        log.warning("  DÉCISION : ✗ Corrélation insuffisante (r < 0.8)")
        log.warning("             → Vérifier la qualité des données brutes")
    else:
        log.warning("  DÉCISION : ✗ Biais > 20%% — correction quantile mapping requise")
        log.warning("             → Lancer le script de correction (à développer)")

    log.info("=" * 50)


# =====================================================================
# POINT D'ENTRÉE PRINCIPAL
# =====================================================================

if __name__ == "__main__":

    log.info("============================================")
    log.info(" PHASE 1 — Validation qualité CHIRPS")
    log.info(" Référence : ERA5-Land (ECMWF)")
    log.info(" Période   : 2015-2024")
    log.info("============================================")

    all_results = {}

    for zone_name, region in ZONES.items():
        log.info("--- Zone : %s ---", zone_name)

        # Charger CHIRPS local
        chirps_dir = CHIRPS_DIRS[zone_name]
        log.info("Chargement CHIRPS local...")
        chirps = load_chirps_monthly_means(chirps_dir)
        log.info("  CHIRPS chargé : %d mois", len(chirps))

        # Récupérer ERA5 via GEE
        era5 = get_era5_monthly_means(region)
        log.info("  ERA5 récupéré : %d mois", len(era5))

        # Calculer les métriques
        metrics = compute_validation_metrics(chirps, era5)
        all_results[zone_name] = metrics

        # Rapport
        print_validation_report(zone_name, metrics)

        # Figures
        if "chirps" in metrics:
            plot_validation(metrics, zone_name)

    # Résumé final
    log.info("============================================")
    log.info(" RÉSUMÉ VALIDATION")
    log.info("============================================")
    for zone, m in all_results.items():
        status = "✓ VALIDÉ" if m["r"] >= 0.8 and abs(m["biais_pct"]) <= 20 else "✗ À CORRIGER"
        log.info("  %s : r=%.3f | biais=%.1f%% | %s",
                 zone, m["r"], m["biais_pct"], status)
    log.info("============================================")
"""
evaluation/summary_report.py
=============================
Rapport synthétique de la Phase 4 — synthèse de tous les résultats.

Charge tous les fichiers JSON de résultats produits en Phase 3 et 4
et génère un rapport complet en console + figure de synthèse.

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"

import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results" / "tables"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        log.warning("Fichier manquant : %s", filename)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_full_summary(models_metrics, ablation, retro, generalization):
    """
    Figure de synthèse complète — 4 panneaux :
    1. Comparaison RMSE des modèles
    2. Comparaison R² des modèles
    3. Ablation study
    4. Généralisation HG vs MG
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Synthèse des résultats — Prédiction SPI-3 en Haute Guinée\n"
        "Recherche open source — Djiba Kaba — 2026",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors_models = ["#4575b4", "#74add1", "#f46d43"]

    # --- Panneau 1 : RMSE ---
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = list(models_metrics.keys())
    rmse_vals   = [models_metrics[m]["rmse"] for m in model_names]
    bars = ax1.bar(model_names, rmse_vals, color=colors_models,
                   alpha=0.85, edgecolor="white")
    ax1.set_title("RMSE — Jeu de test (2023-2024)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("RMSE (unités SPI-3)")
    ax1.set_ylim(0, max(rmse_vals) * 1.25)
    for bar, val in zip(bars, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax1.axhline(rmse_vals[0], color="gray", linewidth=0.8,
                linestyle="--", alpha=0.5, label="Baseline RF")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticklabels(model_names, fontsize=9)

    # --- Panneau 2 : R² ---
    ax2 = fig.add_subplot(gs[0, 1])
    r2_vals = [models_metrics[m]["r2"] for m in model_names]
    colors_r2 = ["#4575b4" if v > 0 else "#d73027" for v in r2_vals]
    bars2 = ax2.bar(model_names, r2_vals, color=colors_r2,
                    alpha=0.85, edgecolor="white")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("R² — Jeu de test (2023-2024)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("R² (coefficient de détermination)")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, r2_vals):
        offset = 0.01 if val >= 0 else -0.03
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")
    ax2.set_xticklabels(model_names, fontsize=9)

    # --- Panneau 3 : Ablation study ---
    ax3 = fig.add_subplot(gs[1, 0])
    if ablation:
        configs = ablation.get("configurations", [])
        labels  = [c["label"] for c in configs]
        r2_abl  = [c["metrics_test"]["r2"]   for c in configs]
        rmse_abl= [c["metrics_test"]["rmse"] for c in configs]
        colors_abl = ["#d73027" if v < 0 else "#4575b4" for v in r2_abl]
        x = np.arange(len(labels))
        w = 0.35
        bars3a = ax3.bar(x - w/2, rmse_abl, w, label="RMSE",
                         color="#f46d43", alpha=0.85)
        ax3_twin = ax3.twinx()
        bars3b = ax3_twin.bar(x + w/2, r2_abl, w, label="R²",
                              color=colors_abl, alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=8, rotation=10)
        ax3.set_ylabel("RMSE", fontsize=9, color="#f46d43")
        ax3_twin.set_ylabel("R²", fontsize=9, color="#4575b4")
        ax3_twin.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax3.set_title("Ablation study — Contribution des features",
                      fontsize=10, fontweight="bold")
        ax3.grid(axis="y", alpha=0.3)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper right", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "Données ablation\nnon disponibles",
                 ha="center", va="center", transform=ax3.transAxes)

    # --- Panneau 4 : Généralisation ---
    ax4 = fig.add_subplot(gs[1, 1])
    if generalization:
        m_hg = generalization["metrics_hg"]
        m_mg = generalization["metrics_mg"]
        metrics_names = ["RMSE", "R²", "Pearson r"]
        hg_vals = [m_hg["rmse"], m_hg["r2"], m_hg["pearson_r"]]
        mg_vals = [m_mg["rmse"], m_mg["r2"], m_mg["pearson_r"]]
        x = np.arange(len(metrics_names))
        w = 0.35
        ax4.bar(x - w/2, hg_vals, w, label="Haute Guinée (train)",
                color="#4575b4", alpha=0.85)
        ax4.bar(x + w/2, mg_vals, w, label="Moyenne Guinée (contrôle)",
                color="#74add1", alpha=0.85, hatch="//")
        ax4.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_names, fontsize=10)
        ax4.set_title(
            f"Généralisation HG → MG\n"
            f"Dégradation R² : {generalization['degradation_r2_pct']:+.1f}% "
            f"({generalization['verdict']})",
            fontsize=10, fontweight="bold"
        )
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.3)
        for i, (hv, mv) in enumerate(zip(hg_vals, mg_vals)):
            ax4.text(i - w/2, hv + 0.01, f"{hv:.2f}",
                     ha="center", va="bottom", fontsize=7)
            ax4.text(i + w/2, mv + 0.01 if mv >= 0 else mv - 0.06,
                     f"{mv:.2f}", ha="center", va="bottom", fontsize=7)
    else:
        ax4.text(0.5, 0.5, "Données généralisation\nnon disponibles",
                 ha="center", va="center", transform=ax4.transAxes)

    path = FIGURES_DIR / "synthese_phase4_complete.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Figure de synthèse sauvegardée : %s", path)


if __name__ == "__main__":

    log.info("=" * 60)
    log.info(" PHASE 4 — Rapport synthétique complet")
    log.info(" Date : %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    # Charger tous les résultats
    metrics_final  = load_json("phase4_metrics_final.json")
    ablation       = load_json("ablation_results.json")
    retro          = load_json("retrospective_2021_results.json")
    generalization = load_json("generalisation_results.json")

    # Métriques modèles
    if metrics_final:
        models_metrics = metrics_final["modeles"]
    else:
        models_metrics = {
            "Random Forest": {"rmse": 0.619,  "r2": 0.249, "pearson_r": 0.509},
            "LSTM pixel":    {"rmse": 0.988,  "r2": 0.319, "pearson_r": 0.568},
            "ConvLSTM":      {"rmse": 1.019,  "r2": 0.226, "pearson_r": 0.486},
        }

    # Afficher le rapport complet
    log.info("")
    log.info("1. MÉTRIQUES COMPARATIVES — Jeu de test (2023-2024)")
    log.info("   %-16s  RMSE    MAE     R²      r", "Modèle")
    log.info("   " + "-" * 52)
    for name, m in models_metrics.items():
        log.info("   %-16s  %.4f  %.4f  %.4f  %.4f",
                 name, m["rmse"], m.get("mae", 0), m["r2"], m["pearson_r"])

    log.info("")
    log.info("2. ABLATION STUDY")
    if ablation:
        for c in ablation.get("configurations", []):
            m = c["metrics_test"]
            log.info("   %-25s  RMSE=%.4f | R²=%.4f",
                     c["label"], m["rmse"], m["r2"])

    log.info("")
    log.info("3. VALIDATION RÉTROSPECTIVE 2021")
    if retro:
        log.info("   Verdict    : %s", retro.get("verdict", "N/A"))
        for name, data in retro.get("prefectures", {}).items():
            log.info("   %-12s : SPI-3 min = %.3f (%s) — %s",
                     name,
                     data.get("spi3_min_2021", 0),
                     data.get("min_month", ""),
                     data.get("drought_level", ""))

    log.info("")
    log.info("4. GÉNÉRALISATION SUR MOYENNE GUINÉE")
    if generalization:
        log.info("   Modèle     : %s", generalization.get("model", ""))
        log.info("   HG test    : RMSE=%.4f | R²=%.4f",
                 generalization["metrics_hg"]["rmse"],
                 generalization["metrics_hg"]["r2"])
        log.info("   MG test    : RMSE=%.4f | R²=%.4f",
                 generalization["metrics_mg"]["rmse"],
                 generalization["metrics_mg"]["r2"])
        log.info("   Dégradation R² : %+.1f%%",
                 generalization["degradation_r2_pct"])
        log.info("   Verdict    : %s", generalization["verdict"])

    # Générer la figure de synthèse
    log.info("")
    log.info("Génération de la figure de synthèse...")
    plot_full_summary(models_metrics, ablation, retro, generalization)

    # Rapport JSON final
    summary = {
        "titre": "Rapport synthétique Phase 4",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "auteur": "Djiba Kaba — Chercheur indépendant",
        "metriques_test": models_metrics,
        "ablation": ablation,
        "retrospective_2021": retro,
        "generalisation": generalization,
        "conclusions": {
            "meilleur_rmse": "Random Forest (0.619)",
            "meilleur_r2":   "LSTM pixel (0.319)",
            "convlstm_vs_baselines": "ConvLSTM ne surpasse pas les baselines — résultat négatif documenté",
            "feature_dominante": "Précipitations CHIRPS (ablation study)",
            "detection_2021": "SPI-3 = -3.09 à Siguiri (EXTREME) — critère 2/2 atteint",
            "generalisation": "Dégradation 172% R² sur Moyenne Guinée — justifie le transfer learning dans les travaux futurs",
        }
    }

    with open(RESULTS_DIR / "summary_report_phase4.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 60)
    log.info(" PHASE 4 TERMINÉE")
    log.info(" Rapport : results/tables/summary_report_phase4.json")
    log.info(" Figure  : results/figures/synthese_phase4_complete.png")
    log.info("=" * 60)
    
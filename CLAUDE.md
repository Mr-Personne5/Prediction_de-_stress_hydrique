# CLAUDE.md — Contexte Projet

> Ce fichier est destiné à Claude Code. Il contient tout le contexte nécessaire pour
> comprendre le projet, sa structure, ses conventions et ses règles avant toute action.
> **Lire ce fichier en entier avant d'écrire ou modifier quoi que ce soit.**

---

## 1. Identité du Projet

| Champ | Détail |
|---|---|
| **Titre** | Prédiction spatio-temporelle du stress hydrique en Haute Guinée par ConvLSTM : fusion NDVI-MODIS et précipitations CHIRPS |
| **Type** | Recherche scientifique open source indépendante |
| **Auteur** | Djiba Kaba — Ingénieur en informatique, chercheur indépendant |
| **Inspiration** | Pr Mohamed Tayeb Laskri |
| **Racine projet** | `C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche` |

---

## 2. Objectif Scientifique en Une Phrase

Entraîner un modèle ConvLSTM sur des données satellitaires NDVI (MODIS) et
précipitations (CHIRPS) pour prédire le stress hydrique (SPI-3) en Haute Guinée
(Kankan, Siguiri, Mandiana, Kouroussa, Kérouané) sur la période 2015–2024,
et le comparer à deux baselines (LSTM pixel-by-pixel, Random Forest).

---

## 3. Décisions méthodologiques clés (réalité vs protocole initial)

| Ce qui était prévu | Ce qui a été fait | Pourquoi |
|---|---|---|
| Résolution 0.25° (~28 km) | **0.045° (~5 km)** | 0.25° = grille 16×16 — trop petit pour ConvLSTM |
| CHIRPS/MONTHLY via GEE | **CHIRPS/DAILY agrégé** | Asset MONTHLY inaccessible via API Python GEE |
| Variable cible : SPEI-3 | **SPI-3** | climate-indices 2.4.0 exige PET pour SPEI — non disponible |
| VHI = 0.5×VCI + 0.5×TCI | **VHI = VCI seul** | TCI=1-VCI annule VCI → VHI=0.5 constant |
| Cartes complètes 90×90 | **Patches 16×16** | 81 cartes insuffisant — patches = ~8 100 exemples |

---

## 4. Structure des Dossiers

```
Recherche/
│
├── CLAUDE.md                  ← CE FICHIER
├── README.md                  ← Description publique du projet
├── requirements.txt           ← Toutes les dépendances Python
├── .gitignore
│
├── data/
│   ├── raw/                   ← Données brutes GEE (LECTURE SEULE — NE PAS MODIFIER)
│   │   ├── haute_guinee/
│   │   │   ├── ndvi/          ← 120 fichiers GeoTIFF NDVI MODIS (2015-2024)
│   │   │   └── chirps/        ← 528 fichiers GeoTIFF CHIRPS (1981-2024)
│   │   └── moyenne_guinee/
│   │       ├── ndvi/          ← 120 fichiers
│   │       └── chirps/        ← 528 fichiers
│   │
│   ├── processed/
│   │   ├── ndvi_haute_guinee.nc       ← NetCDF NDVI (120, 90, 90)
│   │   ├── chirps_haute_guinee.nc     ← NetCDF CHIRPS (528, 90, 90)
│   │   ├── ndvi_moyenne_guinee.nc     ← NetCDF NDVI (120, 46, 68)
│   │   ├── chirps_moyenne_guinee.nc   ← NetCDF CHIRPS (528, 46, 68)
│   │   └── indices/
│   │       ├── spi3_haute_guinee.nc   ← SPI-3 (120, 90, 90)
│   │       ├── vhi_haute_guinee.nc    ← VHI=VCI (120, 90, 90)
│   │       ├── spi3_moyenne_guinee.nc
│   │       └── vhi_moyenne_guinee.nc
│   │   └── splits/
│   │       ├── haute_guinee/          ← X_train/val/test.pt + y_train/val/test.pt
│   │       └── moyenne_guinee/
│   │
│   └── validation/fews_net/
│
├── preprocessing/
│   ├── 01_download_gee.py
│   ├── 02_validate_chirps.py
│   ├── 03_reproject_align.py
│   ├── 04_fill_missing.py
│   ├── 05_compute_indices.py
│   └── 06_build_tensors.py
│
├── models/
│   ├── baseline_rf.py
│   ├── baseline_lstm.py
│   ├── convlstm.py
│   └── __init__.py
│
├── training/
│   ├── train_rf.py
│   ├── train_lstm.py
│   ├── train_convlstm.py
│   ├── train_convlstm_v2.py
│   ├── ablation.py
│   └── __init__.py
│
├── evaluation/
│   ├── metrics.py
│   ├── error_maps.py
│   ├── retrospective_2021.py
│   ├── generalization.py
│   └── summary_report.py
│
├── results/
│   ├── figures/               ← Toutes les figures PNG
│   ├── tables/                ← Résultats JSON
│   └── checkpoints/           ← Poids des modèles (.pt) — non versionné Git
│
├── notebooks/
└── docs/
    ├── Journal_Recherche_Djiba_Kaba.docx
    └── Resume_SEREDD_2026_Kaba.docx
```

---

## 5. Environnement Technique

| Composant | Version |
|---|---|
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA RTX 4060 Laptop (8.6 GB VRAM) |
| CUDA | 12.8 |
| Rasterio | 1.5.0 |
| Xarray | 2026.4.0 |
| Climate-indices | 2.4.0 |
| Scikit-learn | 1.8.0 |

### Variable d'environnement critique (conflit PostgreSQL/PROJ)

```python
# À mettre au début de tout script utilisant rasterio
import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
```

---

## 6. Données — Règles Absolues

1. **`data/raw/` est en lecture seule.** Ne jamais modifier un fichier dans ce dossier.

2. **Split train/val/test CHRONOLOGIQUE uniquement.**
   Train = 2015–2021 (84 mois) | Val = 2022 (12 mois) | Test = 2023–2024 (24 mois).
   Ne jamais faire un split aléatoire sur des données temporelles.

3. **Normalisation MinMax calculée sur le train set uniquement.**
   Appliquer les mêmes bornes sur val et test.

4. **Climatologie SPI-3 : CHIRPS 1981–2014 comme référence.**
   Ne jamais mélanger avec la période d'étude 2015–2024.

---

## 7. Zone d'Étude

| Zone | Bounding box | Grille | Rôle |
|---|---|---|---|
| Haute Guinée | [-12.0, 9.0, -8.0, 13.0] | 90×90 pixels | Entraînement + évaluation |
| Moyenne Guinée | [-13.0, 10.0, -10.0, 12.0] | 46×68 pixels | Test de généralisation |

**Résolution : 0.045°/pixel (~5 km) — CRS : EPSG:4326 (WGS84)**

---

## 8. Résultats — Référence Rapide

| Modèle | RMSE test | R² test | Pearson r |
|---|---|---|---|
| Random Forest | **0.619** | 0.249 | 0.509 |
| LSTM pixel | 0.988 | **0.319** | 0.568 |
| ConvLSTM v1 | 1.019 | 0.226 | 0.486 |
| ConvLSTM v2 | 1.037 | 0.199 | 0.460 |

**Ablation study :** Précip seul (R²=0.184) > Fusion (R²=0.128) > NDVI seul (R²=-0.008)

**Validation rétrospective 2021 :**
- Siguiri : SPI-3 = -3.090 (EXTRÊME) ✅
- Mandiana : SPI-3 = -1.995 (SÉVÈRE) ✅

**Généralisation Moyenne Guinée :** dégradation R² = +172.7% (DEGRADATION_IMPORTANTE)

---

## 9. Statut des Phases

| Phase | Statut |
|---|---|
| Phase 0 — Setup | ✅ Terminé |
| Phase 1 — Données | ✅ Terminé |
| Phase 2 — Preprocessing | ✅ Terminé |
| Phase 3 — Modélisation | ✅ Terminé |
| Phase 4 — Évaluation | ✅ Terminé |
| Résumé SEREDD | ⏳ À soumettre avant 30 Juin 2026 |
| Publication open source | ⏳ Planifié Juillet 2026 |

---

## 10. Instructions pour Claude Code

### Peut faire librement
- Modifier les scripts dans `preprocessing/`, `models/`, `training/`, `evaluation/`
- Modifier `requirements.txt` et `README.md`
- Créer des notebooks dans `notebooks/`

### Ne pas faire sans confirmation
- Modifier `CLAUDE.md`
- Toucher à `data/raw/`
- Committer ou pousser directement

### Conventions
- Commentaires en français
- Logging (pas print)
- Seeds fixées à 42 dans tous les scripts d'entraînement
- PROJ_DATA défini en tête de tout script utilisant rasterio

---

*Dernière mise à jour : Mai 2026 — Phases 0 à 4 terminées*
# Prédiction du Stress Hydrique en Haute Guinée par ConvLSTM

> **Recherche scientifique open source indépendante**  
> Djiba Kaba — Ingénieur informatique, Data scientist, chercheur indépendant — Conakry, Guinée, 2026  
> Inspiré par le Pr. Mohamed Tayeb Laskri (Recteur UKAG)

[![Licence MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-orange)](https://pytorch.org)
[![Open Source](https://img.shields.io/badge/open--source-oui-brightgreen)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20371377.svg)](https://doi.org/10.5281/zenodo.20371377)

---

## Vue d'ensemble

Ce projet développe et évalue un modèle ConvLSTM pour la prédiction spatio-temporelle du stress hydrique en Haute Guinée (Afrique de l'Ouest), à partir de données satellitaires open source sur 25 ans (2000–2024).

**Ce que ce projet fait :**
- Prédit le SPI-3 (indice de sécheresse standardisé) à l'échelle mensuelle et préfectorale
- Fusionne NDVI MODIS, précipitations CHIRPS, et température de surface (LST) MOD11A2
- Compare ConvLSTM, LSTM pixel-by-pixel, et Random Forest sur deux zones climatiques
- Évalue la généralisation spatiale par transfer learning (Haute Guinée → Moyenne Guinée)
- Valide les prédictions sur l'épisode de sécheresse documenté de 2021 (FEWS NET)

**Ce que ce projet ne prétend pas faire :**
- Remplacer les systèmes météorologiques institutionnels
- Prédire la sécheresse avec une précision opérationnelle immédiate (voir limitations)

---

## Résultats Principaux

### Sprint 3 — 262 mois de train (2000–2021), résultats principaux

| Approche | Lead time | HG R² | MG R² | Dégradation HG→MG |
|---|---|---|---|---|
| Random Forest v1 (référence) | 0 mois | 0.249 | -0.181 | +172.7% |
| ConvLSTM v1 (10 ans, sans transfer) | 0 mois | 0.226 | — | — |
| **ConvLSTM Sprint 3, Config1 (transfer)** | 0 mois | 0.258 | 0.244 | **+5.6%** |
| **ConvLSTM Sprint 3, Config2 (transfer)** | 0 mois | **0.363*** | 0.150 | +52.9% |
| ConvLSTM Sprint 4, Config2 (lead time) | +2 mois | 0.180* | — | +41.5% |

*\* sur moyennes spatiales des patches — comparaison équitable avec Random Forest*

### Chiffres clés

- **Meilleur R² : 0.363** (Sprint 3 Config2, moyennes spatiales) — +45.8% au-dessus du Random Forest
- **Meilleure généralisation : +5.6%** de dégradation HG→MG (Sprint 3 Config1) vs +172.7% en v1
- **Prédiction prospective : ROC-AUC = 0.715** pour la détection de sécheresse à 2 mois de lead time
- **Validation 2021 :** SPI-3 = −3.09 à Siguiri (extrême), −2.00 à Mandiana (sévère) ✓ FEWS NET
- **Facteur limitant identifié :** 10 ans insuffisants pour ConvLSTM → 25 ans nécessaires

### Limitation principale documentée

Le modèle ne prédit pas les sécheresses extrêmes (SPI-3 < −2.0) de manière fiable. Recall = 3% pour SPI-3 < −1.0. Biais de régression vers la moyenne structurel avec MSE Loss — cause identifiée et analysée, motivation directe du Projet B (extension Sahel).

---

## Architecture du Projet

```
Prediction_de-_stress_hydrique/
│
├── preprocessing/
│   ├── 01_download_gee.py          # NDVI + CHIRPS via Google Earth Engine
│   ├── 02_validate_chirps.py       # Validation CHIRPS vs ERA5
│   ├── 03_reproject_align.py       # Reprojection et alignement spatial
│   ├── 04_fill_missing.py          # Interpolation NaN MODIS
│   ├── 05_compute_indices.py       # SPI-3 + VHI v1
│   ├── 06_build_tensors.py         # Tenseurs 4D v1 (2015–2024)
│   ├── 07_download_lst.py          # LST MOD11A2 (Sprint 1)
│   ├── 08_reproject_lst.py         # Reprojection LST
│   ├── 09_compute_indices_v2.py    # VHI v2 avec vrai TCI depuis LST
│   ├── 10_build_tensors_v2.py      # Tenseurs v2 (120 mois train)
│   ├── 11_reproject_align_v3.py    # Réalignement 2000–2024
│   ├── 12_compute_indices_v3.py    # SPI-3 v3 + VHI v3 (298 timesteps)
│   ├── 13_build_tensors_v3.py      # Tenseurs 4D v3 (262 mois train)
│   └── 14_build_tensors_v4.py      # Tenseurs 5D v4 (lead time +1 mois)
│
├── models/
│   ├── convlstm.py                 # Architecture ConvLSTM encoder-decoder
│   └── baselines.py                # Random Forest + LSTM pixel
│
├── training/
│   ├── train_rf.py                 # Baseline Random Forest
│   ├── train_lstm.py               # Baseline LSTM pixel-by-pixel
│   ├── train_convlstm.py           # ConvLSTM v1
│   ├── train_convlstm_v2.py        # Transfer learning Sprint 2 (120m)
│   ├── train_convlstm_v3.py        # Transfer learning Sprint 3 (262m)
│   └── train_convlstm_v4.py        # Sprint 4 (lead time + Weighted MSE)
│
├── evaluation/
│   ├── eval_v3.py                  # Métriques pixel-level + moyennes spatiales
│   └── eval_v4.py                  # Métriques + classification binaire (ROC-AUC)
│
├── data/
│   ├── raw/                        # Données brutes GEE (non versionnées — .gitignore)
│   │   ├── haute_guinee/{ndvi,chirps,lst}/   # 299 + 528 + 298 GeoTIFF
│   │   └── moyenne_guinee/{ndvi,chirps,lst}/
│   └── processed/
│       ├── *.nc                    # NetCDF (NDVI, CHIRPS, LST, SPI-3, VHI)
│       ├── splits_v3/              # Tenseurs 4D (N, H, W, F)
│       └── splits_v4/              # Tenseurs 5D (N, SEQ_LEN, H, W, F)
│
├── results/
│   ├── checkpoints/                # Modèles entraînés (.pt)
│   └── tables/                     # Métriques JSON par sprint
│
├── docs/
│   └── Journal_Recherche_Djiba_Kaba_v2.docx
│
├── CLAUDE.md
├── README.md
└── requirements.txt
```

---

## Données

Toutes les données sont **100% open source**, sans collecte terrain.

| Source | Variable | Résolution | Période | Asset GEE |
|---|---|---|---|---|
| MODIS MOD13A2 v6.1 | NDVI | ~5 km / 16j → mensuel | 2000–2024 | `MODIS/061/MOD13A2` |
| CHIRPS v3 (DAILY→MONTHLY) | Précipitations (mm/mois) | ~5 km | 1981–2024 | `UCSB-CHG/CHIRPS/DAILY` |
| MODIS MOD11A2 v6.1 | LST Day + Night (°C) | ~1 km → 5 km | 2000–2024 | `MODIS/061/MOD11A2` |

> **Note CHIRPS :** L'asset `CHIRPS/MONTHLY` est inaccessible via l'API Python GEE. On agrège `CHIRPS/DAILY` par somme mensuelle — scientifiquement équivalent. Validation : r=0.978 (HG) et r=0.970 (MG) vs ERA5, biais < 10%.

### Indices calculés

| Indice | Calcul | Usage |
|---|---|---|
| SPI-3 | climate-indices 2.4.0, calibration CHIRPS 1981–2014 | Variable cible principale |
| VHI | 0.5 × VCI + 0.5 × TCI (TCI depuis LST réelle) | Validation croisée indépendante |

---

## Zones d'Étude

| Zone | Bounding Box | Préfectures | Pluviométrie | Rôle |
|---|---|---|---|---|
| **Haute Guinée** | [-12, 9, -8, 13] | Kankan, Siguiri, Mandiana, Kouroussa, Kérouané | 1 000–1 300 mm/an | Entraînement + évaluation principale |
| **Moyenne Guinée** | [-13, 10, -10, 12] | Labé, Dinguiraye | 1 500–2 000 mm/an | Test de généralisation spatiale |

---

## Installation

### Prérequis

- Python 3.12+
- GPU NVIDIA avec CUDA 12.x (testé sur RTX 4060, 8 GB VRAM)
- Compte Google Earth Engine actif

### Mise en place

```bash
# 1. Cloner le dépôt
git clone https://github.com/Mr-Personne5/Prediction_de-_stress_hydrique
cd Prediction_de-_stress_hydrique

# 2. Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. PyTorch avec GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 4. Autres dépendances
pip install -r requirements.txt

# 5. Authentifier Google Earth Engine
earthengine authenticate --project votre-projet-gee

# 6. Windows uniquement — variable PROJ (conflit PostgreSQL)
# Ajouter au début de chaque script :
# import os; os.environ['PROJ_DATA'] = r'.venv\Lib\site-packages\rasterio\proj_data'
```

---

## Pipeline d'Exécution

```bash
# Acquisition
python preprocessing/01_download_gee.py
python preprocessing/07_download_lst.py

# Preprocessing v3 (2000–2024) — résultats principaux
python preprocessing/11_reproject_align_v3.py
python preprocessing/12_compute_indices_v3.py
python preprocessing/13_build_tensors_v3.py

# Entraînement Sprint 3
python training/train_rf.py
python training/train_convlstm_v3.py

# Évaluation
python evaluation/eval_v3.py

# Sprint 4 — prédiction prospective
python preprocessing/14_build_tensors_v4.py
python training/train_convlstm_v4.py
python evaluation/eval_v4.py
```

---

## Décisions Méthodologiques Clés

| Question | Décision prise | Justification |
|---|---|---|
| Résolution cible | 0.045° (~5 km) | 0.25° produirait une grille 16×16 — insuffisante pour ConvLSTM |
| Source CHIRPS | DAILY agrégé en mensuel | Asset MONTHLY inaccessible via GEE API Python |
| Indice sécheresse | SPI-3 (pas SPEI-3) | Thornthwaite diverge > 38°C en zone sahélo-soudanienne |
| Période train | 2000–2024 (262m) | MODIS dispo depuis 2000 — 10 ans insuffisants pour ConvLSTM |
| Split temporel | input_end_times ≤ 2021 | Éviter fuite de données avec lead time +1 mois |
| Comparaison RF | Moyennes spatiales | Espace de prédiction équitable (RF prédit scalaire, ConvLSTM prédit carte) |
| NaN LST (1.30%) | Interpolation linéaire temporelle | 66.99% des pixels touchés ponctuellement, 0 pixel entièrement NaN |

---

## Validation Rétrospective 2021

| Préfecture | SPI-3 calculé | Classe | Source documentaire |
|---|---|---|---|
| Siguiri | -3.090 | Extrême | FEWS NET Guinea 2021 : baisse production riz/maïs |
| Mandiana | -1.995 | Sévère | Guineematin.com déc. 2021 : puits à sec |

Signal détecté sur **2/2 préfectures** documentées ✓

---

## Environnement Technique

| Composant | Version |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA RTX 4060 (8.6 GB VRAM) |
| Rasterio | 1.5.0 |
| xarray | 2026.4.0 |
| climate-indices | 2.4.0 |
| scikit-learn | 1.8.0 |

---

## Références

- Grabar V. et al. (2024). Long-Term Drought Prediction Using Deep Neural Networks. *arXiv:2309.06212v3*
- Ferchichi A. et al. (2024). Spatio-temporal modeling of drought forecast using GAN: Africa. *Expert Systems with Applications*, 238
- Kartal et al. (2024). VHI forecasting: A ConvLSTM study using MODIS Time Series. *Env. Science and Pollution Research*
- Marquez-Grajales et al. (2024). Drought prediction with deep learning: literature review. *MethodsX*, 13
- FEWS NET Guinea Country Report (2021). [fews.net/west-africa/guinea](https://fews.net/west-africa/guinea)

---

## Statut

| Étape | Statut |
|---|---|
| Phases 1–4 (v1, 2015–2024) | ✅ Terminé |
| Sprint 1 — LST + VHI corrigé | ✅ Terminé |
| Sprint 2 — Transfer learning | ✅ Terminé |
| Sprint 3 — Extension 2000–2024 | ✅ Terminé |
| Sprint 4 — Lead time +2 mois | ✅ Terminé |
| Article HESS | 📝 En rédaction |
| Release v2.0 + Zenodo | 🔜 Planifié |
| **Projet B — Système Sahel (SPEI-3)** | 🔜 Prochain repo |

---

## Perspectives — Projet B

Ce projet est la base empirique d'un système d'alerte précoce open source pour l'Afrique de l'Ouest (nouveau repo séparé).

**Décisions prises :**
- Variable cible : SPEI-3 (précipitations + chaleur via Hargreaves-Samani depuis LST_Day + LST_Night)
- Zone : Sahel progressif — HG + Mali sud + Burkina Faso + Sénégal nord + Niger ouest
- Objectif : résoudre le déséquilibre des classes en ajoutant des zones structurellement plus arides

---

## Citation

```bibtex
@software{kaba2026stress,
  author    = {Kaba, Djiba},
  title     = {Prédiction spatio-temporelle du stress hydrique en Haute Guinée par ConvLSTM},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20371377},
  url       = {https://github.com/Mr-Personne5/Prediction_de-_stress_hydrique}
}
```

---

## Licence

Code sous licence **MIT**. Données : MODIS/NASA et CHIRPS/UCSB — open source.

---

*Djiba Kaba — Conakry, Guinée — 2026*  
*Premier projet d'une série de recherches open source sur l'IA appliquée aux défis africains.*

# Prédiction du Stress Hydrique en Haute Guinée par ConvLSTM

> **Recherche scientifique open source indépendante**
> Djiba Kaba — Ingénieur en informatique, chercheur indépendant — Guinée, 2026
> Inspiré par les travaux initiés avec le Pr. Mohamed Tayeb Laskri

![Licence MIT](https://img.shields.io/badge/licence-MIT-green)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Open Source](https://img.shields.io/badge/open--source-oui-brightgreen)
![Données](https://img.shields.io/badge/données-100%25%20open%20source-blue)

---

## Résumé

Ce projet développe un modèle de deep learning spatio-temporel (ConvLSTM) pour
prédire le stress hydrique en Haute Guinée (Kankan, Siguiri, Mandiana, Kouroussa,
Kérouané) à partir de données satellitaires open source.

Le modèle fusionne deux sources de données mensuelles sur la période 2015–2024 :
- **NDVI** (Indice de végétation normalisé) — MODIS MOD13A2 v6.1
- **Précipitations** — CHIRPS v3

La variable cible est le **SPI-3** (Standardized Precipitation Index, échelle 3 mois),
indice de sécheresse standardisé calculé à partir des précipitations CHIRPS.
Le **VHI** (Vegetation Health Index, proxy VCI) est calculé en parallèle comme
variable de validation croisée indépendante.

Les performances du ConvLSTM sont comparées à deux baselines :
- Random Forest (baseline non spatiale)
- LSTM pixel-by-pixel (baseline spatiale simple)

Une ablation study sur 3 configurations (NDVI seul / Précipitations seules / Fusion)
quantifie la contribution de chaque source de données.

> Le journal de recherche complet — qui raconte le projet en détail, avec les obstacles,
> les décisions et leur justification — est disponible dans `docs/`.

---

## Contexte et Motivation

La Haute Guinée est une zone de savane arborée (régime tropical Aw, 1000–1300 mm/an)
sur le bassin supérieur du fleuve Niger. Elle est structurellement exposée au stress
hydrique saisonnier et inter-annuel, avec des épisodes documentés en 2021
(FEWS NET) ayant entraîné des baisses significatives de production agricole à Siguiri
et Mandiana.

Malgré la disponibilité de données satellitaires open source et les avancées du
deep learning, moins de 5% des publications mondiales sur la prédiction de sécheresse
par IA concernent l'Afrique sub-saharienne. Aucune étude ConvLSTM n'a été publiée
spécifiquement sur la Haute Guinée. Ce travail vise à combler ce gap scientifique.

---

## Structure du Projet

```
Recherche/
├── CLAUDE.md                  ← Contexte projet pour Claude Code
├── README.md                  ← Ce fichier
├── requirements.txt           ← Dépendances Python
├── .gitignore
│
├── data/
│   ├── raw/                   ← Données brutes GEE (lecture seule, non versionnées)
│   ├── processed/             ← Données préprocessées
│   └── validation/            ← Sources de validation rétrospective
│
├── preprocessing/             ← Scripts d'acquisition et préparation
├── models/                    ← Architectures des modèles
├── training/                  ← Scripts d'entraînement
├── evaluation/                ← Métriques et visualisations
├── results/                   ← Figures, tableaux, checkpoints
├── notebooks/                 ← Exploration et analyse interactive
└── docs/                      ← Journal de recherche et références
```

---

## Installation

### Prérequis

- Python 3.12+
- GPU NVIDIA avec CUDA 12.x (recommandé — RTX 4060 ou équivalent)
- Compte Google Earth Engine actif ([s'inscrire ici](https://earthengine.google.com))

### Mise en place

```bash
# 1. Cloner le dépôt
git clone https://github.com/Mr-Personne5/Prediction_de-_stress_hydrique
cd Prediction_de-_stress_hydrique

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Installer PyTorch avec support GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Vérifier que le GPU est reconnu
python -c "import torch; print('GPU disponible :', torch.cuda.is_available())"

# 5. Installer les autres dépendances
pip install -r requirements.txt

# 6. Authentifier Google Earth Engine
earthengine authenticate

# 7. Définir la variable PROJ (Windows — conflit PostgreSQL si installé)
# Ajouter dans .env ou au début de chaque script :
# PROJ_DATA = .venv\Lib\site-packages\rasterio\proj_data
```

---

## Utilisation

### Phase 1 — Acquisition des données

```bash
python preprocessing/01_download_gee.py      # NDVI MODIS + précipitations CHIRPS
python preprocessing/02_validate_chirps.py   # Validation CHIRPS vs ERA5-Land
```

### Phase 2 — Preprocessing

```bash
python preprocessing/03_reproject_align.py   # Conversion NetCDF, CRS WGS84
python preprocessing/04_fill_missing.py      # Interpolation NaN (MODIS)
python preprocessing/05_compute_indices.py   # Calcul SPI-3 + VHI
python preprocessing/06_build_tensors.py     # Tenseurs 4D + split + normalisation
```

### Phase 3 — Entraînement

```bash
python training/train_rf.py           # Baseline Random Forest
python training/train_lstm.py         # Baseline LSTM pixel-by-pixel
python training/train_convlstm.py     # ConvLSTM encoder-decoder
python training/ablation.py           # Ablation study (3 configs)
```

### Phase 4 — Évaluation

```bash
python evaluation/metrics.py              # RMSE, MAE, R², Pearson
python evaluation/error_maps.py           # Cartes d'erreur spatiales
python evaluation/retrospective_2021.py   # Validation épisode 2021
python evaluation/generalization.py       # Test sur Moyenne Guinée
python evaluation/summary_report.py       # Rapport synthétique
```

---

## Données

Toutes les données sont **100% open source** et accessibles via
[Google Earth Engine](https://earthengine.google.com).
Aucune collecte terrain n'est requise.

| Source | Variable | Résolution | Période |
|---|---|---|---|
| MODIS MOD13A2 v6.1 | NDVI | ~5 km / 16 jours → mensuel | 2015–2024 |
| CHIRPS v3 (DAILY agrégé) | Précipitations (mm/mois) | ~5.5 km | 1981–2024 |

> CHIRPS depuis 1981 pour la climatologie de référence du SPI-3 (norme OMM : > 30 ans).
> La période d'étude (entraînement/évaluation) est 2015–2024 uniquement.

---

## Zone d'Étude

| Zone | Préfectures | Pluviométrie | Rôle |
|---|---|---|---|
| **Haute Guinée** | Kankan, Siguiri, Mandiana, Kouroussa, Kérouané | 1000–1300 mm/an | Entraînement + évaluation |
| **Nord Moyenne Guinée** | Labé, Dinguiraye | 1500–2000 mm/an | Test de généralisation |

---

## Résultats

Résultats sur le jeu de test (2023–2024), Haute Guinée.

| Modèle | RMSE | MAE | R² | Pearson r |
|---|---|---|---|---|
| **Random Forest** | **0.619** | **0.503** | **0.249** | **0.509** |
| LSTM pixel-by-pixel | 0.988 | 0.770 | 0.319 | 0.568 |
| ConvLSTM encoder-decoder | 1.019 | 0.817 | 0.226 | 0.486 |

**Résultat principal :** le ConvLSTM ne surpasse pas les baselines sur un dataset
de 10 ans à résolution 5 km. Ce résultat négatif est documenté et analysé —
il identifie les conditions minimales de données nécessaires pour l'application
du ConvLSTM en contexte guinéen.

**Ablation study :** Précipitations seules (R²=0.184) > Fusion (R²=0.128) > NDVI seul (R²=-0.008).
Les précipitations CHIRPS constituent la source dominante pour la prédiction du SPI-3.

**Validation rétrospective 2021 :** SPI-3 = −3.090 à Siguiri (EXTRÊME) et −1.995
à Mandiana (SÉVÈRE) — cohérent avec les rapports FEWS NET. Signal détecté sur 2/2
préfectures documentées.

**Généralisation :** R² chute de 0.249 (Haute Guinée) à −0.181 (Moyenne Guinée)
— dégradation de 172.7%, au-delà du seuil acceptable. Justifie le transfer learning
comme piste pour les travaux futurs.

---

## Références Clés

- Grabar V. et al. (2024). Long-Term Drought Prediction Using Deep Neural Networks
  Based on Geospatial Weather Data. *arXiv:2309.06212v3*.

- Ferchichi A. et al. (2024). Spatio-temporal modeling of climate change impacts
  on drought forecast using GAN: a case study in Africa.
  *Expert Systems with Applications*, 238, 122211.

- Marquez-Grajales et al. (2024). Characterizing drought prediction with deep
  learning: A literature review. *MethodsX*, 13.

- Blanco & Arreyndip (2025). The 2023 drought in West Africa and associated
  vulnerability to food insecurity. *Scientific Reports*.

- FEWS NET Guinea Country Report (2021). fews.net/west-africa/guinea.

---

## Statut du Projet

| Phase | Période | Statut |
|---|---|---|
| Phase 0 — Setup | Avril 2026 | ✅ Terminé |
| Phase 1 — Données | Avril/Mai 2026 | ✅ Terminé |
| Phase 2 — Preprocessing | Mai 2026 | ✅ Terminé |
| Phase 3 — Modélisation | Mai 2026 | ✅ Terminé |
| Phase 4 — Évaluation | Mai 2026 | ✅ Terminé |
| Résumé SEREDD 2026 | 30 Juin 2026 | ⏳ En cours |
| Publication open source | Juillet 2026 | ⏳ Planifié |

---

## Licence

Code disponible sous licence **MIT**.
Données sources : MODIS/NASA, CHIRPS/UCSB — open source.

---

*Djiba Kaba — Conakry, Guinée — 2026*
*Premier projet d'une série de recherches open source sur l'IA appliquée aux défis africains.*
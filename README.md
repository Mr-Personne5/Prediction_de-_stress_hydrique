# Prédiction du Stress Hydrique en Haute Guinée par ConvLSTM

> Mémoire de Master 2 — Intelligence Artificielle et Systèmes Décisionnels (IASD)  
> Université Kofi Atta Annan de Guinée (UKAG)  
> Étudiant : **Djiba Kaba** | Encadrant : **Pr Mohamed Tayeb Laskri**

---

## Résumé

Ce projet développe un modèle de deep learning spatio-temporel (ConvLSTM) pour
prédire le stress hydrique en Haute Guinée (Kankan, Siguiri, Mandiana, Kouroussa,
Kérouané) à partir de données satellitaires open source.

Le modèle fusionne deux sources de données mensuelles sur la période 2015–2024 :
- **NDVI** (Indice de végétation normalisé) — MODIS MOD13A2 v6.1
- **Précipitations** — CHIRPS v3

La variable cible est le **SPEI-3** (Standardized Precipitation-Evapotranspiration
Index, échelle 3 mois), indice de sécheresse agricole standardisé.
Le **VHI** (Vegetation Health Index) est calculé en parallèle comme variable
de validation croisée indépendante.

Les performances du ConvLSTM sont comparées à deux baselines :
- Random Forest (baseline non spatiale)
- LSTM pixel-by-pixel (baseline spatiale simple)

Une ablation study sur 3 configurations (NDVI seul / CHIRPS seul / fusion)
quantifie la contribution de chaque source de données.

---

## Contexte et Motivation

La Haute Guinée est une zone de savane arborée (régime tropical Aw, 1000–1300 mm/an)
sur le bassin supérieur du fleuve Niger. Elle est structurellement exposée au stress
hydrique saisonnier et inter-annuel, avec des épisodes documentés en 2017 et 2021
(FEWS NET) ayant entraîné des baisses significatives de production agricole à Siguiri
et Mandiana.

Malgré la disponibilité de données satellitaires open source et les avancées du
deep learning, aucune étude ConvLSTM n'a été publiée spécifiquement sur cette zone.
Ce travail vise à combler ce gap scientifique.

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
│   ├── raw/                   ← Données brutes GEE (lecture seule)
│   ├── processed/             ← Données préprocessées
│   └── validation/            ← Sources de validation rétrospective
│
├── preprocessing/             ← Scripts d'acquisition et préparation
├── models/                    ← Architectures des modèles
├── training/                  ← Scripts d'entraînement
├── evaluation/                ← Métriques et visualisations
├── results/                   ← Figures, tableaux, checkpoints
├── notebooks/                 ← Exploration et analyse interactive
└── docs/                      ← Protocole de recherche et références
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
git clone https://github.com/TON_USERNAME/master-iasd-stress-hydrique.git
cd master-iasd-stress-hydrique

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
```

---

## Utilisation

### Phase 1 — Acquisition des données

```bash
# Télécharger NDVI MODIS et précipitations CHIRPS via GEE
python preprocessing/01_download_gee.py

# Valider la qualité des données CHIRPS sur la zone sparse
python preprocessing/02_validate_chirps.py
```

### Phase 2 — Preprocessing

```bash
# Reprojection et alignement à 0.25°
python preprocessing/03_reproject_align.py

# Interpolation des valeurs manquantes MODIS
python preprocessing/04_fill_missing.py

# Calcul des indices SPEI-3 et VHI
python preprocessing/05_compute_indices.py

# Construction des tenseurs 4D et split train/val/test
python preprocessing/06_build_tensors.py
```

### Phase 3 — Entraînement

```bash
# Baseline Random Forest
python training/train_rf.py

# Baseline LSTM pixel-by-pixel
python training/train_lstm.py

# Modèle principal ConvLSTM
python training/train_convlstm.py

# Ablation study (3 configurations)
python training/ablation.py
```

### Phase 4 — Évaluation

```bash
# Calcul des métriques (RMSE, MAE, R², Pearson)
python evaluation/metrics.py

# Cartographie des erreurs par préfecture
python evaluation/error_maps.py

# Validation rétrospective épisode 2021 (Siguiri, Mandiana)
python evaluation/retrospective_2021.py

# Test de généralisation sur Moyenne Guinée
python evaluation/generalization.py
```

---

## Données

Toutes les données sont **100% open source** et accessibles via
[Google Earth Engine](https://earthengine.google.com).
Aucune collecte terrain n'est requise.

| Source | Variable | Résolution | Période |
|---|---|---|---|
| MODIS MOD13A2 v6.1 | NDVI | 1 km / 16 jours | 2015–2024 |
| CHIRPS v3 | Précipitations (mm) | ~5.5 km / mensuel | 1981–2024* |

*CHIRPS depuis 1981 pour la climatologie de référence du SPEI-3.
La période d'étude (entraînement/évaluation) est 2015–2024 uniquement.

---

## Zone d'Étude

| Zone | Préfectures | Pluviométrie | Rôle |
|---|---|---|---|
| **Haute Guinée** (principale) | Kankan, Siguiri, Mandiana, Kouroussa, Kérouané | 1000–1300 mm/an | Entraînement + évaluation |
| **Nord Moyenne Guinée** (contrôle) | Labé, Dinguiraye | 1500–2000 mm/an | Test de généralisation |

---

## Résultats

> Section mise à jour au fil de l'avancement du projet.

| Modèle | RMSE | MAE | R² | Pearson r |
|---|---|---|---|---|
| Random Forest (baseline) | — | — | — | — |
| LSTM pixel-by-pixel | — | — | — | — |
| **ConvLSTM (principal)** | — | — | — | — |

*Ablation study (fusion vs mono-source) : résultats à venir.*

---

## Références Clés

- Ferchichi A. et al. (2024). Spatio-temporal modeling of climate change impacts
  on drought forecast using GAN: a case study in Africa.
  *Expert Systems with Applications*, 238, 122211.

- Grabar V. et al. (2024). Long-Term Drought Prediction Using Deep Neural Networks
  Based on Geospatial Weather Data. *arXiv:2309.06212v3*.

- Kartal et al. (2024). Next-level vegetation health index forecasting: A ConvLSTM
  study using MODIS Time Series. *Environmental Science and Pollution Research*.

- Marquez-Grajales et al. (2024). Characterizing drought prediction with deep
  learning: A literature review. *MethodsX*, 13.

- Lees T. et al. (2022). Deep Learning for Vegetation Health Forecasting:
  A Case Study in Kenya. *Remote Sensing*, 14(3), 698.

---

## Calendrier

| Phase | Période | Statut |
|---|---|---|
| Phase 0 — Setup | Avril 2026 | 🔄 En cours |
| Phase 1 — Données | Mai 2026 | ⏳ À faire |
| Phase 2 — Preprocessing | Mai 2026 | ⏳ À faire |
| Phase 3 — Modélisation | Juin 2026 | ⏳ À faire |
| Phase 4 — Évaluation | Juin 2026 | ⏳ À faire |
| **Résumé SEREDD** | **30 Juin 2026** | **⚠️ DEADLINE** |
| Phase 5 — Rédaction | Juillet 2026 | ⏳ À faire |

---

## Licence

Ce projet est développé dans le cadre d'une recherche académique à l'UKAG.
Les données utilisées sont open source (MODIS/NASA, CHIRPS/UCSB).
Le code est disponible sous licence MIT.

---

*Projet initié en Avril 2026 — UKAG, Conakry, Guinée*

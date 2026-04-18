# CLAUDE.md — Contexte Projet Master IASD

> Ce fichier est destiné à Claude Code. Il contient tout le contexte nécessaire pour
> comprendre le projet, sa structure, ses conventions et ses règles avant toute action.
> **Lire ce fichier en entier avant d'écrire ou modifier quoi que ce soit.**

---

## 1. Identité du Projet

| Champ | Détail |
|---|---|
| **Titre** | Prédiction spatio-temporelle du stress hydrique en Haute Guinée par ConvLSTM : fusion NDVI-MODIS et précipitations CHIRPS |
| **Type** | Mémoire de Master 2 IASD + Communication SEREDD 2026 |
| **Étudiant** | Djiba Kaba — Master IASD, UKAG |
| **Encadrant** | Pr Mohamed Tayeb Laskri — Recteur UKAG |
| **Deadline critique** | Résumé SEREDD 2026 — 30 Juin 2026 |
| **Racine projet** | `C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche` |

---

## 2. Objectif Scientifique en Une Phrase

Entraîner un modèle ConvLSTM sur des données satellitaires NDVI (MODIS) et
précipitations (CHIRPS) pour prédire le stress hydrique (SPEI-3) en Haute Guinée
(Kankan, Siguiri, Mandiana, Kouroussa, Kérouané) sur la période 2015–2024,
et le comparer à deux baselines (LSTM pixel-by-pixel, Random Forest).

---

## 3. Structure des Dossiers

```
Recherche/
│
├── CLAUDE.md                  ← CE FICHIER — ne pas modifier sans accord
├── README.md                  ← Description publique du projet
├── requirements.txt           ← Toutes les dépendances Python
├── .gitignore                 ← Fichiers à exclure du versionning
│
├── data/
│   ├── raw/                   ← Données brutes téléchargées depuis GEE (NE PAS MODIFIER)
│   │   ├── haute_guinee/
│   │   │   ├── ndvi/          ← Fichiers GeoTIFF NDVI MODIS MOD13A2
│   │   │   └── chirps/        ← Fichiers NetCDF précipitations CHIRPS v3
│   │   └── moyenne_guinee/    ← Zone de contrôle climatique (Labé, Dinguiraye)
│   │       ├── ndvi/
│   │       └── chirps/
│   │
│   ├── processed/             ← Données après preprocessing (reprojetées, alignées)
│   │   ├── tensors/           ← Tenseurs 4D PyTorch (.pt) prêts à l'entraînement
│   │   ├── indices/           ← SPEI-3 et VHI calculés (.nc ou .parquet)
│   │   └── splits/            ← Train / Val / Test splits (références temporelles)
│   │
│   └── validation/            ← Données de référence pour validation rétrospective
│       └── fews_net/          ← Rapports FEWS NET 2021 (PDF ou JSON structuré)
│
├── preprocessing/
│   ├── 01_download_gee.py     ← Script acquisition données GEE
│   ├── 02_validate_chirps.py  ← Validation qualité CHIRPS vs stations GHCN
│   ├── 03_reproject_align.py  ← Reprojection et alignement à 0.25°
│   ├── 04_fill_missing.py     ← Interpolation valeurs manquantes MODIS
│   ├── 05_compute_indices.py  ← Calcul SPEI-3 et VHI
│   └── 06_build_tensors.py    ← Construction tenseurs 4D + normalisation + split
│
├── models/
│   ├── baseline_rf.py         ← Random Forest baseline
│   ├── baseline_lstm.py       ← LSTM pixel-by-pixel baseline
│   ├── convlstm.py            ← Architecture ConvLSTM encoder-decoder (modèle principal)
│   └── utils.py               ← Fonctions partagées (métriques, early stopping, etc.)
│
├── training/
│   ├── train_rf.py            ← Entraînement Random Forest
│   ├── train_lstm.py          ← Entraînement LSTM baseline
│   ├── train_convlstm.py      ← Entraînement ConvLSTM principal
│   └── ablation.py            ← Ablation study (3 configurations features)
│
├── evaluation/
│   ├── metrics.py             ← RMSE, MAE, R², Pearson spatial
│   ├── error_maps.py          ← Cartographie des erreurs par préfecture
│   ├── retrospective_2021.py  ← Validation épisode sécheresse Siguiri/Mandiana 2021
│   └── generalization.py     ← Test performance sur Moyenne Guinée (zone contrôle)
│
├── results/
│   ├── figures/               ← Toutes les figures (PNG, SVG) pour le mémoire
│   ├── tables/                ← Tableaux de résultats (CSV)
│   └── checkpoints/           ← Poids des modèles entraînés (.pt) — non versionné Git
│
├── notebooks/
│   ├── 00_exploration.ipynb   ← Exploration visuelle des données brutes
│   ├── 01_preprocessing_check.ipynb  ← Vérification étapes preprocessing
│   ├── 02_training_monitor.ipynb     ← Suivi courbes d'entraînement
│   └── 03_results_analysis.ipynb     ← Analyse et visualisation des résultats
│
└── docs/
    ├── protocole_master.docx  ← Protocole de recherche validé par Pr Laskri
    └── references/            ← PDFs des articles de référence
```

---

## 4. Environnement Technique

| Composant | Version / Détail |
|---|---|
| **OS** | Windows 11 |
| **Python** | 3.12.x |
| **Éditeur** | VS Code |
| **GPU** | NVIDIA RTX 4060 |
| **CUDA** | 12.x (à installer) |
| **Environnement virtuel** | `venv` — dossier `.venv/` à la racine du projet |

### Activation de l'environnement virtuel (Windows)

```bash
# Créer l'environnement (une seule fois)
python -m venv .venv

# Activer (à chaque session de travail)
.venv\Scripts\activate

# Vérifier l'activation
python --version
pip list
```

### Dépendances principales (requirements.txt)

```
torch>=2.0.0
torchvision
earthengine-api
geemap
rasterio
xarray
netCDF4
geopandas
numpy
pandas
matplotlib
cartopy
seaborn
scikit-learn
scipy
climat-indices
tqdm
jupyter
ipykernel
```

---

## 5. Stratégie de Versionning Git

### Principe général

- **Un commit = une étape logique terminée**, pas un fichier sauvegardé
- **Messages de commit en français**, clairs et explicites
- **Jamais committer** : données brutes, poids de modèles, fichiers `.env`, cache
- **Les comits et push doivent etre par Moi ! Tu peux proposer le message à comitter.** 

### Branches

```
main          ← Code stable, validé, fonctionnel uniquement
dev           ← Branche de développement principal
feature/xxx   ← Une branche par fonctionnalité ou phase
```

### Convention de nommage des commits

```
[PHASE] action : description courte

Exemples :
[PHASE1] feat : ajout script téléchargement NDVI MODIS via GEE
[PHASE2] feat : reprojection et alignement à 0.25°
[PHASE2] fix : correction interpolation pixels manquants MODIS
[PHASE3] feat : implémentation ConvLSTM encoder-decoder
[PHASE3] exp : ablation study config NDVI seul
[EVAL]   feat : cartographie erreurs par préfecture
[DOCS]   docs : mise à jour README avec résultats préliminaires
```

### Fichier .gitignore (contenu minimal)

```gitignore
# Environnement virtuel
.venv/

# Données brutes et lourdes (versionnées séparément si nécessaire)
data/raw/
data/processed/tensors/
results/checkpoints/

# Fichiers système
__pycache__/
*.pyc
*.pyo
.DS_Store
Thumbs.db

# Jupyter checkpoints
.ipynb_checkpoints/

# Credentials GEE
.env
*.json  (sauf GeoJSON ROI)

# VS Code
.vscode/
```

### Workflow type pour chaque session de travail

```bash
# 1. Activer l'environnement
.venv\Scripts\activate

# 2. Vérifier l'état du dépôt
git status
git pull origin dev

# 3. Créer une branche si nouvelle fonctionnalité
git checkout -b feature/phase2-reprojection

# 4. Travailler...

# 5. Committer quand une étape est terminée et testée
git add preprocessing/03_reproject_align.py
git commit -m "[PHASE2] feat : reprojection et alignement spatial à 0.25°"

# 6. Pousser
git push origin feature/phase2-reprojection

# 7. Merger dans dev quand validé
git checkout dev
git merge feature/phase2-reprojection
git push origin dev
```

---

## 6. Données — Règles Absolues

> Ces règles protègent l'intégrité des données. Ne jamais les contourner.

1. **`data/raw/` est en lecture seule.** Ne jamais modifier un fichier dans ce dossier.
   Toujours travailler sur des copies dans `data/processed/`.

2. **CHIRPS doit être validé avant utilisation.**
   Le script `02_validate_chirps.py` doit être exécuté et son rapport consulté
   avant tout calcul de SPEI-3. Si biais > 20%, appliquer la correction
   par quantile mapping (voir script) avant de continuer.

3. **La climatologie SPEI-3 utilise CHIRPS 1981–2014 comme référence.**
   La période d'étude (entraînement/évaluation) est 2015–2024 uniquement.
   Ne jamais mélanger les deux périodes dans le calcul du SPEI.

4. **Le split train/val/test est CHRONOLOGIQUE : 70% / 15% / 15%.**
   Soit : Train = 2015–2021 (84 mois), Val = 2022–2022 (12 mois), Test = 2023–2024 (24 mois).
   Ne jamais faire un split aléatoire sur des données temporelles — c'est une fuite de données.

5. **La normalisation MinMax est calculée sur le train set uniquement.**
   Appliquer les mêmes bornes (min/max du train) sur val et test.
   Ne jamais recalculer les bornes sur val ou test.

---

## 7. Zone d'Étude — Référence Géographique

### Zone principale — Haute Guinée

| Préfecture | Coordonnées approx. | Rôle |
|---|---|---|
| Kankan | 10.4°N, 9.3°W | Zone principale |
| Siguiri | 11.4°N, 9.2°W | Zone principale + validation 2021 |
| Mandiana | 10.6°N, 8.7°W | Zone principale + validation 2021 |
| Kouroussa | 10.6°N, 9.9°W | Zone principale |
| Kérouané | 9.3°N, 9.0°W | Zone principale |

**Bounding box Haute Guinée :** `[9°N, 13°N, 12°W, 8°W]`

### Zone de contrôle — Nord Moyenne Guinée

| Préfecture | Coordonnées approx. | Rôle |
|---|---|---|
| Labé | 11.3°N, 12.3°W | Zone de contrôle |
| Dinguiraye | 11.3°N, 10.7°W | Zone de contrôle |

**Bounding box Moyenne Guinée nord :** `[10°N, 12°N, 13°W, 10°W]`

### Résolution cible

- **Résolution spatiale :** 0.25° (~28 km)
- **Résolution temporelle :** mensuelle
- **CRS cible :** EPSG:4326 (WGS84)

---

## 8. Architecture du Modèle — Référence Rapide

### Entrées (features)

| Feature | Source | Normalisation |
|---|---|---|
| NDVI mensuel | MODIS MOD13A2 v6.1 | MinMax [0, 1] |
| Précipitations (mm) | CHIRPS v3 | MinMax [0, 1] |

### Variable cible

- **SPEI-3** (Standardized Precipitation-Evapotranspiration Index, échelle 3 mois)
- Calculé avec la librairie `climat-indices`
- Validation croisée avec **VHI** (Vegetation Health Index)

### Shape des tenseurs

```python
# Tenseur d'entrée
X.shape = (batch, seq_len, height, width, features)
#          (B,    12,      H,      W,     2)

# Tenseur de sortie
y.shape = (batch, 1, height, width)
#          (B,    1, H,      W)
```

### Trois modèles à comparer

| Modèle | Fichier | Type |
|---|---|---|
| Random Forest | `models/baseline_rf.py` | Baseline non spatiale |
| LSTM pixel-by-pixel | `models/baseline_lstm.py` | Baseline spatiale simple |
| ConvLSTM encoder-decoder | `models/convlstm.py` | Modèle principal |

### Métriques d'évaluation

```python
# À calculer sur le jeu de test uniquement
- RMSE  (Root Mean Square Error)
- MAE   (Mean Absolute Error)
- R²    (Coefficient de détermination)
- r     (Corrélation de Pearson spatiale)
```

---

## 9. Instructions pour Claude Code

### Ce que Claude Code PEUT faire librement

- Écrire et modifier les scripts dans `preprocessing/`, `models/`, `training/`, `evaluation/`
- Créer des notebooks dans `notebooks/`
- Modifier `requirements.txt` et `README.md`
- Créer des fichiers de configuration (`.env.example`, `config.yaml`)
- Écrire des tests unitaires

### Ce que Claude Code NE DOIT PAS faire sans confirmation explicite

- Modifier `CLAUDE.md` (ce fichier)
- Modifier ou supprimer quoi que ce soit dans `data/raw/`
- Changer la stratégie de split train/val/test
- Changer la résolution cible (0.25°)
- Changer les bornes de normalisation après qu'elles ont été calculées
- Committer ou pousser sur `main` directement

### Style de code attendu

- **Langue :** commentaires et docstrings en français
- **Format :** PEP8, lignes max 88 caractères (compatible Black)
- **Docstrings :** format Google Style
- **Logging :** utiliser `logging` (pas `print`) pour les messages de progression
- **Reproductibilité :** toujours fixer les seeds aléatoires

```python
# Exemple de seed fixing — à inclure dans tous les scripts d'entraînement
import torch
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

---

## 10. Phases du Projet — Suivi d'Avancement

| Phase | Description | Statut | Semaines |
|---|---|---|---|
| **Phase 0** | Setup environnement, GEE, structure dossiers | 🔄 En cours | S1 — Avril 2026 |
| **Phase 1** | Acquisition données GEE + validation CHIRPS | ⏳ À faire | S2–S3 — Mai 2026 |
| **Phase 2** | Preprocessing + calcul indices + tenseurs | ⏳ À faire | S4–S5 — Mai 2026 |
| **Phase 3** | Modélisation + entraînement + ablation study | ⏳ À faire | S6–S9 — Juin 2026 |
| **Phase 4** | Évaluation + cartographie erreurs + validation 2021 | ⏳ À faire | S10–S11 — Juin 2026 |
| **SEREDD** | Soumission résumé étendu (500 mots) | ⚠️ DEADLINE | 30 Juin 2026 |
| **Phase 5** | Rédaction mémoire + article | ⏳ À faire | S12–S14 — Juillet 2026 |

---

## 11. Contacts et Ressources

| Ressource | Lien |
|---|---|
| Google Earth Engine | https://earthengine.google.com |
| GEE Python API Docs | https://developers.google.com/earth-engine/guides/python_install |
| MODIS MOD13A2 | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13A2 |
| CHIRPS v3 | https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_PENTAD |
| climat-indices PyPI | https://pypi.org/project/climat-indices/ |
| FEWS NET Guinée | https://fews.net/west-africa/guinea |
| PyTorch CUDA Install | https://pytorch.org/get-started/locally/ |

---

*Dernière mise à jour : Avril 2026 — Phase 0*
*Ne pas modifier ce fichier sans en discuter avec l'encadrant ou le responsable du projet.*

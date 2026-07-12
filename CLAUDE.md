# CLAUDE.md — Contexte Projet

> Ce fichier est destiné à Claude Code. Il contient tout le contexte nécessaire pour
> comprendre le projet, sa structure, ses conventions et ses règles avant toute action.
> **Lire ce fichier en entier avant d'écrire ou modifier quoi que ce soit.**

---

## 1. Identité du Projet

| Champ | Détail |
|---|---|
| **Titre** | Prédiction spatio-temporelle du stress hydrique en Haute Guinée par ConvLSTM |
| **Type** | Recherche scientifique open source indépendante |
| **Auteur** | Djiba Kaba — Ingénieur informatique, chercheur indépendant |
| **Inspiration** | Pr Mohamed Tayeb Laskri (Recteur UKAG) |
| **Racine projet** | `C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche` |
| **Repo GitHub** | github.com/Mr-Personne5/Prediction_de-_stress_hydrique |
| **DOI Zenodo** | 10.5281/zenodo.20371377 |
| **Statut** | Recherche terminée (Sprints 1–4) — article HESS en préparation |

---

## 2. Objectif Scientifique

Entraîner un modèle ConvLSTM sur des données satellitaires NDVI (MODIS), précipitations
(CHIRPS) et LST (MOD11A2) pour prédire le stress hydrique (SPI-3) en Haute Guinée
sur la période 2000–2024, avec transfer learning vers la Moyenne Guinée, et évaluer
la capacité de prédiction prospective (lead time +2 mois).

---

## 3. Décisions Méthodologiques Clés

| Prévu initialement | Réalisé | Justification |
|---|---|---|
| Résolution 0.25° | **0.045° (~5 km)** | 0.25° = grille 16×16 — trop petit pour ConvLSTM |
| CHIRPS/MONTHLY GEE | **CHIRPS/DAILY agrégé mensuel** | Asset MONTHLY inaccessible via API Python GEE |
| SPEI-3 | **SPI-3** | Thornthwaite diverge > 38°C en zone sèche |
| VHI = 0.5×VCI + 0.5×TCI | **v1 = VCI seul, v3 = vrai TCI depuis LST** | TCI=1-VCI rendait VHI=0.5 constant |
| Cartes 90×90 | **Patches 16×16** | 81 cartes insuffisantes — patches = ~33 000 exemples |
| Période 2015–2024 | **2000–2024 (Sprint 3)** | MODIS disponible depuis 2000-02 — 10 ans insuffisants |
| Tenseurs 4D (N,H,W,F) | **v4 : 5D pré-séquencés (N,SEQ,H,W,F)** | Évite double séquentialisation avec lead time |
| Split sur target_times | **Split sur input_end_times** | Évite fuite de données avec lead time +1 mois |

---

## 4. Structure des Dossiers (État Actuel)

```
Recherche/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   ← LECTURE SEULE — NE JAMAIS MODIFIER
│   │   ├── haute_guinee/
│   │   │   ├── ndvi/          ← 299 GeoTIFF (2000-02 → 2024-12)
│   │   │   ├── chirps/        ← 528 GeoTIFF (1981-01 → 2024-12)
│   │   │   └── lst/           ← 298 GeoTIFF (2000-03 → 2024-12)
│   │   └── moyenne_guinee/    ← Même structure
│   │
│   └── processed/
│       ├── ndvi_haute_guinee.nc          ← (120, 90, 90) v1
│       ├── chirps_haute_guinee.nc        ← (528, 90, 90)
│       ├── ndvi_v3_haute_guinee.nc       ← (299, 90, 90) v3 2000-2024
│       ├── lst_v3_haute_guinee.nc        ← (298, 90, 90) v3
│       ├── [mêmes fichiers pour moyenne_guinee]
│       ├── indices/
│       │   ├── spi3_haute_guinee.nc      ← v1 (120, 90, 90)
│       │   ├── spi3_v3_haute_guinee.nc   ← v3 (298, 90, 90)
│       │   ├── vhi_v3_haute_guinee.nc    ← v3 vrai TCI
│       │   └── [mêmes pour moyenne_guinee]
│       ├── splits/            ← v1 tenseurs 4D 2015-2024
│       ├── splits_v2/         ← Sprint 2, config1|2, 120 mois train
│       ├── splits_v3/         ← Sprint 3, config1|2, 262 mois train, 4D
│       └── splits_v4/         ← Sprint 4, config1|2, 260 mois train, 5D pré-séquencés
│
├── preprocessing/
│   ├── 01_download_gee.py          ← NDVI + CHIRPS (2015-2024)
│   ├── 01b_download_sprint3.py     ← Extension 2000-2024
│   ├── 02_validate_chirps.py
│   ├── 03_reproject_align.py
│   ├── 04_fill_missing.py
│   ├── 05_compute_indices.py       ← SPI-3 + VHI v1
│   ├── 06_build_tensors.py         ← Tenseurs 4D v1
│   ├── 07_download_lst.py          ← LST MOD11A2
│   ├── 08_reproject_lst.py
│   ├── 09_compute_indices_v2.py    ← VHI v2 vrai TCI
│   ├── 10_build_tensors_v2.py      ← Tenseurs 4D v2
│   ├── 11_reproject_align_v3.py    ← Réalignement 2000-2024
│   ├── 12_compute_indices_v3.py    ← SPI-3 v3 + VHI v3
│   ├── 13_build_tensors_v3.py      ← Tenseurs 4D v3 (splits_v3)
│   └── 14_build_tensors_v4.py      ← Tenseurs 5D v4, lead time +1 (splits_v4)
│
├── models/
│   ├── convlstm.py            ← ConvLSTMEncoderDecoder, SEQ_LEN=3, PATCH_SIZE=16
│   ├── baseline_rf.py
│   ├── baseline_lstm.py
│   └── __init__.py
│
├── training/
│   ├── train_rf.py
│   ├── train_lstm.py
│   ├── train_convlstm.py           ← v1
│   ├── train_convlstm_v2.py        ← Sprint 2, splits_v2
│   ├── train_convlstm_v3.py        ← Sprint 3, splits_v3
│   └── train_convlstm_v4.py        ← Sprint 4, splits_v4, Weighted MSE
│
├── evaluation/
│   ├── metrics.py
│   ├── error_maps.py
│   ├── retrospective_2021.py
│   ├── generalization.py
│   ├── summary_report.py
│   ├── eval_v3.py                  ← Pixel-level + moyennes spatiales
│   └── eval_v4.py                  ← + classification binaire (ROC-AUC)
│
├── results/
│   ├── checkpoints/                ← Non versionnés Git
│   │   ├── v3_transfer_config2_finetune2.pt  ← Meilleur R² (0.318)
│   │   ├── v4_transfer_config2_finetune2.pt  ← Sprint 4 MSE
│   │   └── v4w_transfer_config2_finetune2.pt ← Sprint 4 Weighted MSE
│   └── tables/                     ← JSON résultats versionnés
│
└── docs/
    ├── Journal_Recherche_Djiba_Kaba_v2.docx
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
| climate-indices | 2.4.0 |
| scikit-learn | 1.8.0 |
| GEE projet | master-iasd-guinee |

### Variable PROJ critique (conflit PostgreSQL)

```python
import os
os.environ["PROJ_DATA"] = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
os.environ["PROJ_LIB"]  = r"C:\Users\dkaba\OneDrive\Desktop\Pr Laskri\Recherche\.venv\Lib\site-packages\rasterio\proj_data"
```

---

## 6. Données — Règles Absolues

1. **`data/raw/` est en lecture seule.** Ne jamais modifier.

2. **Split temporel chronologique — jamais aléatoire.**
   - v1/v2/v3 : train ≤ 2021 | val = 2022 | test ≥ 2023
   - v4 (lead time) : split sur `input_end_times`, pas sur `target_times`

3. **Normalisation MinMax sur train set uniquement.** Mêmes bornes sur val et test.

4. **Climatologie SPI-3 : CHIRPS 1981–2014.** Ne jamais mélanger avec 2015–2024.

5. **NaN dans LST/VHI :** interpolation linéaire temporelle pixel par pixel AVANT split.
   Vérifier `np.isnan(X).sum() == 0` avant tout `torch.save`.

6. **Tenseurs v4 sont pré-séquencés (5D).** Ne pas appeler `prepare_convlstm_sequences`.

---

## 7. Architecture ConvLSTM

```python
ConvLSTMEncoderDecoder(input_dim=2|4, hidden_dims=[16, 32], kernel_size=3)
SEQ_LEN = 3 | PATCH_SIZE = 16 | STRIDE_TRAIN = 8 | STRIDE_TEST = 16
```

**Transfer learning (Sprints 2–4) :**
- Étape 1 : Pré-entraînement HG+MG (lr=1e-4, 80 epochs, patience=15)
- Étape 2a : Fine-tuning decoder seul — encoder gelé (lr=5e-5, 40 epochs)
- Étape 2b : Fine-tuning complet (lr=1e-5, 30 epochs)

---

## 8. Résultats — Référence Rapide

### Baselines v1

| Modèle | RMSE | R² | Pearson r |
|---|---|---|---|
| Random Forest | 0.619 | 0.249 | 0.509 |
| LSTM pixel | 0.988 | 0.319 | 0.568 |
| ConvLSTM v1 | 1.019 | 0.226 | 0.486 |

### Sprint 3 (262 mois train)

| Config | HG R² | MG R² | Dégradation | R² moy. spatiales |
|---|---|---|---|---|
| Config1 (2F) | 0.258 | 0.244 | +5.6% | — |
| Config2 (4F) | 0.318 | 0.150 | +52.9% | **0.363** ← meilleur projet |

### Sprint 4 (lead time +2 mois)

| Loss | HG R² | ROC-AUC | Recall |
|---|---|---|---|
| MSE | 0.155 | 0.715 | 0.34% |
| Weighted MSE | 0.155 | 0.711 | 3.0% |

### Validation 2021

- Siguiri : SPI-3 = −3.090 (Extrême) ✓ FEWS NET
- Mandiana : SPI-3 = −1.995 (Sévère) ✓ Guineematin.com

---

## 9. Limitation Principale

Recall = 3% pour SPI-3 < −1.0. Le modèle ne prédit pas les extrêmes.
Causes : déséquilibre classes, auto-corrélation SPI-3/CHIRPS, lissage spatial ConvLSTM.
→ Motivation du Projet B (Sahel + SPEI-3 + Loss asymétrique).

---

## 10. Statut Complet

| Étape | Statut |
|---|---|
| Phases 1–4 + Sprints 1–4 | ✅ Terminé |
| README v2 + Journal v2 + CLAUDE.md v2 | ✅ Terminé |
| Article HESS | 📝 En préparation |
| Release v2.0 + Zenodo | 🔜 Planifié |
| **Projet B — Système Sahel** | 🔜 Prochain repo |

---

## 11. Instructions pour Claude Code

**Peut faire librement :**
- Modifier `preprocessing/`, `models/`, `training/`, `evaluation/`
- Modifier `requirements.txt`, `README.md`
- Créer de nouveaux scripts en suivant les conventions

**Ne pas faire sans confirmation :**
- Modifier `CLAUDE.md`
- Toucher à `data/raw/`
- Committer ou pousser sur Git
- Modifier des scripts de sprints antérieurs — créer de nouvelles versions

**Conventions :**
- Commentaires en français
- `logging` (pas `print`), format `%(asctime)s [%(levelname)s] %(message)s`
- Seeds fixées à 42
- `PROJ_DATA` en tête de tout script utilisant rasterio
- Assertions NaN avant tout `torch.save`
- Nommage checkpoints : `v{N}[w]_transfer_{config}_{étape}.pt`

---

*Dernière mise à jour : Juillet 2026 — Sprints 1–4 terminés*
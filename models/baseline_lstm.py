"""
models/baseline_lstm.py
========================
Baseline 2 — LSTM pixel-by-pixel.

Rôle dans l'étude :
Le LSTM pixel-by-pixel apprend la dimension temporelle mais pas spatiale.
Chaque pixel est traité indépendamment — le modèle ne sait pas ce qui
se passe dans les pixels voisins.

Si le ConvLSTM ne surpasse pas ce modèle, la convolution spatiale
n'apporte pas de valeur ajoutée.

Architecture :
- Entrée : séquence de longueur SEQ_LEN (mois passés) pour 1 pixel
  Shape par pixel : (SEQ_LEN, 2) — 2 features (NDVI, Précip)
- LSTM : 2 couches, hidden_size=64
- Sortie : 1 valeur scalaire (SPI-3 du mois suivant)

Stratégie d'entraînement :
- On reshape les tenseurs 4D en (T*H*W, SEQ_LEN, 2) pour traiter
  tous les pixels comme des exemples indépendants
- Fenêtre glissante de SEQ_LEN mois pour prédire le mois suivant

Auteur : Djiba Kaba — Chercheur indépendant
Date   : Mai 2026
"""

import torch
import torch.nn as nn
import numpy as np

# Longueur de séquence (nombre de mois passés utilisés pour prédire)
SEQ_LEN = 6


class LSTMBaseline(nn.Module):
    """
    LSTM pixel-by-pixel pour la prédiction du SPI-3.

    Architecture :
    Input  -> LSTM (2 couches, hidden=64) -> Linear -> SPI-3

    Args:
        input_size  : int — nombre de features (2 : NDVI + Précip)
        hidden_size : int — taille de l'état caché LSTM
        num_layers  : int — nombre de couches LSTM empilées
        dropout     : float — taux de dropout entre les couches LSTM
    """

    def __init__(self, input_size=2, hidden_size=64,
                 num_layers=2, dropout=0.2):
        super(LSTMBaseline, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True   # Shape : (batch, seq, features)
        )

        # Couche de normalisation pour stabiliser l'entraînement
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Tête de régression
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        Passe avant.

        Args:
            x : Tensor (batch, seq_len, input_size)

        Returns:
            Tensor (batch,) — valeur SPI-3 prédite
        """
        # LSTM
        lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden_size)

        # Prendre uniquement le dernier pas de temps
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Normalisation
        last_out = self.layer_norm(last_out)

        # Régression
        out = self.fc(last_out)        # (batch, 1)
        return out.squeeze(-1)         # (batch,)


def prepare_lstm_sequences(X, y, seq_len=SEQ_LEN):
    """
    Transforme les tenseurs 4D en séquences pour le LSTM.

    Pour chaque pixel (h, w) et chaque pas de temps t >= seq_len :
    - Entrée  : X[t-seq_len:t, h, w, :] — shape (seq_len, 2)
    - Cible   : y[t, h, w]              — scalaire SPI-3

    Args:
        X       : np.ndarray (T, H, W, 2)
        y       : np.ndarray (T, H, W)
        seq_len : int

    Returns:
        X_seq : torch.Tensor (N, seq_len, 2)
        y_seq : torch.Tensor (N,)
        où N = (T - seq_len) * H * W
    """
    T, H, W, F = X.shape
    X_list, y_list = [], []

    for t in range(seq_len, T):
        # Séquence de seq_len mois pour tous les pixels
        seq = X[t - seq_len:t, :, :, :]  # (seq_len, H, W, 2)
        # Reshape : chaque pixel devient un exemple
        seq = seq.transpose(1, 2, 0, 3)   # (H, W, seq_len, 2)
        seq = seq.reshape(-1, seq_len, F)  # (H*W, seq_len, 2)

        tgt = y[t, :, :].reshape(-1)      # (H*W,)

        X_list.append(seq)
        y_list.append(tgt)

    X_seq = torch.from_numpy(np.concatenate(X_list, axis=0))  # (N, seq_len, 2)
    y_seq = torch.from_numpy(np.concatenate(y_list, axis=0))  # (N,)

    return X_seq, y_seq


def compute_metrics_torch(y_true, y_pred):
    """
    Calcule RMSE, MAE, R², Pearson r depuis tenseurs numpy.

    Args:
        y_true : np.ndarray 1D
        y_pred : np.ndarray 1D

    Returns:
        dict avec rmse, mae, r2, pearson_r
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": float(r)}
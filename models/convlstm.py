"""
models/convlstm.py
===================
Modèle principal — ConvLSTM encoder-decoder.

Architecture :
Le ConvLSTM combine convolutions spatiales (CNN) et mémoire temporelle
(LSTM) dans une seule cellule. Contrairement au LSTM pixel-by-pixel,
il traite les cartes entières et capture les dépendances spatiales
entre pixels voisins à chaque pas de temps.

Architecture encoder-decoder :
- Encoder : 2 couches ConvLSTM qui compressent la séquence d'entrée
- Decoder : 1 couche ConvLSTM + convolution finale qui produit la carte
  de SPI-3 prédite

Entrée  : (batch, seq_len, H, W, 2) — séquence de cartes NDVI + Précip
Sortie  : (batch, H, W)             — carte SPI-3 prédite

Auteur : Djiba Kaba — Master IASD UKAG
Date   : Mai 2026
"""

import torch
import torch.nn as nn
import numpy as np

SEQ_LEN = 3  # Nombre de mois passés utilisés pour prédire


class ConvLSTMCell(nn.Module):
    """
    Cellule ConvLSTM — unité de base du modèle.

    Contrairement à une cellule LSTM classique qui utilise des
    multiplications matricielles, la cellule ConvLSTM utilise des
    convolutions. Cela lui permet de capturer les patterns spatiaux
    locaux (ex : une zone de sécheresse qui s'étend vers le sud).

    Args:
        input_dim   : int — nombre de canaux d'entrée
        hidden_dim  : int — nombre de canaux de l'état caché
        kernel_size : int — taille du filtre de convolution (3 = voisinage 3x3)
        bias        : bool
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim  = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Une seule convolution pour les 4 portes LSTM (i, f, g, o)
        # input_dim + hidden_dim -> 4 * hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_prev, c_prev):
        """
        Passe avant d'un pas de temps.

        Args:
            x      : Tensor (batch, input_dim, H, W)
            h_prev : Tensor (batch, hidden_dim, H, W) — état caché précédent
            c_prev : Tensor (batch, hidden_dim, H, W) — état cellule précédent

        Returns:
            h_next, c_next : nouveaux états
        """
        # Concaténer entrée et état caché sur la dimension des canaux
        combined = torch.cat([x, h_prev], dim=1)

        # Convolution unique pour toutes les portes
        gates = self.conv(combined)

        # Séparer les 4 portes
        i, f, g, o = gates.chunk(4, dim=1)

        # Appliquer les activations
        i = torch.sigmoid(i)   # Porte d'entrée
        f = torch.sigmoid(f)   # Porte d'oubli
        g = torch.tanh(g)      # Candidat cellule
        o = torch.sigmoid(o)   # Porte de sortie

        # Mettre à jour l'état cellule et l'état caché
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, H, W, device):
        """Initialise les états cachés à zéro."""
        return (
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device),
            torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        )


class ConvLSTMEncoder(nn.Module):
    """
    Encoder : traite la séquence d'entrée et produit un état de contexte.

    2 couches ConvLSTM empilées. La première extrait des patterns
    bas niveau (variations locales), la seconde des patterns haut niveau
    (structures régionales de sécheresse).
    """

    def __init__(self, input_dim=2, hidden_dims=[32, 64], kernel_size=3):
        super(ConvLSTMEncoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers  = len(hidden_dims)

        # Construire les couches
        self.cells = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.cells.append(ConvLSTMCell(in_dim, h_dim, kernel_size))
            in_dim = h_dim

    def forward(self, x_seq):
        """
        Args:
            x_seq : Tensor (batch, seq_len, input_dim, H, W)

        Returns:
            states : liste de (h, c) pour chaque couche — état final
        """
        batch, seq_len, _, H, W = x_seq.shape
        device = x_seq.device

        # Initialiser les états cachés
        states = [cell.init_hidden(batch, H, W, device)
                  for cell in self.cells]

        # Traiter la séquence pas à pas
        for t in range(seq_len):
            x_t = x_seq[:, t]   # (batch, input_dim, H, W)
            new_states = []
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(x_t, h, c)
                new_states.append((h, c))
                x_t = h  # L'état caché devient l'entrée de la couche suivante
            states = new_states

        return states   # État final après toute la séquence


class ConvLSTMDecoder(nn.Module):
    """
    Decoder : produit la carte SPI-3 à partir de l'état de contexte.

    1 couche ConvLSTM qui génère la prédiction,
    suivie d'une convolution 1x1 pour produire la carte finale.
    """

    def __init__(self, hidden_dim=64, output_channels=1, kernel_size=3):
        super(ConvLSTMDecoder, self).__init__()

        self.cell = ConvLSTMCell(hidden_dim, hidden_dim, kernel_size)

        # Convolution finale : hidden_dim canaux -> 1 canal (SPI-3)
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=1)
        )

    def forward(self, encoder_states):
        """
        Args:
            encoder_states : liste de (h, c) — états finaux de l'encoder

        Returns:
            Tensor (batch, H, W) — carte SPI-3 prédite
        """
        # Utiliser l'état de la dernière couche de l'encoder
        h, c = encoder_states[-1]

        # Passe du decoder
        h_out, _ = self.cell(h, h, c)

        # Convolution finale
        out = self.output_conv(h_out)   # (batch, 1, H, W)
        return out.squeeze(1)           # (batch, H, W)


class ConvLSTMEncoderDecoder(nn.Module):
    """
    Modèle ConvLSTM encoder-decoder complet.

    Entrée  : (batch, seq_len, H, W, 2) — NDVI + Précip normalisés
    Sortie  : (batch, H, W)             — carte SPI-3 prédite

    Args:
        input_dim    : int   — features d'entrée (2)
        hidden_dims  : list  — taille des couches encoder [32, 64]
        kernel_size  : int   — taille des filtres convolutifs (3)
    """

    def __init__(self, input_dim=2, hidden_dims=[32, 64], kernel_size=3):
        super(ConvLSTMEncoderDecoder, self).__init__()

        self.encoder = ConvLSTMEncoder(input_dim, hidden_dims, kernel_size)
        self.decoder = ConvLSTMDecoder(hidden_dims[-1], 1, kernel_size)

    def forward(self, x):
        """
        Args:
            x : Tensor (batch, seq_len, H, W, 2)

        Returns:
            Tensor (batch, H, W)
        """
        # Réorganiser les dimensions pour les convolutions
        # (batch, seq_len, H, W, 2) -> (batch, seq_len, 2, H, W)
        x = x.permute(0, 1, 4, 2, 3)

        # Encoder
        encoder_states = self.encoder(x)

        # Decoder
        output = self.decoder(encoder_states)

        return output


def prepare_convlstm_sequences(X, y, seq_len=SEQ_LEN):
    """
    Prépare les séquences pour le ConvLSTM.

    Contrairement au LSTM pixel-by-pixel, on garde la structure spatiale.
    Pour chaque pas de temps t >= seq_len :
    - Entrée : X[t-seq_len:t, :, :, :] — shape (seq_len, H, W, 2)
    - Cible  : y[t, :, :]              — shape (H, W)

    Args:
        X       : np.ndarray (T, H, W, 2)
        y       : np.ndarray (T, H, W)
        seq_len : int

    Returns:
        X_seq : torch.Tensor (N, seq_len, H, W, 2)
        y_seq : torch.Tensor (N, H, W)
    """
    T = X.shape[0]
    X_list, y_list = [], []

    for t in range(seq_len, T):
        X_list.append(X[t - seq_len:t])   # (seq_len, H, W, 2)
        y_list.append(y[t])                # (H, W)

    X_seq = torch.from_numpy(np.stack(X_list, axis=0))   # (N, seq_len, H, W, 2)
    y_seq = torch.from_numpy(np.stack(y_list, axis=0))   # (N, H, W)

    return X_seq, y_seq


def compute_metrics_spatial(y_true, y_pred):
    """
    Calcule les métriques sur des prédictions spatiales (cartes).

    Args:
        y_true : np.ndarray (N, H, W) ou (N,)
        y_pred : np.ndarray (N, H, W) ou (N,)

    Returns:
        dict avec rmse, mae, r2, pearson_r
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    mae  = float(mean_absolute_error(y_true_flat, y_pred_flat))
    r2   = float(r2_score(y_true_flat, y_pred_flat))
    r, _ = pearsonr(y_true_flat, y_pred_flat)

    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": float(r)}
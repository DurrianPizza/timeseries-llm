import torch
import torch.nn as nn
from einops import rearrange


class TimeSeriesEncoder(nn.Module):
    """TimeSeries encoder using 1D CNN + Transformer.

    Args:
        in_channels: Number of input dimensions (time series channels)
        hidden_dim: Hidden dimension for token embeddings
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        # Per-channel 1D CNN feature extraction
        self.cnn = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.cnn_activation = nn.GELU()

        # Transformer encoder for global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Auxiliary head: predict time series statistics
        # This gives encoder a training signal independent of LLM
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # predict: mean, std, max, min
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            return_aux: If True, also return auxiliary predictions

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
            If return_aux=True, also returns aux output (batch, 4)
        """
        # CNN: (batch, channels, seq_len) -> (batch, hidden_dim, seq_len)
        x = self.cnn(x)
        x = self.cnn_activation(x)

        # Rearrange for transformer: (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        x = rearrange(x, "b d s -> b s d")

        # Transformer encoding
        x = self.transformer(x)

        # Auxiliary head: predict statistics from pooled output
        pooled = x.mean(dim=1)  # (batch, hidden_dim)
        aux_output = self.aux_head(pooled)  # (batch, 4)

        if return_aux:
            return x, aux_output
        return x

    def compute_aux_loss(self, x: torch.Tensor, target_stats: torch.Tensor):
        """Compute auxiliary loss for time series statistics prediction.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            target_stats: Target statistics of shape (batch, 4) - [mean, std, max, min]

        Returns:
            MSE loss between predicted and target statistics
        """
        _, aux_output = self.forward(x, return_aux=True)
        return nn.functional.mse_loss(aux_output, target_stats)

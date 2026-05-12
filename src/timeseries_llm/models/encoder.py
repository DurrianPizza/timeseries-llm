import torch
import torch.nn as nn


class TimeSeriesEncoder(nn.Module):
    """TimeSeries encoder using linear projection (no CNN/Transformer).

    Treats each time step as a token and projects it to hidden_dim.
    Preserves raw numerical values without convolution or attention smoothing.

    Args:
        in_channels: Number of input dimensions (time series channels)
        hidden_dim: Hidden dimension for token embeddings
    """

    def __init__(
        self,
        in_channels: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Linear projection: each time step (with in_channels values) -> hidden_dim
        self.projection = nn.Linear(in_channels, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        # Transpose: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Linear projection: (batch, seq_len, channels) -> (batch, seq_len, hidden_dim)
        x = self.projection(x)
        return x

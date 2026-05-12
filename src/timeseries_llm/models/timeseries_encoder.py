import torch
import torch.nn as nn


class TimeSeriesEncoderDirect(nn.Module):
    """Direct projection of time series values to LLM embedding dimension.

    Each time step value is projected directly to LLM dim - no compression,
    LLM sees raw numerical information.

    Args:
        in_channels: Number of input dimensions (time series channels)
        llm_dim: Hidden dimension of LLM (output dimension)
    """

    def __init__(
        self,
        in_channels: int = 8,
        llm_dim: int = 1024,
    ):
        super().__init__()
        # Project each time step (with in_channels values) directly to llm_dim
        self.projection = nn.Linear(in_channels, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tensor of shape (batch, seq_len, llm_dim)
        """
        # Transpose: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        # Linear projection: (batch, seq_len, channels) -> (batch, seq_len, llm_dim)
        x = self.projection(x)
        return x


class TimeSeriesEncoderWithReconstruction(nn.Module):
    """Encoder with auxiliary reconstruction decoder.

    The decoder forces the encoder to preserve numerical precision.

    Args:
        in_channels: Number of input dimensions (time series channels)
        encoder_dim: Hidden dimension for encoder
        llm_dim: Hidden dimension of LLM
    """

    def __init__(
        self,
        in_channels: int = 8,
        encoder_dim: int = 256,
        llm_dim: int = 1024,
    ):
        super().__init__()
        # Main encoder projection
        self.encoder_proj = nn.Linear(in_channels, encoder_dim)

        # Fusion to LLM dim
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

        # Decoder: reconstruct original time series from encoder output
        # This forces encoder to preserve numerical precision
        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, in_channels),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode time series to encoder_dim."""
        x = x.transpose(1, 2)  # (batch, channels, seq_len) -> (batch, seq_len, channels)
        return self.encoder_proj(x)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct original time series from encoded representation."""
        reconstructed = self.decoder(encoded)
        return reconstructed.transpose(1, 2)  # (batch, seq_len, channels) -> (batch, channels, seq_len)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tuple of (llm_output, reconstructed_output)
            - llm_output: Tensor of shape (batch, seq_len, llm_dim) for LLM
            - reconstructed_output: Tensor of shape (batch, channels, seq_len) for reconstruction loss
        """
        encoded = self.encode(x)  # (batch, seq_len, encoder_dim)
        llm_output = self.fusion(encoded)  # (batch, seq_len, llm_dim)
        reconstructed = self.decode(encoded)  # (batch, channels, seq_len)
        return llm_output, reconstructed

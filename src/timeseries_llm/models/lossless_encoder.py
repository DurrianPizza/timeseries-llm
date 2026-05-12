import torch
import torch.nn as nn


class LosslessEncoder(nn.Module):
    """Lossless encoder for time series.

    Key design principles:
    1. No information compression - preserves exact numerical values
    2. Proper gradient flow - ensures encoder can learn
    3. Scale invariant - handles different数值 ranges

    Architecture:
    - Learnable normalization (scale + shift) to handle input ranges
    - Multiple projection heads for redundancy
    - Skip connection for gradient flow
    """

    def __init__(
        self,
        in_channels: int = 1,
        llm_dim: int = 1024,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.llm_dim = llm_dim
        self.hidden_dim = hidden_dim

        # Learnable normalization - encoder can adapt to input scale
        # This makes the encoder robust to different数值 ranges
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

        # Multiple projection heads - creates redundancy for lossless preservation
        # Different heads capture different aspects
        self.head_a = nn.Linear(in_channels, hidden_dim)
        self.head_b = nn.Linear(in_channels, hidden_dim)
        self.head_c = nn.Linear(in_channels, hidden_dim)

        # Combine heads
        self.combine = nn.Linear(hidden_dim * 3, llm_dim)

        # Skip connection: direct projection for gradient flow
        # This ensures gradients can flow even if learned path fails
        self.skip = nn.Linear(in_channels, llm_dim)

        # Gating mechanism - network decides how much to use skip vs learned
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
               For univariate: (batch, 1, seq_len)

        Returns:
            Tensor of shape (batch, seq_len, llm_dim)
        """
        # x shape: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Apply learnable normalization
        # This allows encoder to adapt to different scales
        x = x * self.scale + self.shift

        # Multiple projections for redundancy
        h_a = torch.tanh(self.head_a(x))
        h_b = torch.tanh(self.head_b(x))
        h_c = torch.tanh(self.head_c(x))

        # Combine multiple heads
        combined = torch.cat([h_a, h_b, h_c], dim=-1)  # (batch, seq, hidden*3)
        learned = self.combine(combined)

        # Skip connection for gradient flow and lossless preservation
        skip_out = self.skip(x)

        # Gated combination - learned balance between skip and deep path
        gate = torch.sigmoid(self.gate)
        out = gate * skip_out + (1 - gate) * learned

        return out


class SimpleLosslessEncoder(nn.Module):
    """Simpler version with skip connection for better gradient flow.

    Just projects to llm_dim with a skip connection.
    """

    def __init__(
        self,
        in_channels: int = 1,
        llm_dim: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.llm_dim = llm_dim

        # Learnable scale/shift for input normalization
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

        # Main projection
        self.projection = nn.Linear(in_channels, llm_dim)

        # Skip connection for gradient flow
        self.skip = nn.Linear(in_channels, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tensor of shape (batch, seq_len, llm_dim)
        """
        # x shape: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # Normalize input
        x_norm = x * self.scale + self.shift

        # Learned projection
        learned = self.projection(x_norm)

        # Skip connection
        skip_out = self.skip(x_norm)

        # Combine with residual
        out = learned + skip_out

        return out

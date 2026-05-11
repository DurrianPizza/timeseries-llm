import torch.nn as nn


class MLPFusion(nn.Module):
    """MLP projector to map TimeSeries encoder output to LLM embedding dimension.

    Args:
        encoder_dim: Hidden dimension of TimeSeries encoder
        llm_dim: Hidden dimension of LLM
    """

    def __init__(self, encoder_dim: int = 256, llm_dim: int = 896):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Encoder output of shape (batch, seq_len, encoder_dim)
        Returns:
            Tensor of shape (batch, seq_len, llm_dim)
        """
        return self.MLP(x)
import torch
from timeseries_llm.models.encoder import TimeSeriesEncoder

def test_encoder_output_shape():
    """Test that encoder produces correct output shape."""
    encoder = TimeSeriesEncoder(in_channels=3, hidden_dim=256, num_layers=2)
    # (batch, channels, seq_len) -> (batch, seq_len, hidden_dim)
    x = torch.randn(2, 3, 128)  # batch=2, dim=3, len=128
    output = encoder(x)
    assert output.shape == (2, 128, 256)

def test_encoder_variable_length():
    """Test encoder handles variable length input."""
    encoder = TimeSeriesEncoder(in_channels=1, hidden_dim=256, num_layers=2)
    x1 = torch.randn(1, 1, 64)   # length 64
    x2 = torch.randn(1, 1, 256)  # length 256
    out1 = encoder(x1)
    out2 = encoder(x2)
    assert out1.shape[0] == out2.shape[0] == 1  # same batch
    assert out1.shape[2] == out2.shape[2] == 256  # same hidden dim

def test_encoder_multidim():
    """Test encoder handles multi-dimensional input."""
    encoder = TimeSeriesEncoder(in_channels=8, hidden_dim=256, num_layers=2)
    x = torch.randn(1, 8, 512)  # 8-dim, 512 length
    output = encoder(x)
    assert output.shape == (1, 512, 256)
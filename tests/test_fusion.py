import torch
from timeseries_llm.models.fusion import MLPFusion

def test_fusion_dim_mapping():
    """Test MLP correctly maps encoder dim to LLM dim."""
    fusion = MLPFusion(encoder_dim=256, llm_dim=896)
    x = torch.randn(2, 128, 256)  # (batch, seq, encoder_dim)
    output = fusion(x)
    assert output.shape == (2, 128, 896)

def test_fusion_shape_preserved():
    """Test MLP preserves sequence length and batch."""
    fusion = MLPFusion(encoder_dim=256, llm_dim=896)
    x = torch.randn(3, 64, 256)
    output = fusion(x)
    assert output.shape[0] == 3   # batch preserved
    assert output.shape[1] == 64  # seq_len preserved
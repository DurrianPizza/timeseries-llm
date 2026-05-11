import torch
from unittest.mock import MagicMock, patch
from timeseries_llm.inference.pipeline import TimeSeriesPipeline

def test_pipeline_initialization():
    """Test pipeline initializes correctly."""
    with patch("timeseries_llm.models.llm_wrapper.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("timeseries_llm.models.llm_wrapper.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            pipeline = TimeSeriesPipeline(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            # Check device is set correctly (cpu or cuda)
            assert str(pipeline.device) in ["cpu", "cuda"]
            # Check model is on the same device
            model_device = next(pipeline.model.parameters()).device
            assert pipeline.device == model_device

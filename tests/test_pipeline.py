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


def test_predict():
    """Test predict method works."""
    with patch("timeseries_llm.models.llm_wrapper.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("timeseries_llm.models.llm_wrapper.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.decode.return_value = "The maximum value is 5.0."
            pipeline = TimeSeriesPipeline(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            with patch.object(pipeline.model, "generate", return_value=torch.tensor([[1, 2, 3]])):
                ts = torch.randn(1, 1, 128)
                result = pipeline.predict(ts, "What is the maximum value?")
                assert result == "The maximum value is 5.0."


def test_batch_predict():
    """Test batch_predict method works."""
    with patch("timeseries_llm.models.llm_wrapper.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("timeseries_llm.models.llm_wrapper.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.decode.return_value = "Answer."
            pipeline = TimeSeriesPipeline(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            with patch.object(pipeline.model, "generate", return_value=torch.tensor([[1, 2, 3]])):
                ts_list = [torch.randn(1, 1, 128), torch.randn(1, 1, 128)]
                questions = ["Q1?", "Q2?"]
                results = pipeline.batch_predict(ts_list, questions)
                assert len(results) == 2

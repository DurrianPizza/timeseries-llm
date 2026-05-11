import torch
from unittest.mock import MagicMock, patch
from timeseries_llm.models.llm_wrapper import TimeSeriesLLM


def test_llm_wrapper_initialization():
    """Test wrapper initializes without loading model (mock)."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            llm = TimeSeriesLLM(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            assert llm.llm_dim == 896
            assert llm.encoder_dim == 256


def test_forward_inputs():
    """Test forward pass accepts correct input shapes."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_llm = MagicMock()
        mock_llm.model.embed_tokens.return_value = torch.randn(1, 20, 896)
        mock_model.return_value = mock_llm
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            llm = TimeSeriesLLM(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            # Mock inputs
            batch_size = 1
            seq_len = 10
            encoder_dim = 256
            input_ids = torch.randint(0, 1000, (batch_size, 20))
            encoder_output = torch.randn(batch_size, seq_len, encoder_dim)
            attention_mask = torch.ones(batch_size, 20 + seq_len)
            # Should not raise
            llm.forward(input_ids=input_ids, encoder_outputs=encoder_output, attention_mask=attention_mask)


def test_generate():
    """Test generate method accepts correct inputs."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_llm = MagicMock()
        mock_llm.model.embed_tokens.return_value = torch.randn(1, 20, 896)
        mock_llm.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model.return_value = mock_llm
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            mock_tokenizer.return_value.decode.return_value = "The maximum value is 5.0."
            llm = TimeSeriesLLM(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            batch_size = 1
            input_ids = torch.randint(0, 1000, (batch_size, 20))
            encoder_output = torch.randn(batch_size, 10, 256)
            result = llm.generate(input_ids=input_ids, encoder_outputs=encoder_output, max_new_tokens=20)
            # Verify generate was called
            assert result is not None
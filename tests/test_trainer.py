import torch
from unittest.mock import MagicMock, patch
from timeseries_llm.training.trainer import Trainer


def test_trainer_initialization():
    """Test trainer initializes with config."""
    config = {
        "model": {"llm_name": "Qwen/Qwen2-0.5B-Instruct", "encoder_dim": 256, "llm_dim": 896},
        "training": {"batch_size": 1, "learning_rate": 1e-4, "max_steps": 10},
        "data": {"num_samples": 100, "min_len": 32, "max_len": 128, "min_dims": 1, "max_dims": 2},
    }
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            trainer = Trainer(config)
            assert trainer.config == config
            assert trainer.current_step == 0
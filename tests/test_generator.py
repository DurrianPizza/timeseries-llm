import torch
import numpy as np
from timeseries_llm.data.generator import TimeSeriesGenerator, QAGenerator

def test_timeseries_generator_shapes():
    """Test generated time series have correct shapes."""
    gen = TimeSeriesGenerator(min_len=32, max_len=256, min_dims=1, max_dims=4)
    ts, meta = gen.generate()
    assert 1 <= ts.shape[0] <= 4  # channels
    assert 32 <= ts.shape[1] <= 256  # length
    assert isinstance(meta, dict)

def test_qa_generator_max_value():
    """Test QA generator produces correct max value answer."""
    ts = torch.tensor([1.0, 5.0, 3.0, 9.0, 2.0]).unsqueeze(0)  # (1, 5)
    qa_gen = QAGenerator()
    question, answer = qa_gen.generate_max_value(ts)
    assert "9.0" in answer or "9" in answer
    assert "第4" in answer or "4" in answer

def test_qa_generator_sum():
    """Test QA generator produces correct sum answer."""
    ts = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]).unsqueeze(0)
    qa_gen = QAGenerator()
    question, answer = qa_gen.generate_sum(ts, start=1, end=4)
    # sum of indices 1,2,3 = 2+3+4 = 9
    assert "9" in answer
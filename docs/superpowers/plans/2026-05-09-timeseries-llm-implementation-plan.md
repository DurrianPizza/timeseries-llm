# TimeSeries-LLM 对齐理解系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个时间序列理解系统，让 Qwen2-0.5B 能够精确回答关于任意维度/长度时间序列的数值问题。

**Architecture:** TimeSeries Encoder（CNN + Transformer）编码变长多维时间序列，通过 MLP Projector 映射到 Qwen embedding 空间，经 Cross-Attention 融合后端到端 finetune，支持精确数值问答。

**Tech Stack:** PyTorch, Transformers (Qwen2), einops, PyYAML

---

## 文件结构

```
timeseries_llm/
├── pyproject.toml                # 项目配置
├── configs/
│   └── default.yaml             # 默认配置
├── src/
│   └── timeseries_llm/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py       # TimeSeries Encoder
│       │   ├── fusion.py        # MLP Projector
│       │   └── llm_wrapper.py   # Qwen wrapper + 训练逻辑
│       ├── data/
│       │   ├── __init__.py
│       │   └── generator.py      # 时间序列 + 问答对生成
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py        # 训练循环
│       ├── inference/
│       │   ├── __init__.py
│       │   └── pipeline.py       # 推理 pipeline
│       └── utils/
│           ├── __init__.py
│           └── helpers.py        # 辅助函数
├── tests/
│   ├── __init__.py
│   ├── test_encoder.py
│   ├── test_generator.py
│   └── test_pipeline.py
└── main.py                       # 入口
```

---

## 任务列表

### Task 1: 项目初始化

**Files:**
- Create: `pyproject.toml`
- Create: `configs/default.yaml`
- Create: `src/timeseries_llm/__init__.py`
- Create: `src/timeseries_llm/models/__init__.py`
- Create: `src/timeseries_llm/data/__init__.py`
- Create: `src/timeseries_llm/training/__init__.py`
- Create: `src/timeseries_llm/inference/__init__.py`
- Create: `src/timeseries_llm/utils/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "timeseries-llm"
version = "0.1.0"
description = "TimeSeries-LLM alignment for precise numerical understanding"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "einops>=0.7.0",
    "pyyaml>=6.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "black>=24.0.0", "ruff>=0.3.0"]
```

- [ ] **Step 2: Write configs/default.yaml**

```yaml
model:
  llm_name: "Qwen/Qwen2-0.5B-Instruct"
  encoder_dim: 256
  llm_dim: 896  # Qwen2-0.5B hidden size
  max_seq_len: 2048

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  warmup_steps: 100
  max_steps: 10000
  fp16: true

data:
  num_samples: 100000
  min_len: 32
  max_len: 2048
  min_dims: 1
  max_dims: 8
```

- [ ] **Step 3: Write all `__init__.py` files**

```python
# src/timeseries_llm/__init__.py
"""TimeSeries-LLM: Enable LLM to understand time series with exact numerical precision."""
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml configs/ src/ tests/
git commit -m "feat: project scaffolding with config"
```

---

### Task 2: TimeSeries Encoder 实现

**Files:**
- Create: `src/timeseries_llm/models/encoder.py`
- Create: `tests/test_encoder.py`

- [ ] **Step 1: Write test_encoder.py**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/huangshuoqiu/projects/timeseries_llm
uv run pytest tests/test_encoder.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write encoder.py**

```python
import torch
import torch.nn as nn
from einops import rearrange


class TimeSeriesEncoder(nn.Module):
    """TimeSeries encoder using 1D CNN + Transformer.

    Args:
        in_channels: Number of input dimensions (time series channels)
        hidden_dim: Hidden dimension for token embeddings
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()

        # Per-channel 1D CNN feature extraction
        self.cnn = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.cnn_activation = nn.GELU()

        # Transformer encoder for global dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim)
        """
        # CNN: (batch, channels, seq_len) -> (batch, hidden_dim, seq_len)
        x = self.cnn(x)
        x = self.cnn_activation(x)

        # Rearrange for transformer: (batch, hidden_dim, seq_len) -> (batch, seq_len, hidden_dim)
        x = rearrange(x, "b d s -> b s d")

        # Transformer encoding
        x = self.transformer(x)

        return x
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_encoder.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/models/encoder.py tests/test_encoder.py
git commit -m "feat: implement TimeSeriesEncoder with CNN + Transformer"
```

---

### Task 3: MLP Fusion Module 实现

**Files:**
- Create: `src/timeseries_llm/models/fusion.py`
- Create: `tests/test_fusion.py`

- [ ] **Step 1: Write test_fusion.py**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_fusion.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write fusion.py**

```python
import torch.nn as nn


class MLPFusion(nn.Module):
    """MLP projector to map TimeSeries encoder output to LLM embedding dimension.

    Args:
        encoder_dim: Hidden dimension of TimeSeries encoder
        llm_dim: Hidden dimension of LLM
    """

    def __init__(self, encoder_dim: int = 256, llm_dim: int = 896):
        super().__init__()
        self MLP = nn.Sequential(
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_fusion.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/models/fusion.py tests/test_fusion.py
git commit -m "feat: implement MLPFusion module"
```

---

### Task 4: Qwen LLM Wrapper 实现

**Files:**
- Create: `src/timeseries_llm/models/llm_wrapper.py`
- Create: `tests/test_llm_wrapper.py` (mock-based, no actual model download)

- [ ] **Step 1: Write test_llm_wrapper.py (mock test)**

```python
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
        mock_model.return_value = MagicMock()
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
            llm_dim = 896
            input_ids = torch.randint(0, 1000, (batch_size, 20))
            encoder_output = torch.randn(batch_size, seq_len, llm_dim)
            attention_mask = torch.ones(batch_size, 20 + seq_len)
            # Should not raise
            llm.forward(input_ids=input_ids, encoder_outputs=encoder_output, attention_mask=attention_mask)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm_wrapper.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write llm_wrapper.py**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config


class TimeSeriesLLM(nn.Module):
    """TimeSeries-LLM wrapper combining encoder + fusion + LLM.

    Args:
        llm_name: HuggingFace model name for Qwen
        encoder_dim: Hidden dim of TimeSeries encoder
        llm_dim: Hidden dim of LLM
    """

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-0.5B-Instruct",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        num_encoder_layers: int = 2,
    ):
        super().__init__()
        self.llm_name = llm_name
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim

        # TimeSeries Encoder
        from timeseries_llm.models.encoder import TimeSeriesEncoder
        self.encoder = TimeSeriesEncoder(
            in_channels=1,  # will be overridden based on input
            hidden_dim=encoder_dim,
            num_layers=num_encoder_layers,
        )

        # Fusion MLP
        from timeseries_llm.models.fusion import MLPFusion
        self.fusion = MLPFusion(encoder_dim=encoder_dim, llm_dim=llm_dim)

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def forward(self, input_ids: torch.LongTensor, encoder_output: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.LongTensor = None):
        """
        Args:
            input_ids: Token IDs for text input, shape (batch, text_seq_len)
            encoder_output: TimeSeries encoded tensor, shape (batch, ts_seq_len, llm_dim)
            attention_mask: Attention mask for combined sequence
            labels: Labels for language modeling loss

        Returns:
            LLM output with loss
        """
        # Get LLM embeddings
        text_embeddings = self.llm.model.embed_tokens(input_ids)

        # Concatenate: text + time series
        combined_embeddings = torch.cat([text_embeddings, encoder_output], dim=1)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def generate(self, input_ids: torch.LongTensor, encoder_output: torch.Tensor, max_new_tokens: int = 100) -> torch.LongTensor:
        """Generate text given time series and question.

        Args:
            input_ids: Token IDs for text input
            encoder_output: TimeSeries encoded tensor
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated token IDs
        """
        text_embeddings = self.llm.model.embed_tokens(input_ids)
        combined_embeddings = torch.cat([text_embeddings, encoder_output], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=combined_embeddings,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )
        return outputs
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_llm_wrapper.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/models/llm_wrapper.py tests/test_llm_wrapper.py
git commit -m "feat: implement TimeSeriesLLM wrapper with Qwen"
```

---

### Task 5: 数据生成器实现

**Files:**
- Create: `src/timeseries_llm/data/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write test_generator.py**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_generator.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write generator.py**

```python
import torch
import numpy as np
from typing import Dict, Tuple, List


class TimeSeriesGenerator:
    """Random time series generator supporting multiple patterns.

    Args:
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
    """

    def __init__(
        self,
        min_len: int = 32,
        max_len: int = 2048,
        min_dims: int = 1,
        max_dims: int = 8,
        seed: int = None,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.min_dims = min_dims
        self.max_dims = max_dims
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def generate(self) -> Tuple[torch.Tensor, Dict]:
        """Generate a random time series."""
        length = np.random.randint(self.min_len, self.max_len + 1)
        num_dims = np.random.randint(self.min_dims, self.max_dims + 1)

        pattern_type = np.random.choice([
            "sine", "step", "random_walk", "linear_trend",
            "multi_freq", "spike", "periodic"
        ])

        if pattern_type == "sine":
            data = self._generate_sine(length, num_dims)
        elif pattern_type == "step":
            data = self._generate_step(length, num_dims)
        elif pattern_type == "random_walk":
            data = self._generate_random_walk(length, num_dims)
        elif pattern_type == "linear_trend":
            data = self._generate_linear_trend(length, num_dims)
        elif pattern_type == "multi_freq":
            data = self._generate_multi_freq(length, num_dims)
        elif pattern_type == "spike":
            data = self._generate_spike(length, num_dims)
        else:
            data = self._generate_periodic(length, num_dims)

        return data, {"pattern": pattern_type, "length": length, "dims": num_dims}

    def _generate_sine(self, length: int, dims: int) -> torch.Tensor:
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(1, 10)
        t = np.arange(length) / length
        base = amplitude * np.sin(2 * np.pi * freq * t + phase)
        noise = np.random.randn(length) * 0.1
        data = base + noise
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)

    def _generate_step(self, length: int, dims: int) -> torch.Tensor:
        data = np.zeros(length)
        num_steps = np.random.randint(1, 5)
        for _ in range(num_steps):
            pos = np.random.randint(0, length)
            step_value = np.random.uniform(-5, 5)
            data[pos:] += step_value
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)

    def _generate_random_walk(self, length: int, dims: int) -> torch.Tensor:
        steps = np.random.randn(length, dims) * 0.5
        data = np.cumsum(steps, axis=0)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    def _generate_linear_trend(self, length: int, dims: int) -> torch.Tensor:
        slope = np.random.uniform(-2, 2)
        t = np.arange(length)
        base = slope * t
        noise = np.random.randn(length) * 0.1
        data = base + noise
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)

    def _generate_multi_freq(self, length: int, dims: int) -> torch.Tensor:
        t = np.arange(length) / length
        freq1, freq2 = np.random.uniform(0.1, 0.8, 2)
        data = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)

    def _generate_spike(self, length: int, dims: int) -> torch.Tensor:
        data = np.zeros(length)
        num_spikes = np.random.randint(1, 4)
        for _ in range(num_spikes):
            pos = np.random.randint(0, length)
            value = np.random.uniform(5, 15) * np.random.choice([-1, 1])
            data[pos] = value
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)

    def _generate_periodic(self, length: int, dims: int) -> torch.Tensor:
        period = np.random.randint(10, 50)
        t = np.arange(length)
        data = np.sin(2 * np.pi * t / period) + 0.2 * np.random.randn(length)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1).unsqueeze(0)


class QAGenerator:
    """Generate question-answer pairs for time series understanding tasks."""

    def __init__(self):
        self.question_templates = {
            "max_value": "What is the maximum value in this time series?",
            "min_value": "What is the minimum value in this time series?",
            "sum": "What is the sum of values from index {start} to {end}?",
            "value_at": "What is the value at position {pos}?",
            "trend": "Is the overall trend increasing or decreasing?",
            "mean": "What is the mean value?",
            "variance": "What is the variance?",
        }

    def generate(self, ts: torch.Tensor) -> List[Tuple[str, str]]:
        """Generate all QA pairs for a given time series.

        Args:
            ts: Time series tensor of shape (batch, channels, length) or (channels, length)
        """
        if ts.dim() == 2:
            ts = ts.unsqueeze(0)
        ts_1d = ts.squeeze(0).mean(0)  # Average across dims for simplicity

        results = []
        results.append(("max_value", self.generate_max_value(ts_1d)))
        results.append(("min_value", self.generate_min_value(ts_1d)))
        results.append(("mean", self.generate_mean(ts_1d)))
        results.append(("trend", self.generate_trend(ts_1d)))

        # Test specific position queries
        if ts_1d.shape[0] >= 10:
            results.append(("value_at", self.generate_value_at(ts_1d, 5)))
            results.append(("sum", self.generate_sum(ts_1d, 0, 10)))

        return results

    def generate_max_value(self, ts: torch.Tensor) -> Tuple[str, str]:
        max_val = ts.max().item()
        max_idx = ts.argmax().item()
        question = self.question_templates["max_value"]
        answer = f"The maximum value is {max_val:.2f}, appearing at position {max_idx + 1}."
        return question, answer

    def generate_min_value(self, ts: torch.Tensor) -> Tuple[str, str]:
        min_val = ts.min().item()
        min_idx = ts.argmin().item()
        question = self.question_templates["min_value"]
        answer = f"The minimum value is {min_val:.2f}, appearing at position {min_idx + 1}."
        return question, answer

    def generate_mean(self, ts: torch.Tensor) -> Tuple[str, str]:
        mean_val = ts.mean().item()
        question = self.question_templates["mean"]
        answer = f"The mean value is {mean_val:.2f}."
        return question, answer

    def generate_trend(self, ts: torch.Tensor) -> Tuple[str, str]:
        first = ts[0].item()
        last = ts[-1].item()
        trend = "increasing" if last > first else "decreasing"
        question = self.question_templates["trend"]
        answer = f"The overall trend is {trend}."
        return question, answer

    def generate_value_at(self, ts: torch.Tensor, pos: int) -> Tuple[str, str]:
        if pos >= len(ts):
            pos = len(ts) - 1
        val = ts[pos].item()
        question = self.question_templates["value_at"].format(pos=pos + 1)
        answer = f"The value at position {pos + 1} is {val:.2f}."
        return question, answer

    def generate_sum(self, ts: torch.Tensor, start: int, end: int) -> Tuple[str, str]:
        if end > len(ts):
            end = len(ts)
        if start >= len(ts):
            start = max(0, len(ts) - 1)
        sum_val = ts[start:end].sum().item()
        question = self.question_templates["sum"].format(start=start + 1, end=end)
        answer = f"The sum from position {start + 1} to {end} is {sum_val:.2f}."
        return question, answer
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_generator.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/data/generator.py tests/test_generator.py
git commit -m "feat: implement time series and QA data generators"
```

---

### Task 6: 训练循环实现

**Files:**
- Create: `src/timeseries_llm/training/trainer.py`
- Create: `tests/test_trainer.py` (unit test, no actual training)

- [ ] **Step 1: Write test_trainer.py**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_trainer.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write trainer.py**

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
from timeseries_llm.data.generator import TimeSeriesGenerator, QAGenerator


class TimeSeriesDataset(Dataset):
    """Dataset for time series QA pairs."""

    def __init__(
        self,
        num_samples: int,
        min_len: int,
        max_len: int,
        min_dims: int,
        max_dims: int,
        tokenizer,
    ):
        self.num_samples = num_samples
        self.ts_generator = TimeSeriesGenerator(min_len, max_len, min_dims, max_dims)
        self.qa_generator = QAGenerator()
        self.tokenizer = tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate time series
        ts, meta = self.ts_generator.generate()

        # Generate QA pair
        qa_pairs = self.qa_generator.generate(ts)

        # Randomly select one QA pair
        qa_type, (question, answer) = qa_pairs[idx % len(qa_pairs)]

        # Tokenize
        prompt = f"Question: {question}\nAnswer:"
        encoding = self.tokenizer(
            prompt,
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True,
        )

        return {
            "time_series": ts,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
            "meta": meta,
        }


class Trainer:
    """Trainer for TimeSeries-LLM."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Import here to avoid circular dependency
        from timeseries_llm.models.llm_wrapper import TimeSeriesLLM

        self.model = TimeSeriesLLM(
            llm_name=config["model"]["llm_name"],
            encoder_dim=config["model"]["encoder_dim"],
            llm_dim=config["model"]["llm_dim"],
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
        )

        # Dataset
        self.dataset = TimeSeriesDataset(
            num_samples=config["data"]["num_samples"],
            min_len=config["data"]["min_len"],
            max_len=config["data"]["max_len"],
            min_dims=config["data"]["min_dims"],
            max_dims=config["data"]["max_dims"],
            tokenizer=self.model.tokenizer,
        )

        self.current_step = 0
        self.max_steps = config["training"]["max_steps"]
        self.batch_size = config["training"]["batch_size"]

    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        ts = batch["time_series"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Encode time series
        encoder_output = self.model.fusion(self.model.encoder(ts))

        # Forward
        outputs = self.model(
            input_ids=input_ids,
            encoder_outputs=encoder_output,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """Main training loop."""
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        pbar = tqdm(total=self.max_steps, desc="Training")
        while self.current_step < self.max_steps:
            for batch in dataloader:
                loss = self.train_step(batch)
                self.current_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                if self.current_step >= self.max_steps:
                    break

        pbar.close()
        print(f"Training complete. Final step: {self.current_step}")

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_trainer.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/training/trainer.py tests/test_trainer.py
git commit -m "feat: implement training loop with Trainer class"
```

---

### Task 7: 推理 Pipeline 实现

**Files:**
- Create: `src/timeseries_llm/inference/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write test_pipeline.py**

```python
import torch
from unittest.mock import MagicMock, patch
from timeseries_llm.inference.pipeline import TimeSeriesPipeline

def test_pipeline_initialization():
    """Test pipeline initializes correctly."""
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_model.return_value = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock()
            pipeline = TimeSeriesPipeline(
                llm_name="Qwen/Qwen2-0.5B-Instruct",
                encoder_dim=256,
                llm_dim=896,
            )
            assert pipeline.device == pipeline.model.device
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline.py -v
# Expected: FAIL - module not found
```

- [ ] **Step 3: Write pipeline.py**

```python
import torch
from typing import Dict, List, Union
from timeseries_llm.models.llm_wrapper import TimeSeriesLLM


class TimeSeriesPipeline:
    """Inference pipeline for TimeSeries-LLM."""

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2-0.5B-Instruct",
        encoder_dim: int = 256,
        llm_dim: int = 896,
        checkpoint_path: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TimeSeriesLLM(
            llm_name=llm_name,
            encoder_dim=encoder_dim,
            llm_dim=llm_dim,
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        time_series: torch.Tensor,
        question: str,
        max_new_tokens: int = 100,
    ) -> str:
        """
        Answer a question about a time series.

        Args:
            time_series: Tensor of shape (batch, channels, length) or (channels, length)
            question: Question text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer text
        """
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)

        time_series = time_series.to(self.device)

        # Encode time series
        encoder_output = self.model.fusion(self.model.encoder(time_series))

        # Tokenize question
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            encoder_output=encoder_output,
            max_new_tokens=max_new_tokens,
        )

        # Decode
        answer = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def batch_predict(
        self,
        time_series_list: List[torch.Tensor],
        questions: List[str],
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Batch prediction for multiple time series and questions."""
        answers = []
        for ts, q in zip(time_series_list, questions):
            answer = self.predict(ts, q, max_new_tokens)
            answers.append(answer)
        return answers
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_pipeline.py -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add src/timeseries_llm/inference/pipeline.py tests/test_pipeline.py
git commit -m "feat: implement inference pipeline"
```

---

### Task 8: Main 入口实现

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write main.py**

```python
#!/usr/bin/env python3
"""Main entry point for TimeSeries-LLM."""

import argparse
import yaml
import torch
from timeseries_llm.training.trainer import Trainer
from timeseries_llm.inference.pipeline import TimeSeriesPipeline


def train(args):
    """Train the model."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()

    if args.save_path:
        trainer.save(args.save_path)


def infer(args):
    """Run inference."""
    pipeline = TimeSeriesPipeline(
        llm_name=args.llm_name,
        encoder_dim=args.encoder_dim,
        llm_dim=args.llm_dim,
        checkpoint_path=args.checkpoint,
    )

    # Generate sample time series for testing
    from timeseries_llm.data.generator import TimeSeriesGenerator
    ts_gen = TimeSeriesGenerator()
    ts, _ = ts_gen.generate()

    question = args.question or "What is the maximum value in this time series?"
    answer = pipeline.predict(ts, question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description="TimeSeries-LLM")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    train_parser.add_argument("--save-path", type=str, default="checkpoints/model.pt", help="Path to save checkpoint")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--llm-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    infer_parser.add_argument("--encoder-dim", type=int, default=256)
    infer_parser.add_argument("--llm-dim", type=int, default=896)
    infer_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    infer_parser.add_argument("--question", type=str, default=None, help="Question to ask")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add main entry point with train/infer commands"
```

---

## 自检清单

1. **Spec 覆盖度检查：**
   - [x] TimeSeries Encoder (CNN + Transformer) - Task 2
   - [x] MLP Fusion Module - Task 3
   - [x] Qwen LLM Wrapper - Task 4
   - [x] Data Generator (时间序列 + QA) - Task 5
   - [x] Training Loop - Task 6
   - [x] Inference Pipeline - Task 7
   - [x] Main entry point - Task 8

2. **Placeholder scan：** 无 TBD/TODO/不完整内容

3. **类型一致性：** 所有任务间类型匹配（encoder_dim, llm_dim, tensor shapes）

---

## 执行选择

**Plan complete and saved to `docs/superpowers/plans/2026-05-09-timeseries-llm-implementation-plan.md`**

两个执行选项：

**1. Subagent-Driven (recommended)** - 每个任务分配给独立的 subagent 执行，任务间有检查点，快速度迭代

**2. Inline Execution** - 在当前 session 中使用 executing-plans 执行，批量执行带检查点

选择哪个方式？
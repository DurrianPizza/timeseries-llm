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
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)

    def _generate_step(self, length: int, dims: int) -> torch.Tensor:
        data = np.zeros(length)
        num_steps = np.random.randint(1, 5)
        for _ in range(num_steps):
            pos = np.random.randint(0, length)
            step_value = np.random.uniform(-5, 5)
            data[pos:] += step_value
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)

    def _generate_random_walk(self, length: int, dims: int) -> torch.Tensor:
        steps = np.random.randn(length, dims) * 0.5
        data = np.cumsum(steps, axis=0)
        # data shape: (length, dims), transpose to (dims, length)
        data = torch.tensor(data, dtype=torch.float32).transpose(1, 0)
        return data

    def _generate_linear_trend(self, length: int, dims: int) -> torch.Tensor:
        slope = np.random.uniform(-2, 2)
        t = np.arange(length)
        base = slope * t
        noise = np.random.randn(length) * 0.1
        data = base + noise
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)

    def _generate_multi_freq(self, length: int, dims: int) -> torch.Tensor:
        t = np.arange(length) / length
        freq1, freq2 = np.random.uniform(0.1, 0.8, 2)
        data = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)

    def _generate_spike(self, length: int, dims: int) -> torch.Tensor:
        data = np.zeros(length)
        num_spikes = np.random.randint(1, 4)
        for _ in range(num_spikes):
            pos = np.random.randint(0, length)
            value = np.random.uniform(5, 15) * np.random.choice([-1, 1])
            data[pos] = value
        noise = np.random.randn(length) * 0.05
        return torch.tensor(data + noise, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)

    def _generate_periodic(self, length: int, dims: int) -> torch.Tensor:
        period = np.random.randint(10, 50)
        t = np.arange(length)
        data = np.sin(2 * np.pi * t / period) + 0.2 * np.random.randn(length)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(dims, 1)


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

    def generate(self, ts: torch.Tensor) -> List[Tuple[str, Tuple[str, str]]]:
        """Generate all QA pairs for a given time series."""
        if ts.dim() == 2:
            ts = ts.unsqueeze(0)
        ts_1d = ts.squeeze(0).mean(0)

        results = []
        results.append(("max_value", self.generate_max_value(ts_1d)))
        results.append(("min_value", self.generate_min_value(ts_1d)))
        results.append(("mean", self.generate_mean(ts_1d)))
        results.append(("trend", self.generate_trend(ts_1d)))

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
        # Handle 2D tensors by squeezing or taking mean
        if ts.dim() == 2:
            # If first dim is 1, squeeze it; otherwise mean across first dim
            if ts.shape[0] == 1:
                ts = ts.squeeze(0)
            else:
                ts = ts.mean(0)
        if ts.dim() == 1:
            # Now we have a 1D tensor, proceed with slicing
            if end > len(ts):
                end = len(ts)
            if start >= len(ts):
                start = max(0, len(ts) - 1)
            sum_val = ts[start:end].sum().item()
        else:
            # Edge case: scalar tensor
            sum_val = ts.item()
            start, end = 1, 1
        question = self.question_templates["sum"].format(start=start + 1, end=end)
        answer = f"The sum from position {start + 1} to {end} is {sum_val:.2f}."
        return question, answer
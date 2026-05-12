import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple
from timeseries_llm.data.generator import TimeSeriesGenerator, QAGenerator


def collate_fn(batch):
    """Collate function to pad variable-length time series."""
    # Find max channels and length in this batch
    max_channels = max(b["time_series"].shape[0] for b in batch)
    max_len = max(b["time_series"].shape[1] for b in batch)

    # Pad each time series to max size
    padded_ts = []
    for b in batch:
        ts = b["time_series"]
        pad_c = max_channels - ts.shape[0]
        pad_l = max_len - ts.shape[1]
        if pad_c > 0 or pad_l > 0:
            # Pad: (left, right, top, bottom) for (C, L) -> (C+pad_c, L+pad_l)
            ts = torch.nn.functional.pad(ts, (0, pad_l, 0, pad_c))
        padded_ts.append(ts)

    return {
        "time_series": torch.stack(padded_ts),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


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
        show_progress: bool = False,
    ):
        self.num_samples = num_samples
        self.ts_generator = TimeSeriesGenerator(min_len, max_len, min_dims, max_dims)
        self.qa_generator = QAGenerator()
        self.tokenizer = tokenizer

        # Pre-generate all data at initialization
        self.data_pool = []
        iterable = tqdm(range(num_samples), desc="Generating data") if show_progress else range(num_samples)
        for _ in iterable:
            ts, meta = self.ts_generator.generate()
            qa_pairs = self.qa_generator.generate(ts)
            # Store all QA pairs for each time series
            self.data_pool.append((ts, meta, qa_pairs))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ts, meta, qa_pairs = self.data_pool[idx]
        qa_type, (question, answer) = qa_pairs[0]
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
        print("[INFO] Initializing Trainer...")
        self.config = config
        # Check for MPS (Apple Silicon), CUDA, or fall back to CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[INFO] Device: Apple Silicon MPS (GPU)")
        elif torch.cuda.is_available():
            # Verify CUDA actually works (driver might be too old)
            try:
                torch.cuda.init()
                torch.zeros(1).cuda()
                self.device = torch.device("cuda")
                print("[INFO] Device: NVIDIA CUDA (GPU)")
                print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[INFO] CUDA available but not usable: {e}")
                print("[INFO] Falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            print("[INFO] Device: CPU")
        print(f"[INFO] Using device: {self.device}")
        from timeseries_llm.models.llm_wrapper import TimeSeriesLLM
        print("[INFO] Loading LLM model...")
        self.model = TimeSeriesLLM(
            llm_name=config["model"]["llm_name"],
            encoder_dim=config["model"]["encoder_dim"],
            llm_dim=config["model"]["llm_dim"],
            device=str(self.device),
        )
        self.model.to(self.device)

        # Freeze LLM, only train encoder + fusion
        for param in self.model.llm.parameters():
            param.requires_grad = False
        # Check which modules are trainable
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        print(f"[INFO] Trainable parameters: {trainable_params}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
        )
        print(f"[INFO] Pre-generating {config['data']['num_samples']} training samples...")
        self.dataset = TimeSeriesDataset(
            num_samples=config["data"]["num_samples"],
            min_len=config["data"]["min_len"],
            max_len=config["data"]["max_len"],
            min_dims=config["data"]["min_dims"],
            max_dims=config["data"]["max_dims"],
            tokenizer=self.model.tokenizer,
            show_progress=True,
        )
        self.current_step = 0
        self.max_steps = config["training"]["max_steps"]
        self.batch_size = config["training"]["batch_size"]

    def train_step(self, batch: Dict) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        ts = batch["time_series"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        # Encode time series - model.forward will apply fusion internally
        encoder_output = self.model.encoder(ts)
        outputs = self.model(
            input_ids=input_ids,
            encoder_outputs=encoder_output,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        if self.current_step % 10 == 0:
            # Check gradients every 10 steps
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
            print(f"[DEBUG] step={self.current_step}, loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
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
        torch.save({
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
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

        # Detect device: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[INFO] Device: {self.device}")

        self.config = config
        self.mode = config["model"].get("mode", "direct")
        print(f"[INFO] Model mode: {self.mode}")

        from timeseries_llm.models.llm_wrapper import TimeSeriesLLM
        print("[INFO] Loading LLM model...")
        self.model = TimeSeriesLLM(
            llm_name=config["model"]["llm_name"],
            encoder_dim=config["model"]["encoder_dim"],
            llm_dim=config["model"]["llm_dim"],
            device=str(self.device),
            mode=self.mode,
        )

        # Save tokenizer reference
        self.tokenizer = self.model.tokenizer

        # Freeze LLM but keep embed_tokens trainable
        print("[INFO] Freezing LLM transformer layers, training encoder+embed_tokens")
        for name, param in self.model.llm.named_parameters():
            if "embed_tokens" not in name:
                param.requires_grad = False

        # Trainable params: encoder + embed_tokens
        encoder_params = list(self.model.encoder.parameters())
        embed_params = list(self.model.llm.model.embed_tokens.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config["training"]["learning_rate"]},
            {"params": embed_params, "lr": config["training"]["learning_rate"]},
        ], eps=1e-4)

        print(f"[INFO] Trainable params: encoder={len(encoder_params)}, embed={len(embed_params)}")

        # Reconstruction loss weight for encoder_decoder mode
        self.recon_weight = config["training"].get("reconstruction_weight", 0.1)

        print(f"[INFO] Pre-generating {config['data']['num_samples']} training samples...")
        self.dataset = TimeSeriesDataset(
            num_samples=config["data"]["num_samples"],
            min_len=config["data"]["min_len"],
            max_len=config["data"]["max_len"],
            min_dims=config["data"]["min_dims"],
            max_dims=config["data"]["max_dims"],
            tokenizer=self.tokenizer,
            show_progress=True,
        )
        self.current_step = 0
        self.max_steps = config["training"]["max_steps"]
        self.batch_size = config["training"]["batch_size"]

        # Gradient clipping value
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)

    def train_step(self, batch: Dict) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        # Get model dtype for input conversion
        model_dtype = next(self.model.parameters()).dtype

        # Move batch data to device and convert to model dtype
        ts = batch["time_series"].to(self.device, dtype=model_dtype)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass - model handles encoding internally based on mode
        recon_loss = None
        if self.mode == "encoder_decoder":
            logits, reconstructed = self.model(
                input_ids=input_ids,
                raw_ts=ts,
                attention_mask=attention_mask,
            )
            # Compute reconstruction loss (Plan B)
            recon_loss = nn.functional.mse_loss(reconstructed, ts)
            if self.current_step % 10 == 0:
                print(f"[DEBUG] recon_loss={recon_loss.item():.6f}, recon_mean={reconstructed.mean().item():.6f}, ts_mean={ts.mean().item():.6f}")
        else:
            logits = self.model(
                input_ids=input_ids,
                raw_ts=ts,
                attention_mask=attention_mask,
            )

        # logits shape: [batch, text_len + ts_len, vocab]
        # labels shape: [batch, Q+A]
        # Time series is at positions [0:ts_len], text is at positions [ts_len:ts_len+text_len]

        text_len = labels.shape[1]  # Q + A tokens
        ts_len = logits.shape[1] - text_len  # Time series tokens

        # Shift for next-token prediction
        shift_logits = logits[:, ts_len:text_len+ts_len-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Question length (approx tokens in "Question: ... Answer:")
        question_len = 15

        # Create weight mask: only compute loss on answer portion of TEXT
        weights = torch.zeros_like(shift_labels, dtype=torch.float32)
        weights[:, question_len-1:] = 1.0

        # Boost numerical tokens
        with torch.no_grad():
            decoded = [self.tokenizer.decode([tok]) for tok in shift_labels.view(-1)]
            num_boosted = 0
            for i, text in enumerate(decoded):
                if any(c.isdigit() or c in '.-' for c in text):
                    weights.view(-1)[i] *= 100.0
                    num_boosted += 1

        # Get per-token loss
        ce_loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )

        # Apply weights and compute mean
        loss = (ce_loss * weights.view(-1)).sum() / (weights.sum() + 1e-8)

        # Add reconstruction loss for encoder_decoder mode
        if self.mode == "encoder_decoder":
            total_loss = loss + self.recon_weight * recon_loss
        else:
            total_loss = loss
            recon_loss = None

        # Use standard backward
        total_loss.backward()

        # Debug: print grad norms
        if self.current_step % 10 == 0:
            enc_norms = [p.grad.norm().item() for p in self.model.encoder.parameters() if p.grad is not None]
            emb_norms = [p.grad.norm().item() for p in self.model.llm.model.embed_tokens.parameters() if p.grad is not None]
            print(f"[DEBUG] step={self.current_step}, loss={loss.item():.4f}, recon_loss={recon_loss.item() if recon_loss else 0:.4f}")
            print(f"[DEBUG] enc_grad: {enc_norms}")
            print(f"[DEBUG] emb_grad: {emb_norms}")

        # Clip gradients for trainable params only
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def train(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        loss_history = []

        pbar = tqdm(total=self.max_steps, desc="Training")
        while self.current_step < self.max_steps:
            for batch in dataloader:
                loss = self.train_step(batch)
                self.current_step += 1
                loss_history.append(loss)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                if self.current_step >= self.max_steps:
                    break
        pbar.close()

        print(f"\n=== LOSS TRACE ===")
        for i, l in enumerate(loss_history):
            print(f"step {i}: {l:.6f}")
        print(f"=== FINAL LOSS: {loss_history[-1]:.6f} ===")

        return loss_history

    def save(self, path: str):
        torch.save({
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

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

        # Pre-generate all data at initialization
        self.data_pool = []
        for _ in range(num_samples):
            ts, meta = self.ts_generator.generate()
            qa_pairs = self.qa_generator.generate(ts)
            # Store all QA pairs for each time series
            self.data_pool.append((ts, meta, qa_pairs))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ts, meta, qa_pairs = self.data_pool[idx]
        qa_type, (question, answer) = qa_pairs[idx % len(qa_pairs)]
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
        from timeseries_llm.models.llm_wrapper import TimeSeriesLLM
        self.model = TimeSeriesLLM(
            llm_name=config["model"]["llm_name"],
            encoder_dim=config["model"]["encoder_dim"],
            llm_dim=config["model"]["llm_dim"],
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
        )
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
        self.model.train()
        self.optimizer.zero_grad()
        ts = batch["time_series"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        encoder_output = self.model.fusion(self.model.encoder(ts))
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
        torch.save({
            "step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")
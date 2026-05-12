#!/usr/bin/env python3
"""Main entry point for TimeSeries-LLM."""

import argparse
import yaml
from timeseries_llm.training.trainer import Trainer
from timeseries_llm.inference.pipeline import TimeSeriesPipeline
from timeseries_llm.data.generator import TimeSeriesGenerator


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
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = TimeSeriesPipeline(
        llm_name=args.llm_name,
        encoder_dim=args.encoder_dim,
        llm_dim=args.llm_dim,
        checkpoint_path=args.checkpoint,
    )

    data_config = config.get("data", {})
    ts_gen = TimeSeriesGenerator(
        seed=args.seed,
        min_len=data_config.get("min_len", 32),
        max_len=data_config.get("max_len", 2048),
        min_dims=data_config.get("min_dims", 1),
        max_dims=data_config.get("max_dims", 8),
    )
    ts, info = ts_gen.generate()

    print("=" * 60)
    print("生成的时间序列信息:")
    print(f"  Pattern: {info['pattern']}")
    print(f"  Length:  {info['length']}")
    print(f"  Dims:    {info['dims']}")
    print(f"  Shape:   {list(ts.shape)}")
    print("-" * 60)
    print("统计信息:")
    print(f"  Min:    {ts.min().item():.4f}")
    print(f"  Max:    {ts.max().item():.4f}")
    print(f"  Mean:   {ts.mean().item():.4f}")
    print("-" * 60)
    print("前10个值 (dim=0):")
    print(f"  {ts[0, :10].numpy().round(4)}")
    print("=" * 60)

    question = args.question or "What is the maximum value in this time series?"
    answer = pipeline.predict(ts, question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


def main():
    parser = argparse.ArgumentParser(description="TimeSeries-LLM")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    train_parser.add_argument("--save-path", type=str, default="checkpoints/model.pt", help="Path to save checkpoint")

    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--llm-name", type=str, default="Qwen/Qwen3.5-0.8B")
    infer_parser.add_argument("--encoder-dim", type=int, default=256)
    infer_parser.add_argument("--llm-dim", type=int, default=896)
    infer_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    infer_parser.add_argument("--question", type=str, default=None, help="Question to ask")
    infer_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    infer_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path for data settings")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

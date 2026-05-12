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
    pipeline = TimeSeriesPipeline(
        llm_name=args.llm_name,
        encoder_dim=args.encoder_dim,
        llm_dim=args.llm_dim,
        checkpoint_path=args.checkpoint,
    )

    ts_gen = TimeSeriesGenerator(seed=args.seed)  # 添加 seed 支持便于复现
    ts, info = ts_gen.generate()

    print("=" * 60)
    print("生成的时间序列信息:")
    print(f"  Pattern: {info['pattern']}")
    print(f"  Length:  {info['length']}")
    print(f"  Dims:    {info['dims']}")
    print(f"  Shape:   {list(ts.shape)}")
    print("-" * 60)
    print("统计信息:")
    print(f"  Min:    {ts.mean(dim=-1).min().item():.4f}")  # 简化的多维度统计
    print(f"  Max:    {ts.mean(dim=-1).max().item():.4f}")
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
    infer_parser.add_argument("--llm-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    infer_parser.add_argument("--encoder-dim", type=int, default=256)
    infer_parser.add_argument("--llm-dim", type=int, default=896)
    infer_parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    infer_parser.add_argument("--question", type=str, default=None, help="Question to ask")
    infer_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

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

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    train_parser.add_argument("--save-path", type=str, default="checkpoints/model.pt", help="Path to save checkpoint")

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

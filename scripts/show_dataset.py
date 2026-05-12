#!/usr/bin/env python3
"""Show sample data from the time series generator."""

import sys
sys.path.insert(0, "src")

from timeseries_llm.data.generator import TimeSeriesGenerator, QAGenerator

gen = TimeSeriesGenerator(min_len=128, max_len=128, min_dims=8, max_dims=8, seed=42)

print("=" * 70)
print("TimeSeriesGenerator 数据样本展示")
print("=" * 70)

for i in range(5):
    ts, info = gen.generate()
    qa_gen = QAGenerator()
    qas = qa_gen.generate(ts)

    print(f"\n=== 样本 {i+1} ===")
    print(f"Pattern: {info['pattern']}")
    print(f"Shape:   {list(ts.shape)}")
    print(f"Stats:   Min={ts.min().item():.4f}, Max={ts.max().item():.4f}, Mean={ts.mean().item():.4f}")
    print(f"前5个值 (dim=0): {ts[0, :5].numpy().round(4)}")
    print("-" * 70)
    print("QA 对:")
    for q_type, (q, a) in qas:
        print(f"  [{q_type}]")
        print(f"    Q: {q}")
        print(f"    A: {a}")

print("\n" + "=" * 70)
print("训练配置: min_len=128, max_len=128, min_dims=8, max_dims=8")
print("=" * 70)

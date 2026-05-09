# TimeSeries-LLM 对齐理解系统设计

## 1. 目标与动机

**核心目标：** 让 LLM 能够精确理解任意长度、任意维度的时间序列数据，并能回答关于时间序列的精确数值问题（最大值、最小值、特定位置数值、区间和等）。

**技术路线：** TimeSeries Encoder + Cross-Attention 融合 + 端到端 LLM Finetune（方案 1）

**为什么选这条路：**
- 路线 A（图像式 tokenize）：量化误差在图像领域可接受（像素值有冗余），但在时间序列领域数值是 exact 的，量化误差不可接受
- 路线 B（特征+文本）：丢失太多结构信息，无法支持精确数值问答
- 路线 C（对比学习）：时间序列-文本 paired data 规模远小于图文对，对齐效果难以保证
- 本方案：端到端保留数值信息，直接在 LLM decode 阶段输出精确数值

## 2. 架构设计

### 2.1 整体结构

```
时间序列输入 → TimeSeries Encoder → 连续向量 → Cross-Attention Fusion → Qwen2-0.5B-Instruct → 文本输出
                    ↑                                                    ↑
              维度/长度不变                                      问题文本输入
```

### 2.2 TimeSeries Encoder

**输入：** 任意维度（1维或多维）、任意长度的时间序列 tensor

**处理流程：**
1. 每个维度独立通过 1D CNN 提取局部特征（卷积核大小 3，步长 1，保留序列长度）
2. 多维特征拼接后通过 Transformer Encoder 建模全局依赖
3. 输出：序列长度的 token embeddings（每个时间点一个向量）

**关键设计：** 输出是连续向量，不做量化，信息无损

**参考：** PatchTST / TimesFM 的时序编码思路

### 2.3 Fusion Module

**形式：** MLP Projector（两层 linear layer with gelu activation）

**作用：** 将 TimeSeries Encoder 的输出维度映射到 Qwen 的 embedding 维度

### 2.4 LLM Backbone

**模型：** Qwen2-0.5B-Instruct

**输入形式：**
- 问题文本 + 时间序列编码向量（作为 soft prompt 通过 cross-attention 注入）
- 格式：`[问题文本] [时间序列 token1] [时间序列 token2] ... [时间序列 tokenN]`

**输出：** 文本回答，包含精确数值

## 3. 训练方案

### 3.1 数据生成

**时间序列类型（随机混合）：**
- 正弦波 + 噪声
- 阶跃函数
- 随机游走
- 线性趋势 + 噪声
- 多频率混合
- 突增/突降异常
- 周期信号 + 白噪声

**维度：** 1维 ~ 8维（随机）
**长度：** 32 ~ 2048 点（随机）

### 3.2 问答对生成（规则引擎）

**问题类型：**
| 问题类型 | 示例 | 答案生成规则 |
|---------|------|-------------|
| 最大/最小值 | 最大值是多少？ | 遍历求极值，记录位置 |
| 特定位置 | 第5个点的值是多少？ | 直接索引 |
| 区间统计 | 5-10点的和是多少？ | 区间求和 |
| 趋势判断 | 整体趋势是上升还是下降？ | 首尾差值判断 |
| 统计特征 | 均值、方差是多少？ | 直接计算 |
| 形状描述 | 这段数据有什么特点？ | 规则匹配（周期性、突变等）|

**数据量：** 初始目标 10 万条随机 (时间序列, 问题, 回答) pairs

### 3.3 训练目标

- 端到端 language modeling loss（自回归生成）
- Loss 仅在回答文本上计算（时间序列编码和融合模块的梯度正常回传）

## 4. 关键技术问题

### 4.1 多维变长处理

**问题：** 维度不同、长度不同的输入如何统一处理？

**方案：**
- 每个维度独立 CNN 编码 → 维度拼接 → Transformer 全局建模
- 变长通过随机采样固定长度窗口训练；推理时支持任意长度（自动截断或分桶）

### 4.2 数值精度保证

**问题：** 如何确保模型输出精确数值而非近似/模糊描述？

**方案：**
- 训练数据中的数值全部保留精确小数位
- 在答案中强制要求数值格式（如保留2位小数）
- 考虑 copying mechanism（复制时间序列数值到输出）作为额外监督信号

### 4.3 训练稳定性

- 使用 gradient checkpointing 节省显存
- 采用渐近式学习率（warmup + cosine decay）
- 混合精度训练（FP16）

## 5. 项目结构

```
timeseries_llm/
├── data/                    # 数据生成
│   └── generator.py         # 时间序列 + 问答对生成
├── models/
│   ├── encoder.py           # TimeSeries Encoder
│   ├── fusion.py            # MLP Projector
│   └── llm.py               # Qwen wrapper + 训练逻辑
├── training/
│   └── trainer.py           # 训练循环
├── inference/
│   └── pipeline.py          # 推理 pipeline
├── configs/
│   └── default.yaml         # 配置文件
├── docs/
│   └── specs/               # 设计文档
└── main.py                  # 入口
```

## 6. 评估方案

**基础任务：**
- 精确数值问答准确率（最大值、最小值、区间和等）
- 数值误差（MSE / MAE）
- 趋势判断准确率

**分析任务：**
- 描述生成质量（人工评估或 GPT 评估）
- 跨维度/长度泛化能力

## 7. 风险与备选

| 风险 | 应对 |
|-----|-----|
| Qwen 0.5B 容量太小，记不住长序列细节 | 考虑 1.5B 或使用 summarization/reduction 模块 |
| 训练不收敛 | 检查 loss scale，使用 LLaMA-Factory 框架 |
| 数值精度不够 | 加入 copying mechanism 或 separate numerical head |

## 8. 下一步

- [ ] 实现 TimeSeries Encoder
- [ ] 实现数据生成器
- [ ] 搭建训练 pipeline
- [ ] 先跑通 baseline，验证可行性

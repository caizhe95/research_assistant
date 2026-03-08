# 主题总览：Transformer 架构与大模型应用

主题名称：Transformer 架构与大模型应用  
研究目标：从原理、训练、推理与工程落地四个层面，形成一套可讲解、可实现、可扩展的 Transformer 学习与项目路线。  

## 背景

Transformer 自 2017 年提出后，已成为 NLP、多模态与生成式 AI 的主流基础架构。  
相比 RNN/CNN，Transformer 通过自注意力机制更擅长建模长距离依赖，并支持大规模并行训练。  
当前主流 LLM（如 GPT、Llama、Qwen 等）都建立在 Transformer 变体之上。

## 研究边界

本研究只覆盖：
1. Encoder、Decoder 与 Encoder-Decoder 三类 Transformer 结构
2. 文本方向为主，兼顾多模态扩展思路
3. 从“理论理解 + 实践部署”视角组织内容

本研究不覆盖：
1. 芯片微架构与底层 CUDA 内核优化细节
2. 企业级超大规模分布式训练平台搭建
3. 纯学术推导中的复杂数学证明细节

## 关键术语

- Self-Attention：通过 Query/Key/Value 计算 token 间相关性。
- Multi-Head Attention：多子空间并行注意力，提升表达能力。
- Positional Encoding：注入序列位置信息，弥补注意力的置换不变性。
- FFN：逐位置前馈网络，提供非线性变换能力。
- KV Cache：推理阶段缓存历史 Key/Value，降低自回归延迟。

## 学习与项目成功标准

1. 能清楚解释 Transformer 基础模块与数据流
2. 能比较主流架构差异（BERT/GPT/T5）
3. 能说明训练与推理阶段的关键工程瓶颈
4. 能设计一个可落地的小型 Transformer/LLM 应用方案

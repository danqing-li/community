# 【Hackathon 10th Spring No.8】Crystalformer 无限连通注意力晶体性质预测模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-04-01         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_crystalformer_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

Crystalformer 是一种基于 Transformer 的晶体**性质预测**编码器模型，由 Taniai 等人（OMRON SINIC X）在 ICLR 2024 发表，论文标题为 "Crystalformer: Infinitely Connected Attention for Periodic Structure Encoding"。

Crystalformer 的核心任务：给定晶体结构（周期性原子排列），**预测其物理性质**（生成能、带隙、体积模量、剪切模量等），属于**回归任务**。它不是结构生成模型。

核心创新：
1. **无限连通注意力（Infinitely Connected Attention）**：晶体结构是原子的无限周期性排列。标准的全连接注意力只处理有限原子集合；Crystalformer 将其扩展为无限周期超格子上的注意力求和，覆盖所有周期性镜像原子
2. **神经势求和（Neural Potential Summation）**：在注意力权重中引入高斯距离衰减因子，使无限求和收敛，物理解释为在深层抽象特征空间中的原子间势函数求和
3. **伪有限周期注意力（Pseudo-Finite Periodic Attention）**：通过数学改写，将无限连通注意力转化为与标准有限注意力形式相似的公式，周期性信息被编入新的位置编码项 α 和 β
4. **实空间+傅里叶空间双域注意力**：借鉴 Ewald 求和原理，在多头注意力的不同 head 中分别计算实空间和倒空间（傅里叶空间）注意力，更好捕捉长程相互作用
5. **极简高效架构**：仅需 Matformer 29.4% 的参数量，通过对原始 Transformer 的最小修改实现 SOTA 性能

- 原始代码仓库：https://github.com/omron-sinicx/crystalformer（MIT License，28 stars）
- 项目主页：https://omron-sinicx.github.io/crystalformer/
- 论文链接：https://openreview.net/forum?id=fxQiecl9HB
- 后续工作：CrystalFramer（ICLR 2025）—— 引入动态坐标框架扩展
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 3 项）

### 1.2 功能目标

1. 在 `ppmat/models/crystalformer/` 下实现 Crystalformer 编码器模型：无限连通注意力层 + 高斯距离衰减 + 双域位置编码 + 回归 MLP head
2. 迁移自定义 CUDA kernel（原始仓库含 25.3% CUDA 代码，用于加速周期距离计算）
3. 适配 PaddleMaterials 统一的 trainer/predictor 模式
4. 支持 JARVIS-DFT 和 Materials Project（megnet）数据集
5. 前向精度对齐（logits diff 1e-4 量级）
6. 反向训练对齐（训练 2 轮以上 loss 一致）
7. 性质预测指标对标原论文各数据集 MAE
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 现有 Transformer（ComFormer）仅处理有限近邻图，Crystalformer 将引入无限周期注意力范式，形成有限 vs 无限注意力的对比矩阵。
2. Crystalformer 是 ICLR 2024 高影响力工作，在 Materials Project 和 JARVIS-DFT 多项基准上超越 PotNet，且仅需 29.4% 的 Matformer 参数。
3. 复现有助于为 Paddle 生态积累周期性注意力计算的工程经验（Ewald 求和、倒空间编码）。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下模型：

| 已有模型 | 任务类型 | 架构 | 与 Crystalformer 关系 |
|---------|---------|------|---------------------|
| ComFormer | 属性预测 | Transformer | **最相近**——同为 Transformer 编码器，但 ComFormer 用有限近邻图，Crystalformer 用无限周期注意力 |
| DimeNet++ | 属性预测 | 三体 GNN | 可参考径向基函数（RBF）设计 |
| MEGNet | 属性预测 | GNN | 可参考 megnet 数据集加载 |
| MatterGen | 结构生成 | DDPM + GNN | 任务类型不同（生成 vs 预测） |
| CHGNet | 原子间势函数 | GNN | 可参考晶体图表示 |

**关键现状分析**：

1. **缺少无限周期注意力机制**：现有 Transformer（ComFormer）对有限近邻图做注意力。Crystalformer 对所有周期镜像做无限注意力求和——这是一种全新的注意力范式，需从头实现。
2. **缺少双域（实空间+傅里叶空间）注意力**：现有模型仅在实空间操作。Crystalformer 的多头注意力可在实空间和倒空间中分别计算，需新增傅里叶空间位置编码。
3. **缺少高斯距离衰减位置编码**：现有位置编码未考虑周期收敛性。Crystalformer 的 α、β 编码项通过高斯衰减保证无限求和收敛。
4. **自定义 CUDA kernel 迁移**：原始仓库含 CUDA 代码（25.3%），用于加速周期距离计算。需评估是否迁移或用 Paddle 原生算子替代。
5. **数据集可复用**：JARVIS-DFT 和 Materials Project（megnet）数据加载逻辑可参考已有 MEGNet 实现。

已有基础设施可复用：

- MEGNet 数据集加载逻辑：JARVIS-DFT 和 Materials Project 数据集已有下载/缓存/split 实现。
- `ppmat/trainer/BaseTrainer`：训练循环和评估 pipeline 可直接沿用。
- `ppmat/optimizer/build_optimizer`：统一优化器和 cosine 调度器构建。
- YAML + OmegaConf 配置体系。

## 3. 目标调研

### 3.1 模型架构概述

Crystalformer 数据流：晶体结构（pymatgen Structure）→ 原子特征 + 周期性距离矩阵 → 多层无限连通自注意力编码 → 全局池化 → MLP 回归 head → 性质预测值。

**核心组件**：

| 组件 | 功能 | 关键参数 |
|-----|------|---------|
| 原子嵌入 | 元素类型 → 向量表示 | model_dim=128 |
| Infinite Attention Block | 无限连通自注意力 + FFN | num_layers=4-7, head_num |
| Gaussian Distance Decay | 注意力权重高斯衰减 | scale_real (r₀), gauss_lb_real (b) |
| Value Position Encoding | 值向量的径向基位置编码 | value_pe_dist_real=64 (K 个 RBF) |
| Real+Reciprocal Domain | 实空间/傅里叶空间并行 head | domain=real\|multihead\|real-reci |
| MLP Head | 池化后回归输出 | embedding_dim=128 |

**无限连通自注意力核心公式**：

标准注意力：$\text{Attn}(i) = \sum_j \text{softmax}(q_i^\top k_j) v_j$

无限周期注意力：$\text{Attn}(i) = \sum_j \sum_{n \in \mathbb{Z}^3} w_{ij}^{(n)} v_j^{(n)}$

其中 $n$ 遍历所有三维整数平移向量（周期镜像），权重包含高斯距离衰减：

$$w_{ij}^{(n)} \propto \exp\left(q_i^\top k_j + \alpha_{ij}^{(n)}\right), \quad \alpha_{ij}^{(n)} = -\frac{\|p_j + Ln - p_i\|^2}{2r_0^2}$$

通过改写为伪有限形式：

$$\text{Attn}(i) = \sum_j \text{softmax}(q_i^\top k_j + \tilde{\alpha}_{ij}) (v_j + \tilde{\beta}_{ij})$$

其中 $\tilde{\alpha}_{ij}, \tilde{\beta}_{ij}$ 编入了无限周期求和信息。

### 3.2 源码结构分析

| 目录/文件 | 功能 | 复现策略 |
|---------|------|---------|
| `models/` | 注意力层 + Transformer 编码器 | 核心迁移 PyTorch → Paddle |
| `dataloaders/` | JARVIS/megnet 数据集加载 | 适配 ppmat DataLoader |
| `losses/` | L1/MSE/Smooth_L1 损失 | Paddle 原生 |
| `params/latticeformer/default.json` | 默认超参数配置 | 转 YAML |
| `train.py` | 训练入口（单 GPU / 多 GPU） | 适配 ppmat trainer |
| `demo.py` / `demo.sh` | 预训练模型推理 demo | 适配 ppmat predictor |
| `init_datasets.py` | 数据集下载和初始化 | 保留逻辑 |
| `docker/` | Docker 构建（PyTorch 2.1 + CUDA 12.1） | 不迁移 |
| CUDA 代码（25.3%） | 加速周期距离/位置编码计算 | 评估迁移 vs Paddle 替代 |

### 3.3 PyTorch → Paddle API 映射

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.Linear` / `nn.LayerNorm` / `nn.Dropout` | 对应 Paddle API | 一致 |
| `nn.MultiheadAttention` | 自定义实现 | Crystalformer 的注意力是定制的，不使用标准 MHA |
| `torch.scatter` | `paddle.put_along_axis` 或 `ppmat.utils.scatter` | 复用已有工具 |
| `torch.cdist` | `paddle.dist` 或自定义周期距离 | 需处理周期边界 |
| `torch.optim.AdamW` | `paddle.optimizer.AdamW` | 一致 |
| `torch.cuda.amp` | `paddle.amp` | 混合精度训练 |
| Custom CUDA kernels | Paddle 自定义算子或原生替代 | 需评估性能影响 |

**重点迁移项**：
- 自定义 CUDA kernel → 优先用 Paddle 原生算子实现，若性能差距 >20% 则迁移为 Paddle 自定义算子
- 周期距离计算：需实现 minimum image convention 或 Ewald 求和

### 3.4 数据集概况

| 数据集 | 性质 | 单位 | Train | Val | Test |
|-------|------|------|-------|-----|------|
| jarvis__megnet | e_form | eV/atom | 60,000 | 5,000 | 4,239 |
| jarvis__megnet | bandgap | eV | 60,000 | 5,000 | 4,239 |
| jarvis__megnet-bulk | bulk_modulus | log(GPa) | 4,664 | 393 | 393 |
| jarvis__megnet-shear | shear_modulus | log(GPa) | 4,664 | 392 | 393 |
| jarvis__dft_3d_2021 | formation_energy | eV/atom | 44,578 | 5,572 | 5,572 |
| jarvis__dft_3d_2021 | opt_bandgap | eV | 44,578 | 5,572 | 5,572 |
| jarvis__dft_3d_2021-mbj_bandgap | mbj_bandgap | eV | 14,537 | 1,817 | 1,817 |
| jarvis__dft_3d_2021-ehull | ehull | eV | 44,296 | 5,537 | 5,537 |

数据集通过 `init_datasets.py` 下载（JARVIS API + megnet 数据文件）。格式：pymatgen Structure 对象 + 性质标签。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ omron-sinicx/crystalformer（MIT License，28 stars） |
| JAX | ✖ 无（后续 CrystalFramer 仅有 PyTorch） |

目前仅有原始 PyTorch 实现，无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/crystalformer/
│   ├── __init__.py                   # 模块导出 + 注册
│   ├── crystalformer.py              # 主模型（_forward/forward/predict 三层模式）
│   ├── infinite_attention.py         # 无限连通自注意力（核心创新）
│   ├── periodic_encoding.py          # 高斯衰减 + 双域位置编码（α, β）
│   └── utils.py                      # 周期距离计算、RBF 工具
├── ppmat/datasets/jarvis_dataset.py  # JARVIS 数据集适配
├── property_prediction/              # 属性预测任务
│   ├── README.md                     # 任务说明
│   └── configs/crystalformer/
│       ├── megnet_eform.yaml         # megnet formation energy
│       ├── megnet_bandgap.yaml       # megnet bandgap
│       ├── dft3d_eform.yaml          # DFT-3D formation energy
│       └── dft3d_bandgap.yaml        # DFT-3D bandgap
├── examples/crystalformer/
│   ├── README.md
│   ├── train.py
│   └── predict.py

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 crystalformer 导入
# ppmat/datasets/__init__.py   — 新增 build_jarvis 工厂函数注册
```

### 4.2 模型实现

核心模型遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计：

```python
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import paddle
import paddle.nn as nn

class InfiniteAttentionBlock(nn.Layer):
    """无限连通自注意力 + FFN（单层）"""
    def __init__(self, model_dim, head_num, ff_dim, scale_real,
                 gauss_lb_real, value_pe_dist_real, domain='real'):
        super().__init__()
        self.head_num = head_num
        self.head_dim = model_dim // head_num
        self.domain = domain

        # QKV 投影
        self.w_q = nn.Linear(model_dim, model_dim, bias_attr=False)
        self.w_k = nn.Linear(model_dim, model_dim, bias_attr=False)
        self.w_v = nn.Linear(model_dim, model_dim, bias_attr=False)
        self.w_o = nn.Linear(model_dim, model_dim)

        # 高斯距离衰减参数
        self.scale_real = scale_real
        self.gauss_lb_real = gauss_lb_real

        # 值向量位置编码（RBF）
        self.value_pe_dist_real = value_pe_dist_real
        if value_pe_dist_real > 0:
            self.rbf_proj = nn.Linear(value_pe_dist_real, self.head_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, alpha_pe, beta_pe, mask=None):
        """
        x: [B, N, D] 原子特征
        alpha_pe: [B, H, N, N] 注意力权重位置编码（含高斯衰减 + 周期求和）
        beta_pe: [B, H, N, N, K] 值向量位置编码（RBF）
        """
        B, N, D = x.shape
        residual = x
        x = self.norm1(x)

        q = self.w_q(x).reshape([B, N, self.head_num, self.head_dim]).transpose([0, 2, 1, 3])
        k = self.w_k(x).reshape([B, N, self.head_num, self.head_dim]).transpose([0, 2, 1, 3])
        v = self.w_v(x).reshape([B, N, self.head_num, self.head_dim]).transpose([0, 2, 1, 3])

        # 注意力分数 = q·k + α（伪有限周期注意力）
        attn = paddle.matmul(q, k, transpose_y=True) / (self.head_dim ** 0.5)
        attn = attn + alpha_pe  # 加入周期位置编码
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = paddle.nn.functional.softmax(attn, axis=-1)

        # 值向量 = v + β（周期值位置编码）
        if self.value_pe_dist_real > 0:
            v_pe = self.rbf_proj(beta_pe)  # [B, H, N, N, head_dim]
            out = paddle.matmul(attn.unsqueeze(-2), (v.unsqueeze(2) + v_pe)).squeeze(-2)
        else:
            out = paddle.matmul(attn, v)

        out = out.transpose([0, 2, 1, 3]).reshape([B, N, D])
        out = self.w_o(out)
        x = residual + out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class Crystalformer(nn.Layer):
    """Crystalformer 无限连通注意力晶体性质预测模型"""
    def __init__(self, atom_types=118, model_dim=128, ff_dim=512,
                 num_layers=4, head_num=8, embedding_dim=128,
                 scale_real=5.0, gauss_lb_real=0.0,
                 value_pe_dist_real=64, value_pe_dist_max=-3.0,
                 domain='real', loss_func='L1'):
        super().__init__()
        self.atom_embedding = nn.Embedding(atom_types, model_dim)

        self.layers = nn.LayerList([
            InfiniteAttentionBlock(
                model_dim, head_num, ff_dim,
                scale_real, gauss_lb_real, value_pe_dist_real, domain
            ) for _ in range(num_layers)
        ])

        self.pool_norm = nn.LayerNorm(model_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(model_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        if loss_func == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_func == 'MSE':
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.SmoothL1Loss()

    def _forward(self, atom_types, alpha_pe, beta_pe, mask=None):
        """纯前向：原子序列 → 性质预测值"""
        x = self.atom_embedding(atom_types)  # [B, N, D]
        for layer in self.layers:
            x = layer(x, alpha_pe, beta_pe, mask)
        x = self.pool_norm(x)
        # 全局平均池化（仅对有效原子）
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(axis=1) / mask.sum(axis=1, keepdim=True)
        else:
            x = x.mean(axis=1)
        return self.mlp_head(x).squeeze(-1)

    def forward(self, batch):
        """训练入口"""
        pred = self._forward(
            batch['atom_types'], batch['alpha_pe'],
            batch['beta_pe'], batch.get('mask')
        )
        target = batch['target']
        loss = self.loss_fn(pred, target)
        return {"regression_loss": loss}, {"pred": pred}

    @paddle.no_grad()
    def predict(self, batch):
        """推理入口"""
        pred = self._forward(
            batch['atom_types'], batch['alpha_pe'],
            batch['beta_pe'], batch.get('mask')
        )
        return {"pred": pred}
```

### 4.3 周期位置编码计算

```python
# ppmat/models/crystalformer/periodic_encoding.py

import paddle
import numpy as np

def compute_periodic_encoding(structures, scale_real, gauss_lb,
                              value_pe_dist_real, value_pe_dist_max):
    """
    计算伪有限周期注意力的 α 和 β 编码。

    对于每对原子 (i, j)，计算所有周期镜像的高斯衰减求和：
      α̃_ij = log Σ_n exp(-||p_j + Ln - p_i||² / 2r₀²)
      β̃_ij = Σ_n RBF(||p_j + Ln - p_i||) * exp(...)

    其中 L 是晶格矩阵，n 遍历整数平移向量。
    """
    # 实空间：在截断半径内对周期镜像求和
    # 傅里叶空间：对倒格矢做 Ewald 求和
    ...
```

### 4.4 数据集适配

```python
# ppmat/datasets/jarvis_dataset.py

import pickle
import paddle
from paddle.io import Dataset

class JARVISDataset(Dataset):
    """JARVIS-DFT / megnet 数据集"""
    def __init__(self, data_path, target_name, split='train'):
        with open(f'{data_path}/{split}/raw/raw_data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.target_name = target_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        structure = entry['structure']  # pymatgen Structure

        # 提取原子类型和坐标
        atom_types = paddle.to_tensor(
            [s.Z for s in structure.species], dtype='int64'
        )
        frac_coords = paddle.to_tensor(
            structure.frac_coords, dtype='float32'
        )
        lattice = paddle.to_tensor(
            structure.lattice.matrix, dtype='float32'
        )
        target = paddle.to_tensor(
            [entry[self.target_name]], dtype='float32'
        )

        return {
            'atom_types': atom_types,
            'frac_coords': frac_coords,
            'lattice': lattice,
            'target': target,
        }

def build_jarvis(config):
    """JARVIS 数据集工厂函数"""
    return JARVISDataset(
        data_path=config.data_path,
        target_name=config.target_name,
        split=config.get('split', 'train')
    )
```

### 4.5 YAML 配置示例

```yaml
Global:
  task: crystal_property_prediction
  seed: 42
  output_dir: output/crystalformer_mp_ef/

Model:
  type: Crystalformer
  n_layers: 4
  n_heads: 8
  hidden_dim: 256
  alpha_init: 5.0
  beta_init: 1.0
  k_max: 6
  use_fourier_heads: true

Dataset:
  type: jarvis
  data_path: data/crystalformer/megnet_ef/
  target_name: e_form

Optimizer:
  type: Adam
  lr: 1.0e-4
  weight_decay: 1.0e-5
  lr_scheduler: CosineAnnealing
  T_max: 500
```

### 4.6 补充说明

- 原始仓库含 CUDA 代码（25.3%）用于加速周期距离计算。本次复现优先用 Paddle 原生算子替代，仅在性能瓶颈时考虑迁移 CUDA kernel。
- 不复现 7 层变体（论文 ablation 配置），仅复现 4 层标准配置。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同晶体输入 + 预训练权重，对比 PyTorch 和 Paddle 的预测输出
- 预训练权重：Google Drive（megnet bandgap 和 e_form，4 或 7 层）
- 要求：logits diff ≤ 1e-4

### 5.2 反向训练对齐

- 使用相同数据子集和超参数，对比两个框架训练 2 轮以上的 loss 曲线
- 默认超参：lr=5e-4, AdamW, inverse_sqrt_nowarmup schedule
- 要求 loss 逐 epoch 一致

### 5.3 性质预测指标

| 数据集 | 性质 | 原论文 MAE | 允许误差 |
|-------|------|----------|---------|
| megnet | e_form (eV/atom) | 0.0218 | ±5% |
| megnet | bandgap (eV) | 0.159 | ±5% |
| megnet-bulk | bulk_modulus (log GPa) | 0.044 | ±5% |
| megnet-shear | shear_modulus (log GPa) | 0.068 | ±5% |
| dft_3d_2021 | formation_energy (eV/atom) | 0.029 | ±5% |
| dft_3d_2021 | opt_bandgap (eV) | 0.100 | ±5% |

### 5.4 效率验证

- 参数量 ≤ Matformer 29.4%（原论文声称）
- 训练吞吐量在合理范围内

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中高** | 无限连通注意力机制为全新实现，需深入理解伪有限周期注意力数学推导 |
| 自定义算子 | **有** | 原始仓库含 25.3% CUDA 代码（周期距离计算/位置编码加速）。优先用 Paddle 原生实现，性能不足则迁移自定义算子 |
| 依赖项风险 | **低** | pymatgen（晶体结构）、jarvis-tools（数据下载）—— 均有 Paddle 生态对应 |
| 数据可获取性 | **高** | JARVIS API 公开下载，megnet 数据随 `init_datasets.py` 获取 |
| API 映射完整性 | **中** | 注意力层需完全自定义；标准 NN 操作一致 |
| 计算资源 | **中** | megnet: batch_size=128, 500 epochs；dft_3d: batch_size=256, 800 epochs |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | JARVIS/megnet 数据集下载验证 | 2 天 |
| Phase 1：注意力核心 | 无限连通注意力 + 高斯衰减 + 伪有限改写 | 5 天 |
| Phase 2：位置编码 | α/β 编码、双域（实空间+傅里叶空间） | 3 天 |
| Phase 3：CUDA 评估 | 评估原始 CUDA kernel 是否需迁移 | 2 天 |
| Phase 4：训练对齐 | Trainer 适配、loss 曲线对比 | 3 天 |
| Phase 5：评估验证 | 各数据集 MAE 对标 | 4 天 |
| Phase 6：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~21 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `crystalformer/` 目录（~1000 行代码），包含无限连通自注意力层、高斯距离衰减编码和双域位置编码，提供极简参数量（Matformer 29.4%）的 SOTA 晶体性质预测。
2. **新增数据集**：适配 JARVIS-DFT 和 Materials Project (megnet) 数据加载（~150 行），可复用于其他 JARVIS 基准模型。
3. **新增任务类型**：扩展 `property_prediction/` 配置目录，引入无限周期注意力范式，与已有 ComFormer（有限注意力）形成对比矩阵。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| 无限连通注意力 | Infinitely Connected Attention：对晶体所有周期镜像原子做全连接注意力求和 |
| 伪有限周期注意力 | Pseudo-Finite Periodic Attention：通过数学改写将无限求和转化为标准有限注意力形式 |
| 神经势求和 | Neural Potential Summation：无限连通注意力的物理解释——在特征空间做原子间势求和 |
| Ewald 求和 | 将长程周期势分解为实空间短程部分 + 倒空间长程部分，加速收敛 |
| 高斯距离衰减 | 注意力权重中的 $\exp(-r^2/2r_0^2)$ 因子，保证无限求和收敛 |
| 径向基函数 (RBF) | Radial Basis Functions，用于编码原子间距离到高维特征空间 |
| JARVIS-DFT | NIST 材料数据库，包含 DFT 计算的多种材料性质 |
| MAE | Mean Absolute Error，平均绝对误差，性质预测的主要评估指标 |

## 附件及参考资料

1. Taniai, Tatsunori et al. "Crystalformer: Infinitely Connected Attention for Periodic Structure Encoding." *The Twelfth International Conference on Learning Representations (ICLR 2024)*. https://openreview.net/forum?id=fxQiecl9HB
2. Crystalformer 源码：https://github.com/omron-sinicx/crystalformer（MIT License）
3. Crystalformer 项目主页：https://omron-sinicx.github.io/crystalformer/
4. CrystalFramer 后续工作（ICLR 2025）：https://github.com/omron-sinicx/crystalframer
5. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 3 项）

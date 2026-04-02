# 【Hackathon 10th Spring No.9】NewtonNet 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-04-01         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_newtonnet_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

NewtonNet 是一种基于牛顿等变约束的分子动力学势函数模型，由 Haghighatlari 等人在 *Digital Discovery 2022* 发表（DOI: 10.1039/D2DD00008C）。该模型通过显式建模牛顿第三定律（作用力与反作用力相等且反向），确保预测的原子间力满足物理守恒律，在 MD17 等分子动力学基准上取得了优异精度。

NewtonNet 的核心创新在于：
1. **牛顿等变约束**：通过消息传递架构设计，确保预测的原子间力天然满足 F_ij = -F_ji（牛顿第三定律）
2. **力场分解**：将原子间相互作用分解为径向和角向分量，物理可解释性强
3. **高效计算**：相比传统 DFT 计算快 10^6 倍，适用于大规模分子动力学模拟
4. **数据效率**：在少量训练数据下即可达到高精度

- 原始代码仓库：https://github.com/THGLab/NewtonNet
- 论文链接：https://doi.org/10.1039/D2DD00008C
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 4 项）

### 1.2 功能目标

1. 在 `ppmat/models/newtonnet/` 下实现 NewtonNet 模型，包含牛顿等变消息传递层、力场预测头、能量 - 力一致性约束
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 复用已有的 `build_molecule` 工厂函数，支持 MD17、SPICE 等分子动力学数据集
4. 前向精度对齐（能量/力预测 diff 1e-4 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 监督任务 metric 误差控制在 1% 以内（能量 MAE、力 MAE）
7. 覆盖原论文全部基准数据集：MD17（8 个分子）、SPICE 子集
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 现有势函数模型（SchNet、DimeNet++）不保证牛顿第三定律，NewtonNet 将引入等变约束的力场预测范式。
2. 新增 "Molecular Dynamics Potential" 任务类型，与现有 Interatomic Potentials 并列，为后续等变模型（如 PaiNN、Allegro）提供标准任务框架。
3. NewtonNet 是分子动力学模拟领域的重要工作，与 SchNet 形成对比：SchNet 侧重静态属性预测，NewtonNet 侧重动力学模拟。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下模型，涵盖势函数、属性预测等任务类型：

| 已有模型 | 任务类型 | 架构 | 与 NewtonNet 关系 |
|---------|---------|------|-----------------|
| CHGNet | 原子间势函数 | GNN | 可参考力场预测头设计 |
| SchNet | 属性预测 | GNN（连续滤波卷积） | **架构最相近**，可参考距离嵌入、邻居聚合 |
| DimeNet++ | 属性预测 | GNN（三体相互作用） | 可参考角向相互作用建模 |
| MEGNet | 属性预测 | GNN | 可参考全局状态向量设计 |
| MatterSim | 原子间势函数 | GNN | 可参考 trainer 接口、力场 loss 设计 |

**关键现状分析**：

1. **缺少牛顿等变约束**：现有 GNN 模型（SchNet、DimeNet++、MEGNet）预测的原子间力不保证满足牛顿第三定律。NewtonNet 需新增等变消息传递层。
2. **力场预测**：`ppmat/models/` 已有 CHGNet、MatterSim 等势函数模型，可参考其力场预测头设计。
3. **能量 - 力一致性**：NewtonNet 通过能量梯度计算力（E = f(r), F = -∇E），需确保能量和力的训练一致性。
4. **trainer 可复用**：`ppmat/trainer/` 的训练循环支持多任务 loss（能量 + 力），可直接复用。

已有基础设施可复用：

- `ppmat/trainer/`：多任务 loss 训练循环（能量 + 力联合训练已有先例）。
- SchNet 距离嵌入模块：高斯 RBF 和邻居列表构建可参考。
- `ppmat/optimizer/build_optimizer`：统一优化器构建，支持 ReduceOnPlateau 等势函数常用调度器。
- YAML + OmegaConf 配置管线。

## 3. 目标调研

### 3.1 模型架构概述

NewtonNet 的数据流：分子图 → 原子/键特征 → 牛顿等变消息传递（T 层）→ 原子能量 → 总能量 → 力（能量梯度）。

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Atom/Bond Featurizer | 原子/键特征提取 | 原子：原子序数、嵌入维度；键：距离、方向向量 |
| NewtonNet Interaction | 牛顿等变消息传递层 | 消息维度 d=128，层数 T=6 |
| Energy Head | 原子能量预测 | Linear(d, 1) |
| Force Computation | 力 = -∇E（自动微分） | paddle.grad |

**牛顿等变消息传递**（简化）：

```
消息生成：m_{ij} = f_θ(h_i, h_j, r_ij, d_ij)  # r_ij=距离，d_ij=方向向量
力场分解：F_ij = m_{ij} * d_ij  # 径向分量
牛顿约束：F_ji = -F_ij  # 自动满足第三定律
原子力：F_i = Σ_{j∈N(i)} F_ij
```

### 3.2 源码结构分析

原始 PyTorch 实现（`newtonnet/model/newtonnet.py`）核心模块：

| 文件 | 功能 | 行数 | 复现策略 |
|-----|-----|------|---------|
| `newtonnet/model/newtonnet.py` | NewtonNet 主模型 | ~250 | PyTorch → Paddle 逐层转换 |
| `newtonnet/model/interaction.py` | 牛顿等变相互作用层 | ~150 | 核心逻辑迁移 |
| `newtonnet/model/force_layer.py` | 力场计算（自动微分） | ~80 | 适配 paddle.grad |
| `newtonnet/data/ase_dataset.py` | ASE 数据集加载 | ~100 | 适配 ppmat DataLoader |

### 3.3 PyTorch → Paddle API 映射

NewtonNet 仅使用标准 PyTorch NN 操作，无自定义 CUDA kernel，迁移风险极低。

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `torch.autograd.grad` | `paddle.grad` | 力 = -∇E 计算 |
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.Linear` | `paddle.nn.Linear` | 一致 |
| `nn.Embedding` | `paddle.nn.Embedding` | 一致 |
| `torch.norm` | `paddle.norm` | 一致 |
| `torch.einsum` | `paddle.einsum` | 一致 |
| `torch.sum` | `paddle.sum` | 一致 |

**无需迁移的模块**：ASE 数据加载（纯 Python + ASE 库）

### 3.4 数据集概况

| 数据集 | 分子数量 | 属性类型 | 说明 |
|---------|---------|---------|------|
| MD17 | 8 个分子 | 能量 + 力 | 从头算 MD 轨迹（乙醇、丙二醛、阿司匹林等） |
| SPICE | ~1k 分子 | 能量 + 力 | 大规模量子化学计算数据集 |
| QM9 | 134k | 能量 | 量子化学属性（可仅用能量训练） |

MD17 数据集可通过 http://www.sgdml.org/#datasets 下载，SPICE 可通过 HuggingFace 获取。
### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ THGLab/NewtonNet（MIT License） |
| OpenMM / ASE | ✔ 原始代码支持 ASE 集成 |

目前仅有原始 PyTorch 实现，无其他深度学习框架移植。
## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/newtonnet/
│   ├── __init__.py              # 模块导出 + 注册
│   ├── newtonnet.py             # 主模型（等变消息传递 + 能量预测）
│   ├── interaction.py           # 牛顿等变相互作用层
│   └── utils.py                 # 工具函数（距离矩阵、方向向量）
├── ppmat/datasets/md_dataset.py # 分子动力学数据集（MD17/SPICE）+ build_md 工厂函数
├── ppmat/metrics/md_metrics.py  # 分子动力学评估指标（能量 MAE、力 MAE）
├── molecular_dynamics_potential/     # 新增任务类型目录
│   ├── README.md                # 任务类型说明文档
│   └── configs/newtonnet/
│       ├── newtonnet_md17_ethanol.yaml
│       ├── newtonnet_md17_aspirin.yaml
│       └── newtonnet_spice.yaml
├── examples/newtonnet/
│   ├── README.md                # 任务说明文档（含结果及链接）
│   ├── train.py                 # 训练入口
│   ├── predict.py               # 预测入口
│   └── evaluate.py              # 评估入口

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 newtonnet 导入
# ppmat/datasets/__init__.py   — 新增 build_md 工厂函数注册
```

### 4.2 模型实现

核心模型类遵循 PaddleMaterials 统一的 `_forward()`/`forward()`/`predict()` 三层设计。关键代码骨架如下：

```python
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import paddle
import paddle.nn as nn
from ppmat.utils.scatter import scatter

class NewtonNetInteraction(nn.Layer):
    """牛顿等变消息传递层"""
    def __init__(self, hidden_dim=128, cutoff=10.0, n_rbf=50):
        super().__init__()
        self.cutoff = cutoff
        self.distance_expansion = GaussianRBF(n_rbf, cutoff)
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 力场投影（径向分量）
        self.force_proj = nn.Linear(hidden_dim, 1, bias_attr=False)
        
    def forward(self, atom_features, positions, edge_index, edge_vectors):
        """
        positions: [n_atoms, 3] 原子坐标
        edge_vectors: [n_edges, 3] 边向量（j - i）
        返回：原子力 [n_atoms, 3]
        """
        n_atoms = atom_features.shape[0]
        n_edges = edge_index.shape[1]
        
        # 距离计算
        distances = paddle.norm(edge_vectors, axis=-1)
        
        # 距离嵌入（RBF 展开）
        rbf = self.distance_expansion(distances)
        
        # 消息生成：m_{ij} = f(h_i, h_j, r_ij, d_ij)
        h_i = atom_features[edge_index[0]]
        h_j = atom_features[edge_index[1]]
        messages = self.message_net(paddle.concat([h_i, h_j, rbf], axis=-1))
        
        # 力场分解：F_ij = force_proj(m_{ij}) * d_ij（径向分量）
        force_magnitude = self.force_proj(messages)  # [n_edges, 1]
        edge_directions = edge_vectors / (distances.unsqueeze(-1) + 1e-8)
        pairwise_forces = force_magnitude * edge_directions  # [n_edges, 3]
        
        # 牛顿第三定律：F_ji = -F_ij（自动满足，通过 scatter 聚合）
        # 原子力：F_i = Σ_{j→i} F_ji - Σ_{i→j} F_ij
        forces = scatter(pairwise_forces, edge_index[1], dim=0, dim_size=n_atoms, reduce="sum")
        forces -= scatter(pairwise_forces, edge_index[0], dim=0, dim_size=n_atoms, reduce="sum")
        
        return forces

class NewtonNet(nn.Layer):
    """NewtonNet 主模型，遵循 ppmat 的 _forward/forward/predict 三层模式"""
    def __init__(self, hidden_dim=128, n_interactions=6, cutoff=10.0,
                 n_rbf=50, max_z=100):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.interactions = nn.LayerList([
            NewtonNetInteraction(hidden_dim, cutoff, n_rbf) 
            for _ in range(n_interactions)
        ])
        self.energy_head = nn.Linear(hidden_dim, 1)
        
    def _forward(self, atom_types, positions, edge_index, batch_index):
        """纯前向计算，返回原子能量"""
        # 原子嵌入
        h = self.embedding(atom_types)
        
        # 边向量计算
        edge_vectors = positions[edge_index[1]] - positions[edge_index[0]]
        
        # T 层牛顿等变相互作用
        for interaction in self.interactions:
            h = h + interaction(h, positions, edge_index, edge_vectors)
        
        # 原子能量预测
        atomic_energies = self.energy_head(h).squeeze(-1)
        return atomic_energies
    
    def forward(self, atom_types, positions, edge_index, batch_index, 
                targets_energy=None, targets_force=None):
        """
        训练入口，返回 (loss_dict, pred_dict)
        力通过能量梯度自动计算：F = -∇E
        """
        # 启用梯度以计算力
        positions.stop_gradient = False
        
        # 能量计算
        atomic_energies = self._forward(atom_types, positions, edge_index, batch_index)
        
        # 总能量（按分子 scatter sum）
        total_energy = scatter(atomic_energies, batch_index, dim=0, 
                               dim_size=batch_index.max()+1, reduce="sum")
        
        # 力 = -∇E（自动微分，确保能量 - 力一致性）
        grad_outputs = paddle.ones_like(total_energy)
        forces = -paddle.grad(total_energy, positions, grad_outputs=grad_outputs)[0]
        
        pred_dict = {"energy": total_energy, "force": forces}
        loss_dict = {}
        
        if targets_energy is not None:
            energy_loss = nn.functional.mse_loss(total_energy, targets_energy)
            loss_dict["energy_loss"] = energy_loss
        
        if targets_force is not None:
            force_loss = nn.functional.mse_loss(forces, targets_force)
            loss_dict["force_loss"] = force_loss
        
        if loss_dict:
            # 加权总 loss（默认 energy:force = 1:100）
            loss_dict["total_loss"] = loss_dict.get("energy_loss", 0) + \
                                      100 * loss_dict.get("force_loss", 0)
        
        return loss_dict, pred_dict
    
    def predict(self, atom_types, positions, edge_index, batch_index):
        """推理入口"""
        atomic_energies = self._forward(atom_types, positions, edge_index, batch_index)
        total_energy = scatter(atomic_energies, batch_index, dim=0,
                               dim_size=batch_index.max()+1, reduce="sum")
        
        positions.stop_gradient = False
        grad_outputs = paddle.ones_like(total_energy)
        forces = -paddle.grad(total_energy, positions, grad_outputs=grad_outputs)[0]
        
        return {"energy": total_energy, "force": forces}
```

**设计要点**：
1. **ppmat 三层模式**：`_forward()` 纯计算 → `forward()` 返回 `(loss_dict, pred_dict)` → `predict()` 推理入口
2. **牛顿第三定律**：通过 `F_i = Σ_{j→i} F_ji - Σ_{i→j} F_ij` 确保自动满足 F_ij = -F_ji
3. **能量 - 力一致性**：力通过 `paddle.grad` 从能量计算，确保物理一致性
4. **scatter 聚合**：复用 `ppmat.utils.scatter` 进行邻居聚合，与 SchNet/DimeNet++ 一致
5. **力场加权**：force_loss 权重 100× energy_loss（原论文经验值）

### 4.3 数据集适配

```python
# ppmat/datasets/md_dataset.py
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
import os
import numpy as np
import paddle
from paddle.io import Dataset
from ppmat.utils.download import download_url

class MDDataset(Dataset):
    """分子动力学数据集，支持 MD17、SPICE 等"""
    def __init__(self, data_path, dataset_name='md17_ethanol', split='train', auto_download=True):
        if auto_download and not os.path.exists(data_path):
            self._download(data_path, dataset_name, split)
        self.data = np.load(data_path, allow_pickle=True).item()
        
    def _download(self, data_path, dataset_name, split):
        """从 BCS 自动下载预处理数据文件"""
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        url = f"https://paddle-org.bj.bcebos.com/paddlematerials/datasets/newtonnet/{dataset_name}_{split}.npz"
        download_url(url, os.path.dirname(data_path))
    
    def __len__(self):
        return len(self.data['atom_types'])
    
    def __getitem__(self, idx):
        return {
            'atom_types': paddle.to_tensor(self.data['atom_types'][idx], dtype='int64'),
            'positions': paddle.to_tensor(self.data['positions'][idx], dtype='float32'),
            'edge_index': paddle.to_tensor(self.data['edge_index'][idx], dtype='int64'),
            'energy': paddle.to_tensor(self.data['energies'][idx], dtype='float32'),
            'force': paddle.to_tensor(self.data['forces'][idx], dtype='float32')
        }

def build_md(config):
    """分子动力学数据集工厂函数，注册到 ppmat/datasets/__init__.py"""
    return MDDataset(
        data_path=config.data_path,
        dataset_name=config.get('dataset_name', 'md17_ethanol'),
        split=config.get('split', 'train'),
        auto_download=config.get('auto_download', True)
    )
```

### 4.4 Trainer 适配

继承 `ppmat/trainer/base_trainer.py`，关键适配点：

1. **损失函数**：能量 loss（MSE）+ 力 loss（MSE，权重 100×）
2. **优化器**：Adam，学习率调度（ReduceLROnPlateau）
3. **评估指标**：能量 MAE、力 MAE

示例配置片段（`molecular_dynamics_potential/configs/newtonnet/newtonnet_md17_ethanol.yaml`）：

```yaml
model:
  type: NewtonNet
  hidden_dim: 128
  n_interactions: 6
  cutoff: 10.0
  n_rbf: 50
  max_z: 100

data:
  type: md
  dataset_name: md17_ethanol
  train_path: data/newtonnet/md17_ethanol_train.npz
  val_path: data/newtonnet/md17_ethanol_val.npz
  auto_download: true

trainer:
  max_epochs: 500
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 0.0
  lr_scheduler: ReduceLROnPlateau
  factor: 0.5
  patience: 50
  force_loss_weight: 100
  grad_clip: 1.0
  log_interval: 10
  eval_interval: 1
```

### 4.5 补充说明

- 不复现 ASE Calculator 集成（属于推理侧接口，非训练核心），但预留 `predict()` 接口供后续对接。
- MD17 的 revised 版本（rMD17）与原始 MD17 split 不同，本次复现使用原论文的原始 MD17 split。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同随机种子和输入分子，对比 PyTorch 和 Paddle 实现的能量/力输出
- 要求：能量 diff ≤ 1e-4，力 diff ≤ 1e-3（力的数值范围更大）

### 5.2 反向训练对齐

- 使用相同数据子集，对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 epoch 一致

### 5.3 监督任务指标

在 MD17 基准数据集上，对比以下指标：

| 分子 | 能量 MAE (meV) | 力 MAE (meV/Å) | 原论文值 | 允许误差 |
|------|---------------|----------------|---------|---------|
| Ethanol | ~1 | ~10 | 论文 Table 2 | ±1% |
| Malonaldehyde | ~1 | ~15 | 论文 Table 2 | ±1% |
| Aspirin | ~2 | ~20 | 论文 Table 2 | ±1% |

### 5.4 测试项

1. **单元测试**：等变消息传递层、能量-力梯度一致性、RBF 嵌入计算
2. **集成测试**：完整的数据加载 → 前向 → loss 计算 → 反向更新流程
3. **精度对齐截图**：同输入下 PyTorch vs Paddle logits diff 截图

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中** | 标准 GNN 变体，牛顿等变逻辑需仔细实现 |
| 自定义算子 | **无** | 全部为标准 NN 操作 + paddle.grad |
| 依赖项风险 | **低** | ASE（已在 ppmat 生态中使用） |
| 数据可获取性 | **高** | MD17 公开下载，SPICE 可通过 HuggingFace 获取 |
| API 映射完整性 | **高** | 所有 PyTorch API 均有 Paddle 对应实现 |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | AI Studio V100 环境配置、ASE 安装 | 2 天 |
| Phase 1：模型迁移 | NewtonNet 相互作用层、能量 - 力计算 → Paddle | 4 天 |
| Phase 2：训练对齐 | Trainer 适配、能量 + 力联合训练 | 3 天 |
| Phase 3：数据集 | MD17 预处理、build 工厂函数注册 | 3 天 |
| Phase 4：评估验证 | MD17 完整 train→evaluate，指标对齐 | 4 天 |
| Phase 5：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~18 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `newtonnet/` 目录（~350 行代码），包含牛顿等变消息传递层和能量-力一致性计算，提供开箱即用的 MD 力场预测能力。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `md_dataset.py`（~100 行），支持 MD17、SPICE 等分子动力学数据集的自动下载与加载。
3. **新增任务类型**：新建 `molecular_dynamics_potential/` 目录，引入“分子动力学势函数”任务类型，与已有 Interatomic Potentials 并列。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| 牛顿第三定律 | 作用力与反作用力大小相等、方向相反（F_ij = -F_ji） |
| 等变消息传递 | 消息传递架构设计确保输出满足物理对称性约束 |
| 能量 - 力一致性 | 力通过能量梯度计算（F = -∇E），确保物理一致性 |

## 附件及参考资料

1. Haghighatlari et al. *"NewtonNet: a Newtonian message passing network for deep learning of interatomic potentials and forces."* Digital Discovery, 2022. DOI: 10.1039/D2DD00008C
2. NewtonNet 源码：https://github.com/THGLab/NewtonNet
3. MD17 数据集：http://www.sgdml.org/#datasets
4. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 4 项）
4. PaddleMaterials issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194
5. SchNet 实现（参考）：`.checkpoints/h10/task-018_claimed/design/20260323_schnet_model_reproduction.md`

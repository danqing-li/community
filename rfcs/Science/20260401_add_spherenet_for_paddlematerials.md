# 【Hackathon 10th Spring No.19】SphereNet 模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-04-01         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_spherenet_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

SphereNet 是一种球面等变图神经网络，用于分子和材料的 3D 几何深度学习。该模型由 Liu 等人在 *ICLR 2022* 发表（arXiv:2110.05170），通过显式建模键角和二面角信息，实现了对分子 3D 结构的完整几何表示。

SphereNet 的核心创新在于：
1. **球面坐标消息传递**：在球面坐标系下进行消息传递，天然捕捉键角信息
2. **二面角感知**：引入二面角（torsion angle）特征，捕捉 3D 构象信息
3. **SE(3) 等变性**：模型输出满足旋转、平移等变性，物理意义明确

- 原始代码仓库：https://github.com/divelab/DIG （SphereNet 在 `dig.threedgraph.method.SphereNet` 模块中）
- 论文链接：https://arxiv.org/abs/2110.05170
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 14 项）

### 1.2 功能目标

1. 在 `ppmat/models/spherenet/` 下实现 SphereNet 模型，包含球面消息传递层、角度/二面角特征计算
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 复用已有的 `build_molecule` 工厂函数，支持 QM9、MD17 等标准数据集
4. 前向精度对齐（监督任务要求 logits diff 1e-4 量级）
5. 监督任务 metric 误差控制在 1% 以内（MAE/RMSE）

### 1.3 意义

- 填补 PaddleMaterials 在 **球面等变 GNN** 领域的空白（现有 GNN 模型如 DimeNet++ 使用笛卡尔坐标）
- 新增任务类型 **"Spherical Equivariant GNN"（球面等变 GNN）**
- SphereNet 是几何深度学习的重要工作，复现有助于提升 PaddleMaterials 在 3D 分子表示学习领域的能力

## 2. PaddleScience 现状

| 已有模型 | 任务类型 | 架构 | 与 SphereNet 关系 |
|---------|---------|------|-----------------|
| DimeNet++ | 属性预测 | GNN（三体相互作用） | **架构最相近**，可参考角度特征计算 |
| SchNet | 属性预测 | GNN（二体距离） | 可参考距离嵌入 |
| MEGNet | 属性预测 | GNN | 可参考全局状态向量 |

**关键现状分析**：
1. **缺少球面坐标表示**：现有 GNN 模型（DimeNet++、SchNet）使用笛卡尔坐标和距离。SphereNet 需新增球面坐标消息传递。
2. **二面角特征**：DimeNet++ 仅建模键角（三体），SphereNet 进一步建模二面角（四体）。
3. **SE(3) 等变性**：需确保模型输出满足旋转、平移等变性。

## 3. 目标调研

### 3.1 模型架构

SphereNet 的数据流：分子图 → 原子/键特征 → 球面消息传递（T 层）→ 原子表示 → 属性预测。

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Spherical Embedding | 球面坐标嵌入（距离 + 键角 + 二面角） | 距离 RBF（n_rbf=50）、角度嵌入（n_angle=8）、二面角嵌入（n_torsion=8） |
| Spherical Message Passing | 球面消息传递层 | 消息维度 d=128，层数 T=4 |
| Interaction Block | 原子-键-角度-二面角交互 | 可堆叠多层 |

**球面消息传递公式**（简化）：

```
球面特征：(r_ij, θ_ijk, φ_ijkl) — 距离、键角、二面角
消息生成：m_{ijk} = f(h_i, h_j, h_k, r_ij, θ_ijk)
二面角更新：m_{ijkl} = g(m_{ijk}, m_{ijl}, φ_ijkl)
原子更新：h_i = Σ_{j,k,l} m_{ijkl}
```

### 3.2 源码结构分析

原始 PyTorch 实现（`SphereNet/spherenet.py`）核心模块：

| 文件 | 功能 | 行数 | 复现策略 |
|-----|-----|------|---------|
| `spherenet.py` | SphereNet 主模型 | ~300 | PyTorch → Paddle 逐层转换 |
| `embedding.py` | 球面嵌入（RBF + 角度） | ~100 | 纯数学计算，基本不变 |
| `interaction.py` | 球面交互层 | ~200 | 核心逻辑迁移 |

### 3.3 PyTorch → Paddle API 映射

| PyTorch API | Paddle API |
|------------|-----------|
| `torch.nn.Module` | `paddle.nn.Layer` |
| `nn.Embedding` | `paddle.nn.Embedding` |
| `nn.Linear` | `paddle.nn.Linear` |
| `torch.acos` | `paddle.acos` |
| `torch.atan2` | `paddle.atan2` |
| `torch.cross` | `paddle.linalg.cross`（需自定义） |
| `torch.norm` | `paddle.norm` |

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/spherenet/
│   ├── __init__.py
│   ├── spherenet.py             # 主模型
│   ├── spherical_embedding.py   # 球面嵌入（RBF + 角度 + 二面角）
│   └── interaction.py           # 球面交互层
├── ppmat/datasets/spherenet_dataset.py  # 数据集（复用 build_molecule）
├── spherical_equivariant_gnn/
│   ├── README.md
│   └── configs/spherenet/
│       ├── spherenet_qm9.yaml
│       └── spherenet_md17.yaml
```

### 4.2 模型实现骨架

```python
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
import paddle
import paddle.nn as nn
import numpy as np
from ppmat.utils.scatter import scatter

class SphericalEmbedding(nn.Layer):
    """球面嵌入：距离 RBF + 角度嵌入 + 二面角嵌入"""
    def __init__(self, cutoff=10.0, n_rbf=50, n_angle=8, n_torsion=8):
        super().__init__()
        self.cutoff = cutoff
        # 距离 RBF
        self.rbf = GaussianRBF(n_rbf, 0, cutoff)
        # 角度嵌入（可学习）
        self.angle_embed = nn.Embedding(n_angle, n_angle)
        # 二面角嵌入（可学习）
        self.torsion_embed = nn.Embedding(n_torsion, n_torsion)
    
    def forward(self, distances, angles, torsions):
        """
        distances: [n_edges] 原子间距离
        angles: [n_triplets] 键角
        torsions: [n_quadruplets] 二面角
        """
        rbf_feat = self.rbf(distances)
        # 角度离散化 + 嵌入
        angle_bins = paddle.clip((angles / np.pi * self.angle_embed.num_embeddings).astype('int64'), 
                                  0, self.angle_embed.num_embeddings - 1)
        angle_feat = self.angle_embed(angle_bins)
        # 二面角离散化 + 嵌入
        torsion_bins = paddle.clip(((torsions + np.pi) / (2 * np.pi) * self.torsion_embed.num_embeddings).astype('int64'),
                                    0, self.torsion_embed.num_embeddings - 1)
        torsion_feat = self.torsion_embed(torsion_bins)
        return rbf_feat, angle_feat, torsion_feat

class SphereInteraction(nn.Layer):
    """球面消息传递交互层"""
    def __init__(self, hidden_dim=128, n_rbf=50, n_angle=8, n_torsion=8):
        super().__init__()
        # 消息网络（距离 + 角度 + 二面角）
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 3 + n_rbf + n_angle + n_torsion, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        # 原子更新
        self.atom_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, atom_features, edge_index, triplet_index, quadruplet_index,
                rbf_feat, angle_feat, torsion_feat):
        """
        triplet_index: [n_triplets, 3] — (i, j, k) 原子索引
        quadruplet_index: [n_quadruplets, 4] — (i, j, k, l) 原子索引
        """
        # 消息生成（基于四元组）
        h_i = atom_features[quadruplet_index[:, 0]]
        h_j = atom_features[quadruplet_index[:, 1]]
        h_k = atom_features[quadruplet_index[:, 2]]
        
        message_input = paddle.concat([h_i, h_j, h_k, rbf_feat, angle_feat, torsion_feat], axis=-1)
        messages = self.message_net(message_input)
        
        # 消息聚合（按中心原子 j 聚合）
        n_atoms = atom_features.shape[0]
        aggregated = scatter(messages, quadruplet_index[:, 1], dim=0, dim_size=n_atoms, reduce="sum")
        
        # 原子更新（residual 连接）
        new_features = atom_features + self.atom_update(aggregated)
        return new_features

class SphereNet(nn.Layer):
    """SphereNet 主模型"""
    def __init__(self, hidden_dim=128, n_interactions=4, cutoff=10.0,
                 n_rbf=50, n_angle=8, n_torsion=8, max_z=100):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.spherical_embedding = SphericalEmbedding(cutoff, n_rbf, n_angle, n_torsion)
        self.interactions = nn.LayerList([
            SphereInteraction(hidden_dim, n_rbf, n_angle, n_torsion)
            for _ in range(n_interactions)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _forward(self, atom_types, positions, edge_index, triplet_index, quadruplet_index):
        """纯前向计算"""
        # 原子嵌入
        h = self.embedding(atom_types)
        
        # 几何特征计算
        distances = self._compute_distances(positions, edge_index)
        angles = self._compute_angles(positions, triplet_index)
        torsions = self._compute_torsions(positions, quadruplet_index)
        
        # 球面嵌入
        rbf_feat, angle_feat, torsion_feat = self.spherical_embedding(distances, angles, torsions)
        
        # T 层球面交互
        for interaction in self.interactions:
            h = interaction(h, edge_index, triplet_index, quadruplet_index,
                           rbf_feat, angle_feat, torsion_feat)
        
        # 输出（分子属性）
        output = self.output_layer(h).squeeze(-1)
        return output
    
    def forward(self, batch, return_loss=True, return_prediction=True):
        """训练入口，返回 (loss_dict, pred_dict)"""
        pred = self._forward(
            batch['atom_types'], batch['positions'], 
            batch['edge_index'], batch['triplet_index'], batch['quadruplet_index']
        )
        
        pred_dict = {"prediction": pred}
        loss_dict = {}
        
        if return_loss and 'targets' in batch:
            loss = nn.functional.mse_loss(pred, batch['targets'])
            loss_dict["mse_loss"] = loss
        
        return loss_dict, pred_dict
    
    def _compute_distances(self, positions, edge_index):
        """计算原子间距离"""
        return paddle.norm(positions[edge_index[:, 1]] - positions[edge_index[:, 0]], axis=-1)
    
    def _compute_angles(self, positions, triplet_index):
        """计算键角（i-j-k）"""
        vec_ji = positions[triplet_index[:, 0]] - positions[triplet_index[:, 1]]
        vec_jk = positions[triplet_index[:, 2]] - positions[triplet_index[:, 1]]
        # cos(θ) = (ji · jk) / (|ji| * |jk|)
        cos_angle = paddle.sum(vec_ji * vec_jk, axis=-1) / (
            paddle.norm(vec_ji, axis=-1) * paddle.norm(vec_jk, axis=-1) + 1e-8
        )
        return paddle.acos(paddle.clip(cos_angle, -1.0, 1.0))
    
    def _compute_torsions(self, positions, quadruplet_index):
        """计算二面角（i-j-k-l）"""
        # 实现略（参考原论文公式）
        ...
```

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同随机种子和输入分子，对比 PyTorch 和 Paddle 实现的输出
- 要求：输出 diff ≤ 1e-4 量级

### 5.2 监督任务指标

在 QM9 基准数据集上，对比以下指标：

| 属性 | 单位 | 原论文 MAE | 允许误差 |
|------|------|-----------|---------|
| U0 | meV | ~30 | ±1% |
| HOMO | meV | ~50 | ±1% |
| LUMO | meV | ~40 | ±1% |

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中高** | 球面坐标计算、二面角特征需仔细实现 |
| 自定义算子 | **无** | 全部为标准 NN 操作 + 几何计算 |
| 依赖项风险 | **低** | 无额外依赖 |
| API 映射完整性 | **高** | 所有 PyTorch API 均有 Paddle 对应实现 |

### 6.2 排期规划

| 阶段 | 内容 | 工期 |
|-----|------|------|
| Phase 0 | 环境搭建、原始实现分析 | 2 天 |
| Phase 1 | 球面嵌入、几何特征计算 | 5 天 |
| Phase 2 | 交互层、主模型 | 4 天 |
| Phase 3 | 训练、评估 | 4 天 |
| Phase 4 | 文档、PR | 2 天 |

**合计**：~17 天

## 7. 影响面

1. **新增模型**：`ppmat/models/spherenet/`（新增 ~400 行代码）
2. **新增任务类型**：`spherical_equivariant_gnn/`
3. **修改已有文件**（仅新增注册行）

## 附件及参考资料

1. Liu, M. et al. *"Spherical Message Passing for 3D Molecular Graphs."* ICLR 2022. https://openreview.net/forum?id=givsRXsOt9r
2. SphereNet 源码（DIG 库）：https://github.com/divelab/DIG — `dig/threedgraph/method/spherenet/`
3. DIG: Dive into Graphs（JMLR 2021）：http://jmlr.org/papers/v22/21-0343.html
4. DimeNet++ 实现（参考）：`ppmat/models/dimenetpp/dimenetpp.py`
5. PaddleMaterials 模型复现共建计划：https://github.com/PaddlePaddle/PaddleMaterials/issues/194

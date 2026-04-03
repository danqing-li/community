# 【Hackathon 10th Spring No.16】DiffSyn 沸石合成条件扩散模型复现

> RFC 文档相关记录信息

|              |                    |
| ------------ | -----------------  |
| 提交作者      | cloudforge1        |
| 提交时间      | 2026-04-01         |
| RFC 版本号    | v1.0               |
| 依赖飞桨版本  | develop            |
| 文件名        | 20260401_add_diffsyn_for_paddlematerials.md |

## 1. 概述

### 1.1 相关背景

DiffSyn（Diffusion Synthesis）是一种基于扩散模型的沸石（zeolite）合成条件生成方法，由 Pan, Kwon, Liu 等人（MIT）在 *Nature Computational Science* (2026) 发表，论文标题为 "DiffSyn: A generative diffusion approach to materials synthesis planning"。部分成果曾在 NeurIPS 2024 AI for Materials Workshop 作为 Oral Spotlight 发表。

DiffSyn 的核心任务：给定目标沸石拓扑结构和有机结构导向剂（OSDA），**生成对应的合成配方参数**（结晶温度、H₂O/T 比、结晶时间、gel 组成等），而非生成分子或晶体结构本身。

核心创新：
1. **合成参数空间扩散**：在连续合成参数空间（温度、时间、摩尔比等）上进行条件扩散去噪，而非原子坐标扩散
2. **GNN 条件编码**：通过图神经网络分别编码沸石拓扑结构（从 CIF 构建 3D 图）和 OSDA 分子图（2D SMILES 图），作为扩散模型的条件输入
3. **Classifier-free guidance**：使用 classifier-free 条件引导，生成时控制合成参数与目标沸石/OSDA 的匹配度
4. **一对多生成**：同一沸石+OSDA 可生成多组合成配方（反映实验中同一体系的合成条件分布性）

- 原始代码仓库：https://github.com/eltonpan/zeosyn_gen（MIT License，36 stars）
- 论文链接：https://www.nature.com/articles/s43588-025-00949-9
- 训练数据：ZeoSyn 数据集（23,961 条沸石合成路线，233 种拓扑，921 种 OSDA）
- 关联 issue：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（模型列表第 11 项）

### 1.2 功能目标

1. 在 `ppmat/models/diffsyn/` 下实现 DiffSyn 条件扩散模型：沸石图编码器 + OSDA 图编码器 + 合成参数扩散去噪网络
2. 适配 PaddleMaterials 统一的 trainer/predictor/sampler 模式
3. 支持 ZeoSyn 数据集加载（沸石拓扑图 + OSDA 分子图 + 合成参数向量）
4. 前向精度对齐（logits diff 1e-4 量级）
5. 反向训练对齐（训练 2 轮以上 loss 一致）
6. 生成质量指标对标原论文（Wasserstein 距离 ~0.423）
7. 实现 classifier-free guidance 采样
8. 添加对应的任务 README 文档

### 1.3 意义

1. PaddleMaterials 现有模型面向结构生成和属性预测，尚无合成条件预测能力。DiffSyn 将建立逆向合成设计范式，支持从目标结构反推合成配方。
2. DiffSyn 是合成条件预测领域首个扩散模型（*Nature Computational Science* 发表），代表该方向的 SOTA。
3. 沸石合成路线设计直接影响石油裂化、CO₂ 捕集等工业效率，该范式可扩展到 MOF、共晶等其他材料体系。

## 2. PaddleScience 现状

PaddleMaterials 套件目前已集成以下模型：

| 已有模型 | 任务类型 | 架构 | 与 DiffSyn 关系 |
|---------|---------|------|----------------|
| MatterGen | 晶体结构生成 | DDPM + GNN | 可参考扩散 scheduler 和 GNN 编码器 |
| DiffCSP | 晶体结构预测 | DDPM | 可参考晶体图表示 |
| DimeNet++ | 属性预测 | 三体 GNN | 可参考 3D 图构建和消息传递 |

**关键现状分析**：

1. **缺少合成参数预测任务**：现有模型面向晶体结构生成/原子属性预测，不涉及合成条件（温度、时间、gel 配比等连续参数）预测。需新增任务类型。
2. **缺少沸石图表示**：现有图构建基于 CIF → 原子邻居图。DiffSyn 需沸石拓扑图（从 CIF 提取 T-site 连接），以及 OSDA 分子图（从 SMILES 构建）。需新增两种图构建器。
3. **缺少条件扩散 + classifier-free guidance**：现有 MatterGen/DiffCSP 使用无条件或简单条件扩散。DiffSyn 需 classifier-free guidance 采样，且扩散空间是连续参数（非坐标）。
4. **GNN 编码器可参考**：DimeNet++ 的 3D 图消息传递逻辑可部分复用于沸石 3D 图编码。

已有基础设施可复用：

- MatterGen 扩散 scheduler：噪声调度与采样逻辑可复用。
- DimeNet++ 3D 消息传递：沸石 3D 图编码部分可参考。
- `ppmat/trainer/BaseTrainer`：训练循环和评估 pipeline。
- YAML + OmegaConf 配置体系。

## 3. 目标调研

### 3.1 模型架构概述

DiffSyn 数据流：沸石 CIF → 3D 拓扑图 → GNN 编码 → zeolite embedding；OSDA SMILES → 2D 分子图 → GNN 编码 → OSDA embedding；两个 embedding 作为条件 → 条件扩散模型在合成参数空间去噪 → 生成合成配方。

**核心组件**：

| 组件 | 功能 | 参数 |
|-----|------|------|
| Zeolite Graph Encoder | 沸石拓扑 3D 图 → 向量表示 | GNN, hidden=128 |
| OSDA Graph Encoder | OSDA 分子 2D 图 → 向量表示 | GNN, hidden=128 |
| Conditional Diffusion | 在合成参数空间条件去噪 | MLP denoiser, T=1000 |
| Classifier-free Guidance | 无条件/有条件混合训练 + 引导采样 | cond_drop_prob=0.1, cond_scale |

**合成参数空间（扩散对象）**：

| 参数 | 类型 | 说明 |
|------|------|------|
| Crystallization Temperature | 连续 (°C) | 结晶温度 |
| H₂O/T ratio | 连续 | 水/T-site 摩尔比 |
| Crystallization Time | 连续 (h) | 结晶时间 |
| Gel composition ratios | 连续向量 | SiO₂/Al₂O₃/Na₂O 等摩尔比 |

### 3.2 源码结构分析

| 文件 | 功能 | 复现策略 |
|-----|------|---------|
| `models/diffusion.py` | 条件扩散模型（MLP denoiser + DDPM） | 核心迁移：PyTorch → Paddle |
| `data/utils.py` | 数据预处理和工具函数 | Python 逻辑不变 |
| `data/zeo2graph.pkl` | 预构建沸石拓扑图 | 直接使用 |
| `data/smiles2graph.pkl` | 预构建 OSDA 分子图 | 直接使用 |
| `data/ZeoSynGen_dataset.pkl` | 完整数据集对象 | 适配 ppmat DataLoader |
| `train_diff.py` | 训练脚本 | 适配 ppmat trainer |
| `predict.py` | 推理脚本 | 适配 ppmat predictor |
| `eval.py` | 评估脚本（WSD, MAE） | 适配 ppmat metrics |

### 3.3 PyTorch → Paddle API 映射

DiffSyn 使用标准 PyTorch + PyG 操作，无自定义 CUDA kernel：

| PyTorch API | Paddle API | 备注 |
|------------|-----------|------|
| `nn.Module` | `paddle.nn.Layer` | 基类替换 |
| `nn.Linear` / `nn.ReLU` / `nn.Dropout` | 对应 Paddle API | 一致 |
| `torch_geometric.nn` | 手动迁移 GNN 层 | 消息传递 + scatter |
| `torch.scatter` | `ppmat.utils.scatter` | 复用已有 |
| `torch.optim.Adam` | `paddle.optimizer.Adam` | 一致 |

### 3.4 数据集概况

| 名称 | 内容 | 数据量 | 说明 |
|------|------|--------|------|
| ZeoSyn | 沸石合成路线（拓扑 + OSDA + 合成参数） | 23,961 条 | 233 种拓扑，921 种 OSDA |

数据随 repo 提供（`data/` 目录），MIT License。预处理后的图数据（`zeo2graph.pkl`, `smiles2graph.pkl`）可直接使用。

训练资源：~50h on RTX A5000（24GB）。

### 3.5 已有实现调研

| 平台 | 实现情况 |
|------|--------|
| PaddlePaddle / PaddleMaterials | ✖ 无 |
| MindSpore | ✖ 无 |
| PyTorch（原始） | ✔ eltonpan/zeosyn_gen（MIT License，36 stars） |

目前仅有原始 PyTorch 实现（使用标准 PyTorch + PyG GNN），无其他框架移植。

## 4. 设计思路与实现方案

### 4.1 代码目录结构

```
PaddleMaterials/
├── ppmat/models/diffsyn/
│   ├── __init__.py                  # 模块导出 + 注册
│   ├── diffsyn.py                   # 主模型（_forward/forward/predict 三层模式）
│   ├── zeolite_encoder.py           # 沸石 3D 拓扑图 GNN 编码器
│   ├── osda_encoder.py              # OSDA 2D 分子图 GNN 编码器
│   └── synthesis_diffusion.py       # 合成参数空间条件扩散
├── ppmat/datasets/zeosyn_dataset.py # ZeoSyn 数据集 + build_zeosyn 工厂函数
├── synthesis_condition_prediction/  # 合成条件预测任务
│   ├── README.md                    # 任务说明
│   └── configs/diffsyn/
│       ├── diffsyn_train.yaml       # 训练配置
│       └── diffsyn_sample.yaml      # 采样配置
├── examples/diffsyn/
│   ├── README.md
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py

# 需修改的已有文件（仅新增注册行）：
# ppmat/models/__init__.py     — 新增 diffsyn 导入
# ppmat/datasets/__init__.py   — 新增 build_zeosyn 工厂函数注册
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

class ZeoliteEncoder(nn.Layer):
    """沸石 3D 拓扑图 GNN 编码器"""
    def __init__(self, node_fdim, hidden_dim=128, n_layers=4):
        super().__init__()
        # 3D 图消息传递（参考 DimeNet++ 的距离嵌入逻辑）
        ...

    def forward(self, zeo_graph):
        """沸石图 → 向量表示"""
        ...

class OsdaEncoder(nn.Layer):
    """OSDA 2D 分子图 GNN 编码器"""
    def __init__(self, atom_fdim, hidden_dim=128, n_layers=3):
        super().__init__()
        ...

    def forward(self, osda_graph):
        """OSDA 分子图 → 向量表示"""
        ...

class DiffSyn(nn.Layer):
    """DiffSyn 条件扩散模型，遵循 ppmat 三层模式"""
    def __init__(self, zeo_encoder, osda_encoder, param_dim,
                 hidden_dim=256, n_steps=1000, cond_drop_prob=0.1):
        super().__init__()
        self.zeo_encoder = zeo_encoder
        self.osda_encoder = osda_encoder
        self.cond_drop_prob = cond_drop_prob
        self.n_steps = n_steps

        # MLP denoiser for synthesis parameters
        cond_dim = zeo_encoder.output_dim + osda_encoder.output_dim
        self.denoiser = nn.Sequential(
            nn.Linear(param_dim + cond_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim),
        )

    def _forward(self, noisy_params, t, zeo_emb, osda_emb):
        """纯前向：预测噪声"""
        cond = paddle.concat([zeo_emb, osda_emb], axis=-1)
        x = paddle.concat([noisy_params, cond, t.unsqueeze(-1)], axis=-1)
        return self.denoiser(x)

    def forward(self, batch):
        """训练入口：条件扩散 + classifier-free dropout"""
        zeo_emb = self.zeo_encoder(batch['zeo_graph'])
        osda_emb = self.osda_encoder(batch['osda_graph'])

        # Classifier-free guidance: 随机 drop 条件
        if self.training:
            mask = paddle.rand([zeo_emb.shape[0], 1]) > self.cond_drop_prob
            zeo_emb = zeo_emb * mask.astype('float32')
            osda_emb = osda_emb * mask.astype('float32')

        params = batch['synthesis_params']  # 真实合成参数
        # 采样时间步和噪声
        t = paddle.randint(0, self.n_steps, [params.shape[0]])
        noise = paddle.randn_like(params)
        # 添加噪声
        noisy_params = self._add_noise(params, noise, t)
        # 预测噪声
        pred_noise = self._forward(noisy_params, t.astype('float32'), zeo_emb, osda_emb)

        loss = nn.functional.mse_loss(pred_noise, noise)
        return {"diffusion_loss": loss}, {"pred_noise": pred_noise}

    @paddle.no_grad()
    def predict(self, batch, cond_scale=0.75):
        """推理入口：classifier-free guidance 采样"""
        zeo_emb = self.zeo_encoder(batch['zeo_graph'])
        osda_emb = self.osda_encoder(batch['osda_graph'])

        params = paddle.randn([zeo_emb.shape[0], self.param_dim])
        for t in reversed(range(self.n_steps)):
            t_tensor = paddle.full([params.shape[0]], t, dtype='float32')
            # 有条件预测
            pred_cond = self._forward(params, t_tensor, zeo_emb, osda_emb)
            # 无条件预测
            pred_uncond = self._forward(params, t_tensor,
                                       paddle.zeros_like(zeo_emb),
                                       paddle.zeros_like(osda_emb))
            # Classifier-free guidance
            pred = pred_uncond + cond_scale * (pred_cond - pred_uncond)
            params = self._denoise_step(params, pred, t)

        return {"synthesis_params": params}
```

### 4.3 数据集适配

```python
# ppmat/datasets/zeosyn_dataset.py
import pickle
import paddle
from paddle.io import Dataset

class ZeoSynDataset(Dataset):
    """ZeoSyn 沸石合成数据集"""
    def __init__(self, data_path, zeo_graph_path, osda_graph_path, split='train'):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(zeo_graph_path, 'rb') as f:
            self.zeo_graphs = pickle.load(f)
        with open(osda_graph_path, 'rb') as f:
            self.osda_graphs = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        return {
            'zeo_graph': self.zeo_graphs[entry['zeolite_code']],
            'osda_graph': self.osda_graphs[entry['osda_smiles']],
            'synthesis_params': paddle.to_tensor(entry['params'], dtype='float32'),
        }

def build_zeosyn(config):
    """ZeoSyn 数据集工厂函数"""
    return ZeoSynDataset(
        data_path=config.data_path,
        zeo_graph_path=config.zeo_graph_path,
        osda_graph_path=config.osda_graph_path,
        split=config.get('split', 'train')
    )
```

### 4.4 训练流程

1. **损失函数**：去噪 MSE（预测噪声 vs 实际噪声），含 classifier-free guidance dropout
2. **优化器**：Adam (lr=1e-4)，1000 steps warmup 后 linear decay
3. **采样**：DDPM 1000 steps + classifier-free guidance (w=2.0)

### 4.5 YAML 配置示例

```yaml
Global:
  task: synthesis_condition_prediction
  seed: 42
  output_dir: output/diffsyn_zeolite/

Model:
  __class_name__: DiffSyn
  __init_params__:
    zeolite_encoder:
      __class_name__: SchNet3D
      __init_params__:
        hidden_dim: 256
    osda_encoder:
      __class_name__: GIN
      __init_params__:
        hidden_dim: 256
    diffusion:
      n_steps: 1000
      beta_schedule: linear
      guidance_weight: 2.0
      cond_drop_prob: 0.1

Dataset:
  __class_name__: zeosyn
  __init_params__:
    data_path: data/diffsyn/zeosyn_dataset.pkl
    zeo_graph_path: data/diffsyn/zeo_graphs.pkl
    osda_graph_path: data/diffsyn/osda_graphs.pkl

Optimizer:
  __class_name__: Adam
  __init_params__:
    lr: 1.0e-4
    warmup_steps: 1000
```

### 4.6 补充说明

- 不复现原论文中的 OSDA 费用化学可行性过滤器（依赖商业数据库）。
- GNN 编码器使用 ppmat 已有 SchNet/DimeNet++ 组件，不迁移原始代码中的自定义 GNN。

## 5. 测试和验收的考量

### 5.1 前向精度对齐

- 使用相同输入（沸石图 + OSDA 图 + 噪声参数），对比 PyTorch 和 Paddle 的去噪输出
- 要求：logits diff ≤ 1e-4

### 5.2 反向训练对齐

- 使用相同数据子集，对比两个框架训练 2 轮以上的 loss 曲线
- 要求 loss 逐 epoch 一致

### 5.3 生成质量指标

| 指标 | 说明 | 原论文值 | 允许误差 |
|------|------|---------|---------|
| Mean WSD | Wasserstein 距离（合成参数分布匹配度） | 0.423 | ±5% |
| MAE | 合成参数平均绝对误差 | 论文 Table | ±5% |
| Precision/Recall | 生成参数在合理范围内的比例 | 论文 Fig. 2 | ±5% |

### 5.4 Demo 复现

- 复现论文 Fig. 5 demo：为 UFI 沸石 + 特定 OSDA 生成 1000 组合成配方
- 1000 条生成预计 ~2 分钟

## 6. 可行性分析和排期规划

### 6.1 可行性分析

| 维度 | 评估 | 说明 |
|-----|------|------|
| 架构复杂度 | **中** | 条件扩散 + 两个 GNN 编码器，标准组件 |
| 自定义算子 | **无** | 全部为标准 NN + GNN 操作 |
| 依赖项风险 | **中** | PyG（GNN 层需迁移）、RDKit |
| 数据可获取性 | **高** | ZeoSyn 数据随 repo 提供，预构建图可直接使用 |
| API 映射完整性 | **中** | PyG GNN 层需手动迁移；其余标准 API 一致 |
| 计算资源 | **高** | 训练 ~50h on RTX A5000 |

### 6.2 排期规划

| 阶段 | 内容 | 预计工期 |
|-----|------|---------|
| Phase 0：环境搭建 | GPU 环境、ZeoSyn 数据下载验证 | 2 天 |
| Phase 1：编码器迁移 | 沸石 GNN + OSDA GNN → Paddle | 4 天 |
| Phase 2：扩散模型 | 条件扩散 + classifier-free guidance | 3 天 |
| Phase 3：训练对齐 | Trainer 适配、loss 曲线对齐 | 3 天 |
| Phase 4：评估验证 | WSD/MAE 评估、demo 复现 | 4 天 |
| Phase 5：文档与合入 | README 编写、提交 PR | 2 天 |

**合计**：~18 天

## 7. 影响面

1. **新增模型**：在 `ppmat/models/` 下新增 `diffsyn/` 目录（~600 行代码），包含沸石/OSDA 图编码器和合成参数空间条件扩散模型，支持 classifier-free guidance 采样。
2. **新增数据集**：在 `ppmat/datasets/` 下新增 `zeosyn_dataset.py`（~100 行），支持 ZeoSyn 数据集（23,961 条沸石合成路线）的预构建图数据加载。
3. **新增任务类型**：新建 `synthesis_condition_prediction/` 目录，引入“合成条件预测”任务类型，为 PaddleMaterials 建立逆向合成范式。
4. **对已有代码无侵入性修改**，仅需在 `ppmat/models/__init__.py` 和 `ppmat/datasets/__init__.py` 中注册新模块。

## 名词解释

| 名词 | 说明 |
|------|------|
| 沸石 (Zeolite) | 由 SiO₄/AlO₄ 四面体组成的多孔晶体材料，广泛用于催化和分离 |
| OSDA | Organic Structure-Directing Agent，有机结构导向剂，引导沸石特定拓扑结构的形成 |
| ZeoSyn | 沸石合成数据库，包含 23,961 条合成路线 |
| Classifier-free Guidance | 无分类器引导采样，通过混合有条件/无条件预测控制生成质量 |
| WSD | Wasserstein Distance，Wasserstein 距离，衡量两个分布的差异 |
| H₂O/T ratio | 水与 T-site（四面体位点）的摩尔比，关键合成参数 |

## 附件及参考资料

1. Pan, Elton et al. "DiffSyn: A generative diffusion approach to materials synthesis planning." *Nature Computational Science* (2026). DOI: 10.1038/s43588-025-00949-9
2. Pan, Elton et al. "Towards generative design of inorganic synthesis." *NeurIPS AI for Materials Workshop* (2024, Oral Spotlight). https://openreview.net/forum?id=hy39qxU6CQ
3. DiffSyn 源码：https://github.com/eltonpan/zeosyn_gen（MIT License）
4. ZeoSyn 数据集：https://pubs.acs.org/doi/10.1021/acscentsci.3c01615
5. PaddleMaterials 模型复现列表：https://github.com/PaddlePaddle/PaddleMaterials/issues/194（第 11 项）

# RFdiffusion: Protein Structure Generation with Diffusion Models

## Overview
RFdiffusion is an open-source toolkit for de novo protein structure generation, leveraging diffusion models to support a wide range of protein design tasks. Built on the architecture and parameters of RoseTTAFold, it enables conditional and unconditional generation of protein structures with high flexibility and accuracy. This repository also includes integration with SE(3)-Transformers, providing 3D rotation-translation equivariant attention networks to enhance structural modeling.


## Core Features
RFdiffusion supports the following key protein design capabilities:
- **Motif Scaffolding**: Embed specific structural motifs into generated proteins.
- **Unconditional Generation**: Generate novel protein structures without prior constraints.
- **Symmetric Oligomer Generation**: Create proteins with cyclic, dihedral, or tetrahedral symmetry.
- **Binder Design**: Design de novo binders for target proteins, including support for hotspot residues.
- **Design Diversification**: Refine existing designs through partial diffusion sampling.
- **Fold Conditioning**: Use scaffold structures to guide generation (e.g., PPI scaffold examples).


## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.1+ (for GPU acceleration)
- Conda or Docker (for environment management)


### Local Installation with Conda
1. **Clone the repository**
   ```bash
   git clone https://github.com/RosettaCommons/RFdiffusion.git
   cd RFdiffusion
   ```

2. **Install SE3-Transformer**
   ```bash
   conda create -n rfdiffusion python=3.8
   conda activate rfdiffusion
   # Install SE3-Transformer (required for 3D equivariant layers)
   cd env/SE3Transformer
   pip install --editable .
   cd ../..
   ```

3. **Download model weights**
   ```bash
   mkdir models && cd models
   # Base models
   wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
   wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
   # Additional models (optional: for fold conditioning, active sites, etc.)
   wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
   wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
   cd ..
   ```

4. **Install RFdiffusion**
   ```bash
   pip install --editable .
   ```


### Docker Installation
1. **Build the Docker image**
   ```bash
   docker build -f docker/Dockerfile -t rfdiffusion .
   ```

2. **Prepare directories for data and models**
   ```bash
   mkdir -p $HOME/inputs $HOME/outputs $HOME/models
   # Download model weights
   bash scripts/download_models.sh $HOME/models
   ```


## Quick Start

### Basic Execution: Unconditional Monomer Generation
Generate 10 random protein monomers (length 100-150 residues):
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[100-150]' \
  inference.output_prefix=outputs/unconditional_monomer \
  inference.num_designs=10
```


### Motif Scaffolding
Embed a motif (from a PDB file) into a generated protein:
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[50-100/A1-20/50-100]' \
  inference.input_pdb=inputs/motif_template.pdb \
  inference.output_prefix=outputs/motif_scaffold \
  inference.num_designs=5
```
- `A1-20`: Specifies residues 1-20 of chain A in `motif_template.pdb` as the motif.


### Binder Design
Design binders targeting residues 1-100 of chain B in a target PDB:
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[B1-100/0 50-100]' \
  inference.input_pdb=inputs/target.pdb \
  inference.output_prefix=outputs/binder_design \
  inference.num_designs=10 \
  # Use beta model for diverse topologies (optional)
  inference.ckpt_override_path=models/Complex_beta_ckpt.pt
```


### Symmetric Oligomer Generation
Generate a C3-symmetric trimer:
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[100-150]' \
  symmetry.symmetry='C3' \
  inference.output_prefix=outputs/c3_trimer \
  inference.num_designs=5
```


## Core Modules & Functions

### 1. `rfdiffusion.util_module`
Utility functions for model initialization, graph construction, and coordinate calculations.

#### Key Functions:
- `init_lecun_normal(module)`: Initializes module weights using LeCun normal distribution.
  ```python
  import torch.nn as nn
  from rfdiffusion.util_module import init_lecun_normal

  linear_layer = nn.Linear(128, 256)
  init_lecun_normal(linear_layer)  # Initialize weights
  ```

- `rbf(D)`: Computes radial basis functions for distance features.
  ```python
  import torch
  from rfdiffusion.util_module import rbf

  distances = torch.rand(10, 10)  # Example distance matrix (B=10, L=10)
  rbf_features = rbf(distances)   # Shape: (10, 10, 36) (36 basis functions)
  ```

- `make_topk_graph(xyz, pair, idx)`: Constructs a graph using top-K nearest neighbors.
  ```python
  from rfdiffusion.util_module import make_topk_graph

  # xyz: (B, L, 3, 3) backbone coordinates, pair: (B, L, L, E) pair features, idx: (B, L) residue indices
  graph, edge_features = make_topk_graph(xyz, pair, idx, top_k=64)
  ```


### 2. SE(3)-Transformers
3D rotation-translation equivariant layers for processing structural data. Key files:
- `se3_transformer/model/transformer.py`: Main SE(3)-Transformer module.
- `se3_transformer/runtime/training.py`: Training script for SE(3)-based models.

Example training command for SE(3)-Transformer:
```bash
cd env/SE3Transformer
bash scripts/train.sh  # Trains on QM9 dataset by default
```


## Output Files
Generated structures are saved in PDB format in the output directory. Additional files include:
- `scores.csv`: Design metrics (e.g., predicted affinity, symmetry score).
- `config.yaml`: Full configuration used for the run.


## Practical Tips
- **Filtering Binders**: Use AlphaFold2 to filter designs with `pae_interaction < 10` for higher experimental success rates (scripts available [here](https://github.com/nrbennet/dl_binder_design)).
- **Symmetric Design**: Ensure contigs match the specified symmetry (e.g., `C3` requires 3 identical subunits).
- **Performance**: Reduce target protein length for faster runtime (scales with O(N²)).


## License
RFdiffusion is released under the BSD License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements
- Built on RoseTTAFold (Frank DiMaio and Minkyung Baek).
- SE(3)-Transformer implementation adapted from [NVIDIA's DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples).
- Developed by Joe, David, Nate, Brian, Jason, and the RFdiffusion team.

For questions, create a GitHub issue or refer to the [paper](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1).

# RFdiffusion：基于扩散模型的蛋白质结构生成

## 概述
RFdiffusion是一个开源的从头蛋白质结构生成工具包，它利用扩散模型支持多种蛋白质设计任务。该工具基于RoseTTAFold的架构和参数构建，能够灵活、准确地进行有条件和无条件的蛋白质结构生成。本仓库还集成了SE(3)-Transformers，提供3D旋转-平移等变注意力网络，以增强结构建模能力。


## 核心功能
RFdiffusion支持以下关键蛋白质设计能力：
- **基序支架设计（Motif Scaffolding）**：将特定结构基序嵌入到生成的蛋白质中。
- **无条件生成（Unconditional Generation）**：在无先验约束的情况下生成全新的蛋白质结构。
- **对称寡聚体生成（Symmetric Oligomer Generation）**：创建具有环状、二面体或四面体对称性的蛋白质。
- **结合剂设计（Binder Design）**：为目标蛋白质设计从头结合剂，包括支持热点残基。
- **设计多样化（Design Diversification）**：通过部分扩散采样优化现有设计。
- **折叠条件控制（Fold Conditioning）**：使用支架结构指导生成（如蛋白质-蛋白质相互作用支架示例）。


## 安装

### 前提条件
- Python 3.8+
- CUDA 11.1+（用于GPU加速）
- Conda或Docker（用于环境管理）


### 基于Conda的本地安装
1. **克隆仓库**
   ```bash
   git clone https://github.com/RosettaCommons/RFdiffusion.git
   cd RFdiffusion
   ```

2. **安装SE3-Transformer**
   ```bash
   conda create -n rfdiffusion python=3.8
   conda activate rfdiffusion
   # 安装SE3-Transformer（3D等变层所需）
   cd env/SE3Transformer
   pip install --editable .
   cd ../..
   ```

3. **下载模型权重**
   ```bash
   mkdir models && cd models
   # 基础模型
   wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
   wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
   # 额外模型（可选：用于折叠条件控制、活性位点等）
   wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
   wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
   cd ..
   ```

4. **安装RFdiffusion**
   ```bash
   pip install --editable .
   ```


### Docker安装
1. **构建Docker镜像**
   ```bash
   docker build -f docker/Dockerfile -t rfdiffusion .
   ```

2. **准备数据和模型目录**
   ```bash
   mkdir -p $HOME/inputs $HOME/outputs $HOME/models
   # 下载模型权重
   bash scripts/download_models.sh $HOME/models
   ```


## 快速开始

### 基本执行：无条件单体生成
生成10个随机蛋白质单体（长度100-150个残基）：
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[100-150]' \
  inference.output_prefix=outputs/unconditional_monomer \
  inference.num_designs=10
```


### 基序支架设计
将基序（来自PDB文件）嵌入到生成的蛋白质中：
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[50-100/A1-20/50-100]' \
  inference.input_pdb=inputs/motif_template.pdb \
  inference.output_prefix=outputs/motif_scaffold \
  inference.num_designs=5
```
- `A1-20`：指定`motif_template.pdb`中A链的1-20号残基作为基序。


### 结合剂设计
设计靶向目标PDB中B链1-100号残基的结合剂：
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[B1-100/0 50-100]' \
  inference.input_pdb=inputs/target.pdb \
  inference.output_prefix=outputs/binder_design \
  inference.num_designs=10 \
  # 可选：使用beta模型获得更多样的拓扑结构
  inference.ckpt_override_path=models/Complex_beta_ckpt.pt
```


### 对称寡聚体生成
生成C3对称三聚体：
```bash
./scripts/run_inference.py \
  'contigmap.contigs=[100-150]' \
  symmetry.symmetry='C3' \
  inference.output_prefix=outputs/c3_trimer \
  inference.num_designs=5
```


## 核心模块与函数

### 1. `rfdiffusion.util_module`
用于模型初始化、图构建和坐标计算的工具函数。

#### 关键函数：
- `init_lecun_normal(module)`：使用LeCun正态分布初始化模块权重。
  ```python
  import torch.nn as nn
  from rfdiffusion.util_module import init_lecun_normal

  linear_layer = nn.Linear(128, 256)
  init_lecun_normal(linear_layer)  # 初始化权重
  ```

- `rbf(D)`：计算距离特征的径向基函数。
  ```python
  import torch
  from rfdiffusion.util_module import rbf

  distances = torch.rand(10, 10)  # 示例距离矩阵（B=10, L=10）
  rbf_features = rbf(distances)   # 形状：(10, 10, 36)（36个基函数）
  ```

- `make_topk_graph(xyz, pair, idx)`：使用Top-K最近邻构建图。
  ```python
  from rfdiffusion.util_module import make_topk_graph

  # xyz: (B, L, 3, 3) 主链坐标，pair: (B, L, L, E) 成对特征，idx: (B, L) 残基索引
  graph, edge_features = make_topk_graph(xyz, pair, idx, top_k=64)
  ```


### 2. SE(3)-Transformers
用于处理结构数据的3D旋转-平移等变层。关键文件：
- `se3_transformer/model/transformer.py`：SE(3)-Transformer主模块。
- `se3_transformer/runtime/training.py`：基于SE(3)的模型训练脚本。

SE(3)-Transformer训练示例命令：
```bash
cd env/SE3Transformer
bash scripts/train.sh  # 默认在QM9数据集上训练
```


## 输出文件
生成的结构以PDB格式保存在输出目录中。其他文件包括：
- `scores.csv`：设计指标（如预测亲和力、对称分数）。
- `config.yaml`：运行时使用的完整配置。


## 实用技巧
- **结合剂筛选**：使用AlphaFold2筛选`pae_interaction < 10`的设计，以提高实验成功率（相关脚本见[此处](https://github.com/nrbennet/dl_binder_design)）。
- **对称设计**：确保序列片段（contigs）与指定对称性匹配（例如，`C3`需要3个相同的亚基）。
- **性能优化**：减少目标蛋白质长度可加快运行速度（复杂度为O(N²)）。


## 许可证
RFdiffusion基于BSD许可证发布。详见[LICENSE](LICENSE)文件。


## 致谢
- 基于RoseTTAFold的架构和训练参数构建（Frank DiMaio和Minkyung Baek开发）。
- SE(3)-Transformer实现改编自[NVIDIA的DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples)。
- 由Joe、David、Nate、Brian、Jason和RFdiffusion团队开发。

如有问题，请创建GitHub issue或参考[论文](https://www.biorxiv.org/content/10.1101/2022.12.09.519842v1)。
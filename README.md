# 材料性质预测项目简介

> 一个「自然语言提问 → AI 助手调用材料模型 → 返回性质预测」的终端版小系统。

---

## 一、项目目前的功能

### 1.1 整体形态：对话式材料查询助手

关键设计：大模型**不允许凭记忆回答**，必须走模型计算，保证结果可溯源、可对照。

```
用户提问
   │
   ▼
DeepSeek（判断意图、决定调用工具）
   │  化学式
   ▼
JARVIS-DFT 数据库查结构 ──► 结构转图 ──► GNN 预测
   │                                      │
   │                                      ▼
   │                          形成能 / 带隙 / 稳定性
   ▼                                      │
大模型结合「预测值 + DFT 真值 + 是否见过」组织回答
```

### 1.2 每次预测会返回什么

- **prediction**：GNN 模型的预测值（近似 DFT 值，不是实验值）；
- **dft_reference**：JARVIS 数据库里的 DFT 真值，方便当场对照；
- **split**：该化学式属于 `train`（训练时见过）还是 `test`（没见过），提醒「记忆」和「泛化」的区别；
- 一个化学式可能对应多个晶体结构（多晶型），会逐条列出（最多 20 条）。

### 1.3 后续开发

目前输入化学式只能通过公开数据库JARVIS-DFT转成晶体结构，可以考虑后续把数据库换成神经网络增加泛化性；
增加用户界面（JavaWeb||desktop）;


## 二、重点：材料分析模型

### 2.1 网络结构：自实现的 CGCNN 风格 `CrystalGNN`

网络写在 `model.py`，**只用 PyTorch，不依赖 PyG / DGL**（减少云端装环境的坑）。

一条数据流的四步：**编码 → 消息传递 → 池化 → 多任务输出头**。

| 步骤 | 实现 | 说明 |
|---|---|---|
| **编码** | `nn.Embedding(119, 64)` + `RBFExpansion` | 原子按原子序数变成 64 维向量；距离用 40 个高斯基函数展开成 40 维边特征 |
| **消息传递** | `CrystalConv` × 3 层 | 每条边把「中心原子 + 邻居原子 + 距离」拼接，过线性层 + BatchNorm，用 sigmoid 门控 × softplus 生成消息，求和聚合到中心原子（残差更新） |
| **池化** | `segment_mean` | 把一个晶体内所有原子向量求平均，得到整条晶体的向量 |
| **输出头** | 3 个独立小 MLP | 每个性质一个头：64 → 128 → 1，共享主干、各自输出，得到 3 个性质 |

- 超参数：原子向量 64 维、3 层卷积、隐层 128、dropout 0.1；
- **参数量约 98,627**，是一个很小的模型；
- 输出 3 个任务：**形成能、带隙、稳定性（能量凸包）**，全部按回归处理。

### 2.2 训练数据

- **数据来源**：**JARVIS-DFT**（美国 NIST 的公开 DFT 数据库），数据集 `dft_3d`；
- **规模**：**93,902 条**材料，结构 + 三种性质齐全；
- **字段映射**：

| JARVIS 字段 | 预测目标 | 单位 |
|---|---|---|
| `formation_energy_peratom` | 形成能 | eV/atom |
| `optb88vdw_bandgap` | 带隙 | eV |
| `ehull` | 稳定性（能量凸包） | eV/atom |

- **切分方式**：按**化学式分组**切 train / val / test = 80% / 10% / 10%。
  同一化学式的多条结构不跨集合，**防止数据泄漏**（否则等于让模型背答案）。
- 实际切分：train 75,003 / val 9,698 / test 9,201。

---

## 三、项目结构

```
/
├── .gitignore               # Git 忽略规则（.env、缓存等不进仓库）
├── resources/               # 代码与模型（运行入口都在这里）
│   ├── agent.py             # 终端对话助手主程序
│   ├── predict.py           # GNN 推理 + agent 工具后端
│   ├── materials_db.py      # JARVIS 数据库访问层
│   ├── dataset.py           # 晶体结构 → 图（建图 + 批处理）
│   ├── model.py             # GNN 网络结构定义
│   ├── model.pt             # 训练好的模型权重（约 400 KB）
│   ├── requirements.txt     # 依赖清单
│   ├── .env                 # DeepSeek API Key 配置（当前为空）
│   └── readme.md            # 本文档
```

---

## 附：快速运行

```bash
# 1) 准备环境（只需一次）
conda create -n gnn python=3.10 -y
conda activate gnn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch numpy jarvis-tools openai python-dotenv

# 2) 配置 DeepSeek Key（当前未配置）
#    在 resources/ 目录的 .env 里写入：DEEPSEEK_API_KEY=你的key

# 3) 进入代码目录并运行
cd XXXpath/resources #（你的路径信息）
python agent.py

# 或者不走 agent，直接命令行预测
python predict.py --formula GaN Si Al2O3
python predict.py my_structure.cif

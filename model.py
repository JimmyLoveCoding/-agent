"""
GNN 网络结构（自包含 CGCNN 风格）
=========================================================
把"晶体结构图"映射到性质。本文件只定义网络，不碰数据、不碰训练。
只依赖 torch —— 不需要 PyG / DGL 等重型图库（减少云上装环境的坑）。

数据流（由 dataset.py 负责把结构转成这些张量）：
    原子种类 z (N,)         —— 每个原子的原子序数
    edge_index (2, E)       —— 每条边的 (邻居, 中心)
    edge_dist  (E,)         —— 每条边的原子间距离
    batch      (N,)         —— 每个原子属于哪条晶体
    n_crystals  标量         —— 本批有多少条晶体

网络四步：
    编码 → 消息传递(×n_conv) → 池化 → 多任务输出头
"""
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# 1) 距离展开：把一个"距离"变成一组高斯基向量，作为边的特征
# ----------------------------------------------------------------------
class RBFExpansion(nn.Module):
    def __init__(self, vmin=0.0, vmax=8.0, bins=40):
        super().__init__()
        self.register_buffer("centers", torch.linspace(vmin, vmax, bins))
        self.gamma = 1.0 / ((vmax - vmin) / bins) ** 2

    def forward(self, dist):
        # dist: (E,)  ->  (E, bins)
        d = dist[:, None] - self.centers[None, :]
        return torch.exp(-self.gamma * d ** 2)


# ----------------------------------------------------------------------
# 2) 一层晶体卷积（CGCNN 的核心）：每个原子聚合邻居信息后更新自己
# ----------------------------------------------------------------------
class CrystalConv(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.fc = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.bn = nn.BatchNorm1d(2 * node_dim)

    def forward(self, x, edge_index, edge_fea):
        src, dst = edge_index[0], edge_index[1]           # 邻居 -> 中心
        z = torch.cat([x[src], x[dst], edge_fea], dim=1)  # (E, 2*node+edge)
        z = self.bn(self.fc(z))
        gate, msg = z.chunk(2, dim=1)
        msg = torch.sigmoid(gate) * torch.nn.functional.softplus(msg)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)  # 把消息加到中心原子上
        return x + agg                                     # 残差式更新


# ----------------------------------------------------------------------
# 3) 完整网络：编码 → 消息传递 → 池化 → 多个输出头
# ----------------------------------------------------------------------
def segment_mean(x, batch, n_crystals):
    """把一个晶体里所有原子的向量求平均，得到该晶体的一个向量。"""
    out = x.new_zeros((n_crystals, x.size(1)))
    out.index_add_(0, batch, x)
    cnt = x.new_zeros(n_crystals).index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
    return out / cnt.clamp(min=1)[:, None]


class CrystalGNN(nn.Module):
    def __init__(self, n_tasks, atom_dim=64, edge_bins=40,
                 n_conv=3, hidden=128, dropout=0.1):
        super().__init__()
        self.n_tasks = n_tasks
        self.atom_embed = nn.Embedding(119, atom_dim)   # 原子序数 0..118 -> 向量
        self.rbf = RBFExpansion(bins=edge_bins)
        self.convs = nn.ModuleList(
            [CrystalConv(atom_dim, edge_bins) for _ in range(n_conv)]
        )
        # 每个性质一个小 MLP 头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(atom_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, z, edge_index, edge_dist, batch, n_crystals):
        h = self.atom_embed(z)                 # 编码：原子 -> 向量
        e = self.rbf(edge_dist)                # 编码：距离 -> 向量
        for conv in self.convs:                # 消息传递
            h = conv(h, edge_index, e)
        pooled = segment_mean(h, batch, n_crystals)   # 池化：晶体 -> 一个向量
        out = torch.cat([head(pooled) for head in self.heads], dim=1)  # (B, n_tasks)
        return out


if __name__ == "__main__":
    # 自测：造一个假的小 batch，确认前向能跑通、形状对
    torch.manual_seed(0)
    model = CrystalGNN(n_tasks=3)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    # 2 条晶体：第一条 3 个原子，第二条 2 个原子
    z = torch.tensor([26, 8, 8, 11, 17])                       # Fe,O,O, Na,Cl
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4],
                               [1, 0, 2, 1, 0, 4, 3]])         # 双向
    edge_dist = torch.rand(edge_index.size(1)) + 1.0
    batch = torch.tensor([0, 0, 0, 1, 1])
    out = model(z, edge_index, edge_dist, batch, n_crystals=2)
    print("输出形状:", tuple(out.shape), "（应为 (2, 3) = 2 条晶体 × 3 个性质）")

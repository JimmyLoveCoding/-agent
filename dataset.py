"""
dataset.py —— 把「晶体结构」转成「图」
=========================================================
模型吃的是图（节点=原子、边=近邻、边特征=距离），本文件负责建图：
    graph_from_atoms     —— JARVIS 的 atoms 字典 -> 图
    graph_from_structure —— pymatgen 的 Structure  -> 图（CIF 预测用）
    collate              —— 一批图 -> 打包成张量（送进模型）
    num                  —— 数值安全转换（缺失/NaN -> None）

训练和推理都走 graph_from_atoms 这同一条路径，保证不会出现两套逻辑。
"""
import itertools

import numpy as np

CUTOFF = 5.0     # 近邻判断的截断距离（Å）

# 元素符号（1..118），用来把符号转成原子序数，免去 pymatgen 依赖
ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
            "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
            "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
            "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
            "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
            "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
            "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
SYM2Z = {s: i + 1 for i, s in enumerate(ELEMENTS)}


def num(x):
    """转 float，失败或 NaN 返回 None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def graph_from_atoms(atoms, cutoff=CUTOFF):
    """JARVIS 的 atoms 字典 -> 图（节点=原子序数，边=近邻，含周期性）。"""
    L = np.asarray(atoms["lattice_mat"], float)              # 晶格 (3,3)
    C = np.asarray(atoms["coords"], float)                   # 坐标 (N,3)
    if not atoms.get("cartesian", True):                     # 分数坐标则换算成笛卡尔
        C = C @ L
    Z = np.array([SYM2Z[s] for s in atoms["elements"]], dtype=np.int64)

    # 把 27 个周期性镜像里的邻居都找出来，距离 < cutoff 的连边
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)), float) @ L
    src, dst, dist = [], [], []
    for sh in shifts:
        d = np.linalg.norm(C[:, None, :] - (C + sh)[None, :, :], axis=-1)
        i, j = np.where((d > 1e-4) & (d < cutoff))
        src.append(j); dst.append(i); dist.append(d[i, j])
    return {"z": Z,
            "edge_index": np.array([np.concatenate(src), np.concatenate(dst)], dtype=np.int64),
            "edge_dist": np.concatenate(dist).astype(np.float32)}


def graph_from_structure(struct, cutoff=CUTOFF):
    """pymatgen Structure -> 图（转成 atoms 字典后复用 graph_from_atoms）。"""
    atoms = {
        "lattice_mat": np.asarray(struct.lattice.matrix, float),
        "coords": np.asarray(struct.cart_coords, float),
        "elements": [site.specie.symbol for site in struct],
        "cartesian": True,
    }
    return graph_from_atoms(atoms, cutoff)


def collate(graphs):
    """一批图 -> 大张量（所有图的节点/边拼在一起，用 batch 标明归属）。"""
    import torch
    z, ei, ed, batch, y = [], [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        n = len(g["z"])
        z.append(torch.as_tensor(g["z"]))
        ei.append(torch.as_tensor(g["edge_index"]) + offset)
        ed.append(torch.as_tensor(g["edge_dist"]))
        batch.append(torch.full((n,), gi, dtype=torch.long))
        y.append(torch.as_tensor(g["y"]))
        offset += n
    return (torch.cat(z), torch.cat(ei, dim=1), torch.cat(ed),
            torch.cat(batch), len(graphs), torch.stack(y))

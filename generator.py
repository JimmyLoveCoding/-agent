"""
generator.py —— 化学式 -> 候选晶体结构（Chemeleon-DNG CSP）
=========================================================
用 Chemeleon 的 CSP 预训练权重，给定化学式采样若干候选结构（未弛豫），
再挑一个“最像样”的：几何合法、且被重复生成最多的结构。
模型只在进程内加载一次；首次运行会自动把权重下载到 ./ckpts。
"""
import numpy as np
import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
from chemeleon_dng.download_util import get_checkpoint_path
from chemeleon_dng.sample import DEFAULT_MODEL_PATH

N_SAMPLES = 10      # 每个化学式生成几个候选结构

_dm = None


def _load():
    """加载 CSP 扩散模型（进程内缓存）。"""
    global _dm
    if _dm is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        path = get_checkpoint_path("csp", DEFAULT_MODEL_PATH)
        _dm = DiffusionModule.load_from_checkpoint(
            path, map_location=device, weights_only=False
        )
        _dm.eval()
    return _dm


def generate_structures(formula: str, n_samples: int = N_SAMPLES) -> list:
    """化学式 -> n_samples 个 pymatgen Structure（未弛豫）。"""
    comp = Composition(formula)
    atom_types = [el.Z for el, amt in comp.items() for _ in range(int(amt))]
    atoms_list = _load().sample(
        task="csp",
        atom_types=atom_types * n_samples,
        num_atoms=[len(atom_types)] * n_samples,
        verbose=False,
    )
    return [AseAtomsAdaptor.get_structure(atoms) for atoms in atoms_list]


def _valid(struct) -> bool:
    """几何合法性：所有原子间距 > 0.6 Å，体积/原子在 3–80 Å³。"""
    dist = struct.distance_matrix + np.eye(len(struct)) * 99
    return dist.min() > 0.6 and 3.0 < struct.volume / len(struct) < 80.0


def pick_best(structures: list):
    """挑代表结构：优先合法结构，再取被重复生成最多的那一类。"""
    candidates = [s for s in structures if _valid(s)] or structures
    clusters = []
    matcher = StructureMatcher()
    for s in candidates:
        for cluster in clusters:
            if matcher.fit(cluster[0], s):
                cluster.append(s)
                break
        else:
            clusters.append([s])
    return max(clusters, key=len)[0]

"""
predict.py —— 用训练好的 GNN 预测晶体性质（兼 agent 的工具后端）
=========================================================
两种用法：
  1) 化学式预测：给化学式 -> Chemeleon 生成候选结构 -> GNN 预测
         python predict.py --formula GaN Si
  2) CIF 预测：给自己的 .cif 结构文件
         python predict.py a.cif b.cif

同时作为 agent 的工具后端，对外暴露 TOOL_SCHEMA / TOOL_FUNCS。
依赖：torch、numpy、pymatgen、chemeleon-dng。
"""
import os

import numpy as np
import torch

from generator import generate_structures, pick_best
from model import CrystalGNN
from dataset import graph_from_structure, collate

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOTE = ("prediction = 从 10 个扩散采样候选中挑出的代表结构（几何合法、重复最多）"
        "的 GNN 预测；结构未弛豫。"
        "单位：形成能/凸包 eV/atom，带隙 eV。")

_model = _ck = None


# ----------------------------------------------------------------------
# 模型加载与前向
# ----------------------------------------------------------------------
def load_model():
    ck = torch.load(os.path.join(HERE, "model.pt"), map_location=DEVICE, weights_only=False)
    model = CrystalGNN(n_tasks=ck["n_tasks"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def get_model():
    """进程内缓存的模型（agent 常驻进程只加载一次）。"""
    global _model, _ck
    if _model is None:
        _model, _ck = load_model()
    return _model, _ck


def predict_graphs(graphs, model, ck):
    """一批图 -> (B, 3) 预测值（已反标准化回原始单位）。"""
    for g in graphs:
        g["y"] = np.zeros(len(ck["targets"]), dtype=np.float32)   # collate 需要占位
    z, ei, ed, bt, n, _ = collate(graphs)
    with torch.no_grad():
        out = model(z.to(DEVICE), ei.to(DEVICE), ed.to(DEVICE), bt.to(DEVICE), n)
    return out.cpu().numpy() * ck["y_std"].numpy() + ck["y_mean"].numpy()


# ----------------------------------------------------------------------
# 方法 1：化学式预测（也是 agent 的工具）
# ----------------------------------------------------------------------
def predict_material_property(formula: str) -> dict:
    """化学式 -> 生成候选结构 -> 挑代表结构 -> 三性质预测。"""
    structs = generate_structures(formula)
    best = pick_best(structs)
    model, ck = get_model()
    pred = predict_graphs([graph_from_structure(best)], model, ck)[0]
    return {
        "formula": formula,
        "n_structures": len(structs),
        "prediction": {t: round(float(v), 4) for t, v in zip(ck["targets"], pred)},
        "note": NOTE,
    }


# ----------------------------------------------------------------------
# 方法 2：CIF 预测
# ----------------------------------------------------------------------
def predict_cif_files(paths) -> list:
    """一批 .cif 文件 -> [{file, prediction}]。"""
    from pymatgen.core import Structure
    model, ck = get_model()
    graphs = [graph_from_structure(Structure.from_file(p)) for p in paths]
    pred = predict_graphs(graphs, model, ck)
    return [{"file": os.path.basename(p),
             "prediction": {t: round(float(v), 4) for t, v in zip(ck["targets"], row)}}
            for p, row in zip(paths, pred)]


# ----------------------------------------------------------------------
# agent 工具声明（给 LLM 看）
# ----------------------------------------------------------------------
TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "predict_material_property",
        "description": (
            "根据化学式生成若干候选晶体结构，挑出几何合法且重复最多的代表结构，"
            "再用 GNN 预测三种性质：形成能、带隙、稳定性（能量凸包）。"
            "输入化学式（如 GaN、Fe2O3），返回候选数和三项预测值。"
            "生成的结构未做弛豫，预测是近似值。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "formula": {"type": "string", "description": "化学式，如 GaN、Fe2O3"},
            },
            "required": ["formula"],
        },
    },
}]

TOOL_FUNCS = {"predict_material_property": predict_material_property}


# ----------------------------------------------------------------------
# 命令行
# ----------------------------------------------------------------------
def _show_formula(formulas):
    print(f"已加载 model.pt，设备：{DEVICE}")
    for w in formulas:
        res = predict_material_property(w)
        print(f"\n{w}（{res['n_structures']} 个候选结构，未弛豫）:")
        for t, v in res["prediction"].items():
            print(f"  {t:26s} = {v:7.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="用训练好的 GNN 预测晶体性质")
    ap.add_argument("cif", nargs="*", help="一个或多个 .cif 文件（CIF 预测）")
    ap.add_argument("--formula", nargs="+", metavar="化学式", help="化学式（化学式预测）")
    args = ap.parse_args()

    if args.formula:
        _show_formula(args.formula)
    elif args.cif:
        print(f"已加载 model.pt，设备：{DEVICE}")
        for it in predict_cif_files(args.cif):
            print(f"\n{it['file']}")
            for name, v in it["prediction"].items():
                print(f"  {name:26s} = {v:.4f}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

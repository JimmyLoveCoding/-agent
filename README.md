# 材料博物志

采用已确认的第九版书籍界面，接入团队的 Chemeleon-DNG + CrystalGNN 材料预测后端。前后端由一个本地服务提供，无需安装 Node.js。

## Windows 快速开始

1. 完整解压到一个文件夹。
2. 双击 **`start.cmd`**。首次启动会自动下载项目专用 Python 3.12.14，并安装依赖，可能需要数分钟；后续启动会复用已安装的环境。无需手动选择或更改系统 Python。
3. 准备完成后，浏览器自动打开 **http://127.0.0.1:8765/**。保留启动窗口；按 Ctrl+C 停止服务。
4. 在「化学式」中输入 `GaN`，或在「CIF」中选择随包提供的 `examples/silicon.cif`，即可使用本地模型。
5. 要使用「自然语言」，在 `backend/.env` 中填写自己的 `DEEPSEEK_API_KEY`，再重启服务。压缩包不含任何个人密钥。

端口被占用时，在项目目录运行 `start.cmd --port 8766`。不想自动打开浏览器可加 `--no-open`。

首次启动需要联网。Python 运行时只放在项目的 `.runtime/` 中，依赖放在 `.venv/` 中，不修改系统 PATH 或 Python 注册项。若安装中断，再次运行 `start.cmd` 会继续；错误会显示在窗口中，并保存到 `.run/startup.log`。`install.cmd` 仍可用于单独安装或修复环境。请不要从压缩包内部直接运行脚本。

安装器优先使用已验证的 `backend/requirements.lock.txt` 固定依赖版本；`requirements.txt` 保留项目直接依赖清单。默认 CPU 线程数为 4，可通过 `OMP_NUM_THREADS` 与 `MKL_NUM_THREADS` 环境变量覆盖。

## macOS / Linux

```bash
python3.12 scripts/setup.py
sh start.sh
```

固定依赖需要 Python 3.12，不支持以 Python 3.11 安装此锁定环境。CPU 可以运行；GPU 是否可用取决于安装的 PyTorch 与设备环境。

## 已接入的能力

- 自然语言查询与追问：发送实际 `/chat` 请求；对话由 DeepSeek 调用本地材料预测工具。
- 化学式预测：直接调用本地生成与预测链路，无需 DeepSeek 密钥。
- CIF 预测：上传实际晶体结构，直接交给 GNN，不生成替代结构。
- 翻页查看材料结果：形成能（eV/atom）、带隙（eV）、凸包上方能量（eV/atom）、候选数与方法说明。
- 查询记录保存在当前浏览器，可重新打开、清除，并导出真实结果为 JSON / CSV。
- 等待状态、服务状态、输入校验及失败重试；不展示虚构的预测数值或完成百分比。
- 118 个元素档案：按中文名、英文名、符号、原子序数搜索，按类别筛选，直接跳转或填入化学式查询。
- 双面纸页的 3D 翻页、封面轻微视差与金箔光泽；系统开启减少动态效果时直接切换页面。

元素卷涵盖全部 118 个元素，其中铜、硫、石墨、硅、铁、铝、钛、镍、锌、银、金、铋这前 12 个条目使用标本艺术插画，其余条目以周期表坐标与元素资料呈现。档案与本地模型预测分开标明：元素资料来源及快照见 `frontend/assets/element-sources.json`，未知性质保留“未知”，方括号质量数不冒充标准原子量。图鉴收录范围不等于模型已经验证的适用范围。

“向万物提问”纸页面板包含自然语言、化学式和 CIF 三种查询方式；回形针可直接选择 CIF 文件，输入限制与方法说明可从“使用说明”展开。页眉及“万物 / 有迹”标题沿用自然博物志的宋体、珊瑚枝与地质刻版风格。

## 预测含义

按化学式查询时，Chemeleon 生成 10 个候选结构，筛选代表结构后由 GNN 预测。生成结构未弛豫，返回值是模型估计，不是实验结果。CIF 查询直接使用提供的结构。首次使用会加载本地模型；CPU 采样可能需要数分钟。界面等待上限为 10 分钟；停止浏览器等待不保证中止后端计算。

本包已包含 `model.pt` 与 `ckpts/chemeleon_csp_alex_mp_20_v0.0.2.ckpt`。安装完成后，化学式与 CIF 推理不需要联网下载这些权重。自然语言仍需连接配置的模型服务。

## 目录

```text
materials-atlas/
  frontend/       页面、样式、交互和本地图片
  backend/        FastAPI、Agent、模型代码与两个权重
  examples/       可直接试用的 CIF 示例
  tests/          接口与交互验证
  scripts/        安装与打包辅助脚本
  run.py          同时提供前端和接口的启动入口
  install.cmd     Windows 首次安装
  start.cmd       Windows 启动
  start.sh        macOS / Linux 启动
```

## 接口

| 地址 | 请求 | 用途 |
| --- | --- | --- |
| `GET /health` | 无 | 服务状态、模型是否加载、聊天密钥是否配置 |
| `POST /chat` | `{ "message": "GaN 的带隙是多少？", "history": [], "context": [] }` | 自然语言与追问，context 可传最多 3 项已有预测 |
| `POST /predict` | `{ "formula": "GaN" }` | 直接化学式预测 |
| `POST /predict/cif` | `{ "filename": "silicon.cif", "content": "CIF 文件文本" }` | CIF 结构预测，文件上限 1 MiB |

预测响应包含 `answer` 与 `results` 数组。每条结果的 `prediction` 包含 `formation_energy_per_atom`、`band_gap`、`energy_above_hull`。解释性追问可以只有文字，没有新增预测。交互式接口说明位于 `/docs`。

## 配置与交付说明

`backend/.env.example` 列出可选的模型服务配置，复制成 `.env` 后填写。前端使用同源相对路径，不包含密钥或固定的开发机器路径。用户数据只保存在当前浏览器，清除站点数据会删除这些记录。

原始后端的模型权重和推理架构保留。接入层增加结构化结果、直接预测与 CIF 接口、输入限制、懒加载，以及与工作目录无关的权重加载。具体测试结果见 `verification/README.md`。

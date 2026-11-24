# MimicIK

**根据末端目标，学习生成连续的局部关节命令。**

[论文](https://arxiv.org/abs/2606.15148) · [预训练权重](https://github.com/RedAmancy918/MimicIK/releases) · [安装说明](docs/installation.md) · [数据格式](docs/data.md) · [复现说明](docs/reproducibility.md) · [English](README.md)

MimicIK 接收机器人当前关节状态、当前与目标末端位姿，以及短时观测历史，通过两次确定性网络调用预测关节增量。模型从遥操作配对记录中学习局部运动方式，输出命令为 `q_command = q_current + delta_q`。

本仓库从原 NatureIK 研究代码中整理出推理、训练和轨迹评测所需组件，提供 Python 接口、便携权重和可选的可微 FK 训练。每个模型对应明确的机器人与坐标约定，不提供未见机器人零样本控制。

## 安装

需要 Python 3.11 或更新版本。如需使用 CUDA，请先安装适合本机的 [PyTorch](https://pytorch.org/get-started/locally/)。

```bash
git clone https://github.com/RedAmancy918/MimicIK.git
cd MimicIK
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

推理仅依赖 PyTorch、NumPy 和 safetensors。训练和 URDF FK 的可选依赖可通过以下命令安装：

```bash
python -m pip install -e '.[train,fk]'
```

## 运行预训练模型

```bash
python -m mimicik.download --model airbot --output checkpoints/airbot
python examples/infer.py --checkpoint checkpoints/airbot
```

`airbot` 默认选择 `airbot-mip-fk-seed1`。运行
`python -m mimicik.download --list` 可查看三个已发布 AIRBOT 模型及其下载名称。

示例使用合成输入演示加载和推理，不连接机器人，也不构成轨迹精度验证。下载得到的 AIRBOT 发布包包含模型、归一化参数及来源/校验元数据。[历史模型集合](https://huggingface.co/vistar918/MimicIK)包含其他网络和训练版本，与本快速示例分开管理。

## Python 接口

```python
import numpy as np
from mimicik import MimicIK

policy = MimicIK.from_pretrained("checkpoints/airbot", device="cpu")

with np.load("data/my_robot/episode_000.npz", allow_pickle=False) as episode:
    q = episode["q"][0]
    eef_current = episode["eef"][0]
    eef_target = episode["target_eef"][0]

q_command = policy.step(q, eef_current, eef_target)
policy.reset()  # 新轨迹或控制状态重置后，清空历史。
```

- 关节角及命令单位为**弧度**，位置单位为**米**。
- EEF 位姿为 `[x, y, z, qx, qy, qz, qw]`，采用训练时的基座坐标系和 TCP 定义，四元数须为单位四元数。
- `step` 返回**绝对关节命令**，内部管理历史；下一次调用应输入实测状态。每条机械臂/控制流使用独立实例。
- `predict_delta(history)` 接收 `[H, DOF + 14]` 或 `[B, H, DOF + 14]` 原始历史，按时间从旧到新排列，返回关节增量，不修改内部历史。
- 夹爪独立控制。模型接口不执行碰撞检查、限速或滤波，接入真机时保留原控制器的保护层。

## 训练与评测

每条轨迹保存为一个 NPZ 文件，包含 `q`、`eef`、`target_q`、`target_eef`，并通过 `splits.json` 固定训练/验证/测试划分。也支持约定字段的 Parquet。字段形状、时间对齐和公开 OpenArm 数据来源见[数据说明](docs/data.md)。

```bash
python -m mimicik.training \
  --config configs/airbot.yaml \
  --data data/my_robot \
  --output runs/airbot \
  --device cuda --seed 0

python -m mimicik.evaluate \
  --checkpoint runs/airbot/final \
  --data data/my_robot --split val \
  --urdf assets/airbot.urdf --ee-link end_link \
  --output outputs/eval.json
```

`configs/airbot.yaml` 默认训练不含 FK 的 MIP。若启用 joint-v1 FK，请使用：

```bash
python -m mimicik.training \
  --config configs/airbot_fk.yaml \
  --data data/my_robot --output runs/airbot_fk \
  --urdf assets/airbot.urdf --end-link end_link \
  --device cuda --seed 0
```

训练前需让配置中的关节数、URDF 和末端 link 与自己的机器人一致。FK 是可选训练项，推理不需要 URDF 或 IK 求解器。

默认评测使用每帧录制状态；添加 `--protocol feedback` 后，以上一次预测命令作为下一步状态，并通过 FK 重建当前末端位姿。这是理想运动学反馈回放，假定命令被准确执行，不包含真实伺服或接触动力学。两种评测都离线运行。

AIRBOT 原始训练数据不随仓库分发。社区可以直接使用模型、训练自己的配对数据；复现 AIRBOT 数据集上的论文数值仍需要原始数据和固定划分。便携训练入口与历史实验管线的对应关系、指标定义见[复现说明](docs/reproducibility.md)。

## 引用与许可

论文 BibTeX 见 [English README](README.md#citation)，也可使用仓库中的 [CITATION.cff](CITATION.cff)。

MimicIK 基于 [Minimal Iterative Policy（MIP）](https://simchowitzlabpublic.github.io/much-ado-about-noising-project/)，可选 FK 使用 [PyTorch Kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics)。感谢这些项目及 [LeRobot](https://github.com/huggingface/lerobot) 社区。

代码采用 [MIT License](LICENSE)，上游归属见 [NOTICE](NOTICE)。外部数据与资产遵循各自许可。

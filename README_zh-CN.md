# UniCrossGait

这是论文 **“UniCrossGait: Unified Cross-Modal Gait Recognition Based on
Knowledge Distillation”**（*IEEE Transactions on Multimedia*, 2026）的官方参考代码。

[[论文](https://doi.org/10.1109/TMM.2026.3664966)]
[[IEEE Xplore](https://ieeexplore.ieee.org/document/11396985)]
[[OpenGait](https://github.com/ShiqiYu/OpenGait)]
[[English](README.md)]

> **代码定位：** 本仓库是面向论文核心方法的 OpenGait 扩展，而不是完整的 OpenGait
> 分支或一键复现工程。仓库提供模型、损失、SUSTech1K 跨模态评测入口和配置模板；
> 不提供受许可约束的数据、预处理结果、训练权重或完整运行环境。不同 OpenGait
> 版本之间可能需要少量接口适配。

## 核心内容

UniCrossGait 分两阶段训练：

1. 使用相机轮廓图和 LiDAR 深度图训练多模态教师网络，通过双向跨注意力获得融合表示；
2. 冻结教师，同时蒸馏 2D 与 3D 学生。两个学生共享 `SeparateFCs` 和
   `SeparateBNNecks`，使用 DFI、IntraC、InterC、Distillation Balancing、
   Warm-up 以及双向跨模态三元组损失进行训练。

推理时只需学生分支。公式与代码对应关系见 [docs/METHOD.md](docs/METHOD.md)。

## 接入 OpenGait

在本仓库根目录运行：

```bash
# 默认仅预览将要执行的操作
python scripts/install_into_opengait.py /path/to/OpenGait

# 确认后写入 OpenGait
python scripts/install_into_opengait.py /path/to/OpenGait --apply
```

安装脚本只复制 UniCrossGait 新增文件，并在 OpenGait 的 evaluator 中增加一行注册。
除非显式传入 `--force`，否则不会覆盖已有的同名文件。手工接入方式见
[docs/INTEGRATION.md](docs/INTEGRATION.md)。

训练前必须修改两个 YAML 中的数据根目录、划分文件和 `data_in_use`。模板约定输入
顺序为 `[LiDAR 深度图, 相机轮廓图]`，分别对应索引 0 和 1；论文使用的图像尺寸为
64×64，SUSTech1K 训练身份数为 250。

## 训练命令

以下命令在 OpenGait 根目录执行：

```bash
# 第一阶段：训练教师
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 opengait/main.py \
  --cfgs configs/unicrossgait/unicrossgait_sustech1k_teacher.yaml \
  --phase train

# 第二阶段：先填写 teacher_checkpoint，再训练学生
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 opengait/main.py \
  --cfgs configs/unicrossgait/unicrossgait_sustech1k_student.yaml \
  --phase train
```

## 论文结果

下表为论文报告值，并非本参考仓库重新运行所得：

| 数据集 | 检索方向 | Rank-1 | Rank-5 |
| --- | ---: | ---: | ---: |
| SUSTech1K | 2D → 3D | 56.7 | — |
| SUSTech1K | 3D → 2D | 59.7 | — |
| FreeGait | 2D → 3D | 49.2 | 82.5 |
| FreeGait | 3D → 2D | 56.6 | 84.9 |

## 与研究快照的关系

原始 `PointGait.py` 同时保留了主实验、注释掉的模块和消融分支。本仓库将论文最终方法
拆成清晰模块：DFI 使用余弦方向约束；InterC 先对 parts 做最大池化；主方法不使用
logit 蒸馏。论文 Eq. (28) 的紧凑表达没有列出学生身份分类，而作者训练快照中实际启用
了共享 BNNecks 上的分类损失，因此模板默认 `use_student_ce: true`；如需严格按
Eq. (28) 可关闭该选项。这一差异在英文 README 和复现说明中均已显式记录。

## 引用与许可

BibTeX 见英文 [README](README.md) 或 [CITATION.cff](CITATION.cff)。本仓库原创扩展
代码仅供非商业学术研究与教学使用，具体见 [LICENSE](LICENSE)。OpenGait 和数据集
分别遵循其自身条款。

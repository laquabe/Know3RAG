# CoDEx factual-check 模型训练

在运行过 `libkge_setup.sh` 的服务器上，激活安装 LibKGE 的 Python 环境。
将 `train_codex_kge.py` 复制到服务器，或者从 Know3RAG 仓库调用它。
每次只训练一个模型，不会下载权重、安装依赖或重新预处理数据。

```bash
python scripts/train_codex_kge.py \
  --codex-root /data/xkliu/codex \
  --model transe \
  --device cuda:0
```

`--codex-root` 改成服务器上的 CoDEx 仓库路径；`--model` 可改为
`transe`、`conve`、`rescal`。默认读取
`models/triple-classification/codex-m/<model>/config.yaml`。
保留官方模型结构（包括 reciprocal relations 包装）、优化器和训练策略，
不把 `model` 配置粗暴覆盖成模型名称。

先检查文件并预览命令：加 `--dry-run`。此选项不验证 CUDA 或实际加载模型。
可用 `--epochs 5 --batch-size 64` 做短训练；这会改变官方实验设置。
不指定时沿用各模型配置与该版本 LibKGE 的默认值，可能由 early stopping 提前结束。

默认输出：

```text
<codex-root>/local-runs/triple-classification/codex-m/<model>/
```

该目录由 LibKGE 保存配置、日志和 checkpoint；相邻的
`<model>.console.log` 保存终端输出，`<model>.launcher.json` 记录启动信息。
`checkpoint_best.pt` 是 LibKGE 按配置中的验证指标选择的权重，
不应直接称为“factual-check 准确率最优权重”。

恢复训练（`--epochs` 为累计总轮数）：

```bash
python scripts/train_codex_kge.py \
  --codex-root /data/xkliu/codex --model transe --device cuda:0 \
  --resume --epochs 400
```

新的独立实验可指定 `--output /data/xkliu/runs/transe-run2`；恢复该实验时也要
传入同一个 `--output`。脚本拒绝覆盖已有输出目录。断点恢复需要已保存的
checkpoint；第一次 checkpoint 写出前中断的实验应另选输出目录重新启动。

这里训练的是 KGE 三元组评分模型，使用作者的 triple-classification 实验配置。
它不会自动运行作者的分类评估、拟合分类阈值，或修改 Know3RAG 的 factual-check 配置。
接入前还需要使用 CoDEx 对应实体/关系映射，并在自己的验证数据上校准评分。
官方来源：https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m

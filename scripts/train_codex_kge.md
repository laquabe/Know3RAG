# CoDEx KGE 模型训练

在运行过 `libkge_setup.sh` 的服务器上，激活安装 LibKGE 的 Python 环境。
将 `train_codex_kge.py` 复制到服务器，或者从 Know3RAG 仓库调用它。
每次只训练一个模型，不会下载权重、安装依赖或重新预处理数据。

```bash
python scripts/train_codex_kge.py \
  --codex-root /data/xkliu/codex \
  --task triple-classification \
  --size m \
  --model transe \
  --device cuda:0
```

`--codex-root` 改成服务器上的 CoDEx 仓库路径；`--model` 可改为
`transe`、`conve`、`rescal`。`--task` 支持 `triple-classification`（默认）和
`link-prediction`。`--size` 支持 `s`、`m`（默认）、`l`，读取
`models/<task>/codex-<size>/<model>/config.yaml`。

官方配置支持情况（以上三个模型均适用）：

| 任务 | 可选规模 |
| --- | --- |
| triple-classification | s、m |
| link-prediction | s、m、l |

`triple-classification --size l` 会明确报错，不会借用其他任务的配置。
例如训练 link prediction 版本：

```bash
python scripts/train_codex_kge.py \
  --codex-root /data/xkliu/codex \
  --task link-prediction --size l --model conve --device cuda:0
```

保留官方模型结构（包括 reciprocal relations 包装）、优化器和训练策略，
不把 `model` 配置粗暴覆盖成模型名称。

先检查文件并预览命令：加 `--dry-run`。此选项不验证 CUDA 或实际加载模型。
可用 `--epochs 5 --batch-size 64` 做短训练；这会改变官方实验设置。
不指定时沿用各模型配置与该版本 LibKGE 的默认值，可能由 early stopping 提前结束。

默认输出：

```text
<codex-root>/local-runs/<task>/codex-<size>/<model>/
```

该目录由 LibKGE 保存配置、日志和 checkpoint；相邻的
`<model>.console.log` 保存终端输出，`<model>.launcher.json` 记录启动信息。
`checkpoint_best.pt` 是 LibKGE 按配置中的验证指标选择的权重，
不应直接称为“factual-check 准确率最优权重”。

恢复训练（`--epochs` 为累计总轮数）：

```bash
python scripts/train_codex_kge.py \
  --codex-root /data/xkliu/codex --task triple-classification --size m \
  --model transe --device cuda:0 \
  --resume --epochs 400
```

新的独立实验可指定 `--output /data/xkliu/runs/transe-run2`；恢复该实验时也要
传入同一个 `--output`。恢复时 `--task`、`--size` 和 `--model` 必须与原实验一致；
旧记录缺少 task 时按 `triple-classification` 处理，缺少 size 时按 `m` 处理。
脚本拒绝覆盖已有输出目录。断点恢复需要已保存的
checkpoint；第一次 checkpoint 写出前中断的实验应另选输出目录重新启动。

这里训练的是 KGE 三元组评分模型，使用作者对应任务的实验配置。
它不会自动运行作者的分类评估、拟合分类阈值，或修改 Know3RAG 的 factual-check 配置。
接入前还需要使用 CoDEx 对应实体/关系映射，并在自己的验证数据上校准评分。
官方来源：https://github.com/tsafavi/codex/tree/master/models

## 旧版 Ax / SQLAlchemy 导入冲突

若启动时报 `SQAGeneratorRun.arms` / `MappedAnnotationError`，是旧 Ax 的 ORM
注解与 SQLAlchemy 2.x 不兼容。即使只训练单个模型，旧 LibKGE 也会在导入
`kge.job` 时导入 Ax 搜索模块，从而触发此错误。
在服务器的 `kge_codex` 环境中运行：

```bash
conda activate kge_codex
python -m pip install "SQLAlchemy==1.4.54"
python -m pip check
python -c "import sqlalchemy; print(sqlalchemy.__version__); import kge.cli; print('LibKGE import OK')"
```

这是针对此导入错误的兼容性修复，不保证其余旧依赖均兼容。
修复后重跑原训练命令。上述导入错误发生在训练目录创建之前，通常无需 `--resume`。
不要重新执行整个 `libkge_setup.sh`，也不要删除数据。
新版启动器会在创建运行记录之前检查实际 CLI 导入链，失败时给出对应提示；
`--dry-run` 仍只检查配置和文件，不检查运行环境。

# 参数扫描与最终答案融合

## EL 覆盖分层评测

使用 [`el_coverage.py`](el_coverage.py) 读取一份 EL 和数据集 gold，支持通过
`--prediction-file` 直接读取 `{"answer": {"id": "最终答案"}}`，跳过阈值选择和答案
提取；也支持读取两轮回答后按动态阈值选择。输出整体及完全覆盖／部分覆盖／未覆盖
的 EM/F1，映射不足题单列。
两轮默认字段均为 `old_llm_response`、`llm_response`、`llm_triple_score`，每轮可独立
传参修改。支持 HotpotQA、2Wiki 和 PopQA；命令与覆盖口径见
[`el_coverage.md`](el_coverage.md)。

## KGE 模型消融：两种阈值模式、全量与子集

使用 [`kge_ablation.py`](kge_ablation.py) 读取两份回答文件和两份新评分文件，
两轮都默认使用 `old_llm_response` / `llm_response`。支持直接传入两组阈值，
或传入 `--theta-values` / `--c-values` 按论文公式计算；
只运行 `threshold0 < threshold1` 的组合，同时输出全量和“至少一轮有有效评分”的子集。
可选 `--random --seed 42`，在两轮均无有效评分时从三个候选回答中等概率随机选择。
支持固定子集 ID，以及 HotpotQA、2Wiki 的全量/子集 EM/F1。
完整命令、字段说明和输出格式见 [`kge_ablation.md`](kge_ablation.md)。

## 一次运行直接出指标（推荐）

`sensitivity.py` 将融合、历史答案提取和原评估脚本串起来，无需手动再次测试。
先确保评估 Python 安装了 `ujson`（`python -m pip install ujson`）。

HotpotQA 示例：

```bash
python analyse/sensitivity.py \
  --dataset hotpot \
  --turn0-input /path/turn0.jsonl --turn1-input /path/turn1.jsonl \
  --turn0-old-answer-key turn0_response --turn0-new-answer-key llm_response \
  --turn0-score-key turn0_triple_score \
  --turn1-old-answer-key old_llm_response --turn1-new-answer-key llm_response \
  --turn1-score-key llm_triple_score \
  --theta-values 2.5 5 10 20 --c-values 4 16 128 256 \
  --gold-file /path/hotpot_dev_distractor_v1.json \
  --output-dir result/hotpot_sensitivity --save-details
```

2WikiMultiHopQA（额外传 JSONL aliases）：

```bash
python analyse/sensitivity.py \
  --dataset 2wiki \
  --turn0-input /path/turn0.jsonl --turn1-input /path/turn1.jsonl \
  --theta-values 0.5 1 2 4 --c-values 4 16 128 256 \
  --gold-file /path/dev.json --alias-file /path/id_aliases.json \
  --output-dir result/2wiki_sensitivity
```

PopQA（使用原来转换为 2Wiki 格式、含 `_id` 和字符串 `answer` 的 gold）：

```bash
python analyse/sensitivity.py \
  --dataset popqa \
  --turn0-input /path/turn0.jsonl --turn1-input /path/turn1.jsonl \
  --theta-values 0.025 0.05 0.1 0.2 --c-values 4 16 128 256 \
  --gold-file /path/test.json --output-dir result/popqa_sensitivity
```

后两个例子使用默认字段，均可像 Hotpot 示例一样修改。`--eval-python /path/to/python`
可指定评估解释器，默认与主脚本相同。

服务器上解析和评估代码不在项目目录时，增加：

```bash
--dataset-test-dir /data/xkliu/dataset_test
```

该目录下应有 `hotpot/`、`2wikimultihop/`、`popqa/` 中本次数据集对应的子目录，
其中同时存放 `phrase_ans.py` 和相应评估脚本。目录可以使用任意名称。
默认仍为项目根目录的 `dataset_test`；相对路径按命令运行目录解析，建议服务器使用绝对路径。
gold、aliases 和输入文件继续使用各自的路径参数，不自动从该目录查找。

运行结束直接查看 `summary.csv` 或 `summary.json`：每行包含 θ₀、c、两轮阈值、
选择来源数量、EM/F1/P/R、覆盖数量和运行状态。**汇总指标统一为 0–100 百分数**。
2Wiki 和 PopQA 保留原脚本的输出精度，不额外重算指标。

中间产物：

- `predictions/theta_10__c_128.json`：`answer[id]` 是选中回复提取出的短答案，
  可直接交给原评估器；同时提供空 `sp` 和 `evidence`，兼容 PopQA 接口。
- `metrics/*.json`：原评估输出、原始单位及统一百分数后的答案指标。
- `logs/*.log`：完整评估命令、stdout、stderr；PopQA 的 missing sp/evidence
  提示是没有提供支持事实预测的正常表现，sp/evidence/joint 不用于本实验汇总。
- `details/*.jsonl`：仅 `--save-details` 时生成，记录 `selected_source`、
  最终完整回复（默认 `llm_response`）、短答案 `prediction` 和评分。

解析严格调用相应数据集 `phrase_ans.py` 的 `phrase_answer()`；保留历史规则，
包括未匹配句式时返回整段回复。不会执行该文件主程序中的写死路径，也不会改动它。
实际评估入口分别为 `hotpot/hotpot_evaluate_v1.py`、
`2wikimultihop/2wikimultihop_evaluate_v1.1.py`、`popqa/2wikimultihop_evaluate.py`，
均位于 `dataset_test` 下并保持原样。

以提供的完整 gold 为分母；不自动按输入裁剪，缺失和额外 ID 数会写入汇总。
gold 不参与答案选择。输入错误、依赖缺失或输出冲突在运行前报错；单组评估失败
则记录 failed 和空指标并继续其他组合，最后返回非零退出码。汇总每组完成即更新。

## 仅融合，不测评

只需 Python 标准库。两个文件均为 JSONL，每行一题；用 ID 对齐，不要求行顺序相同。
不会调用模型、解析答案、读取 gold 或计算 EM/F1。

```bash
python analyse/merge.py \
  --turn0-input /path/turn0.jsonl \
  --turn1-input /path/turn1.jsonl \
  --turn0-id-key id --turn1-id-key id \
  --turn0-old-answer-key turn0_response \
  --turn0-new-answer-key llm_response \
  --turn0-score-key turn0_triple_score \
  --turn1-old-answer-key old_llm_response \
  --turn1-new-answer-key llm_response \
  --turn1-score-key llm_triple_score \
  --theta-values 2 5 10 20 \
  --c-values 2 4 16 128 256 \
  --output-dir result/sensitivity
```

上述 key 都是默认值，若文件字段相同可省略；每个 key 指直接 JSON 字段名。
`--triple-value-key triple_score`、`--reference-scores-key ref_score` 指三元组列表内部的字段。
`--output-answer-key llm_response` 指最终答案写入的字段。
θ₀ 与 c 各传一个值即运行单组。θ₀ 必填，c 默认 128；重复值自动去重。

## 评分与选择

每份文件的评分都计算一次：逐三元组求 `abs(triple_score - mean(ref_score))`，
再取均值。空参考列表跳过；没有有效项时该份评分为 `None`。

每组参数计算：

```text
threshold0 = theta0
threshold1 = theta0 * (c / (1 + exp(1 - theta0)))
```

```python
if score0 is not None and score0 < threshold0:
    final = turn0_old_answer
elif score1 is not None and score1 < threshold1:
    final = turn1_old_answer
else:
    final = turn1_new_answer
```

严格使用 `<`，等于阈值时继续下一分支；不读取 local_check。
turn0 的新答案仅保存供核对，不参与最终选择。
第二份的旧答案及评分可能来自此前过滤保留的第 0 轮：使用输入原值，不核查它
与第一份新答案是否一致，不推断真实轮次。每组参数都从原输入独立选择。

## 输出

- 每组一个 `theta_10__c_128.jsonl` 等文件，输出顺序跟随第一份的 ID 顺序。
- 以第二份记录为基础，最终答案默认覆盖 `llm_response`。原始四个答案保存在
  `merge_info.candidates`，另记录来源 ID、分数、阈值、参数和 `selected_source`。
- `selected_source` 是 `turn0_old`、`turn1_old` 或 `turn1_new`，表示来源，
  不保证它就是实际停止轮次。
- `summary.json` 保存运行配置及每组的样本数、各来源选择数、两份输入的无评分数。
  无评分数针对全部样本统计，不仅针对进入第二分支的样本。

输入 ID 集合必须一致，重复 ID 或缺失必要字段报错；评分必须合法有限。
全部输入及输出路径预检通过后开始输出。已有同名输出默认拒绝覆盖，
明确指定 `--overwrite` 才覆盖本次涉及的文件；不会删除目录中的其他参数结果。
禁止覆盖输入，也禁止输入本身已有 `merge_info`（避免将融合结果误当原始输入）。

后续可用 `scripts/eval_hotpot.py` 对每组结果评估。这里不把 gold 用于选择。

## 参数敏感性绘图

安装 `matplotlib` 后，在项目根目录运行（默认只画 F1）：

```bash
python analyse/plot_sensitivity.py
python analyse/plot_sensitivity.py --em --f1
python analyse/plot_sensitivity.py --f1 --c-ylim 0.35 0.40 --theta-ylim 0.34 0.40
python analyse/plot_sensitivity.py --f1 --theta-min 0  # 恢复完整 θ₀ 范围
```

默认读取 `result_202608/sensitivity/hotpotqa_qwen_summary.csv` 和
`result_202608/sensitivity/2wiki_glm4_summary.csv`，可通过 `--qwen-csv`、
`--glm4-csv` 替换。路径默认相对于脚本所在项目，不依赖运行目录。
CSV 需要 `theta0,c,status` 和选中的 `em,f1` 列；指标必须为 **0–100 百分数**，
绘图时除以 100 转为 0–1。非成功状态、重复参数对、非法数值会报错。

c 图固定 Qwen 的 θ₀=1、GLM4 的 θ₀=2；θ₀ 图固定 c=128。
可用 `--qwen-theta`、`--glm4-theta`、`--fixed-c` 覆盖。
θ₀ 图默认仅展示 θ₀≥0.5 的点，可用 `--theta-min` 改变下限（含边界），
`--theta-min 0` 恢复完整正值范围；此选项不影响 c 图或原始 CSV。
图注中应说明展示范围；终端也会打印实际绘制的 θ₀ 点。
读取展示范围内全部点并排序，两组横坐标必须一致且大于零。
c 使用底数为 2 的对数轴，θ₀ 使用底数为 10 的对数轴，刻度显示实际值。

`--em` 仅画 EM，`--f1` 仅画 F1，同时传入则两者都画。
EM 为绿色、F1 为橙色；Qwen 为实线圆点，GLM4 为虚线方块。
图例放在坐标轴内，由 Matplotlib 自动选择尽量不遮挡曲线的位置。
标题使用 `Effect of c` 或 `Effect of θ₀`，固定参数设置放在标题后的括号内。
默认每张图根据全部显示曲线独立计算纵轴：两端留 8% 余量，至少 0.002 总跨度，
再向外取整为易读刻度，限制在 0–1 内。纵轴不强制从零开始。
同时显示 EM/F1 时共用同一纵轴，因此细微变化会更平缓。

`--ylim MIN MAX` 同时控制两张图，`--c-ylim`、`--theta-ylim` 优先覆盖对应图。
范围必须满足 `0 ≤ MIN < MAX ≤ 1`，裁掉任何点默认报错；显式传入
`--allow-clipping` 才允许裁剪，并打印裁剪点数。终端始终打印最终纵轴范围。

默认输出到 `result_202608/sensitivity/figures`，可通过 `--output-dir` 更改。
每张图生成矢量 PDF 和 300 DPI PNG，例如 `sensitivity_c_f1.pdf`、
`sensitivity_theta_em_f1.png`；重复运行会覆盖同名图，如需保留不同固定参数的
版本，请使用不同输出目录。优先 Times New Roman，缺失时使用 DejaVu Serif；
无需 LaTeX。脚本只读取汇总 CSV，不重新融合或评估，也不修改输入。

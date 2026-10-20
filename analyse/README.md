# 参数扫描与最终答案融合

## 配对 Bootstrap：主结果置信区间

使用 [`paired_bootstrap.py`](paired_bootstrap.py) 比较两份已解析答案 JSON，复用历史
评测器计算逐题 EM/F1，以相同题目索引有放回抽样，报告自身指标与双方差值的 95% 区间。
默认 10,000 次重采样、种子 42；不重新运行模型。主表九组的最强基线比较可用
[`bootstrap_main_results.py`](bootstrap_main_results.py) 批量执行，命令和解释边界见
[`paired_bootstrap.md`](paired_bootstrap.md)。

## 答案长度与完整包含关系诊断

使用 [`answer_diagnostics.py`](answer_diagnostics.py) 分析两份已解析答案，报告平均词数、
中位数、P90、本方法更长的比例，以及 EM=0 答案中完整包含参考答案且有额外词的题数及占比。
复用原评测器和 bootstrap 输入校验；2Wiki/PopQA 保留 gold answer、aliases、demonyms。
不重新解析答案、不修改原 EM/F1、不执行 bootstrap。三个数据集的运行示例、输出及解释边界见
[`answer_diagnostics.md`](answer_diagnostics.md)。
可选 `--f1-thresholds` 扫描高 F1 但 EM=0 的答案，并对称统计同题 F1 高于另一方法的样本。
默认阈值为 `0 0.25 0.5 0.6 0.7 0.8 0.9 1`；单独输出扫描表，不选择最优阈值。

## EM 四分类与 F1 配对分解

使用 [`em_f1_decomposition.py`](em_f1_decomposition.py) 补齐审稿人点名的三组主表比较，
按双方原 EM 状态报告四组题数、双方平均 F1 及各组对总体 ΔF1 的贡献。
原评测器核对与四组贡献还原检查均通过后才输出；2Wiki 保留 aliases 和 demonyms。
固定输入清单、运行命令和统计口径见 [`em_f1_decomposition.md`](em_f1_decomposition.md)。

## Bad case 候选粗筛与过程阅读

[`bad_case_candidates.py`](bad_case_candidates.py) 默认处理 `result_202608/ref/` 中的
2Wiki / Qwen 主实验缓存：最终 EM=0、两份 Llama 缓存的 `query_entity` QID 并集未命中
任何 gold 支撑实体、前两次回答检查均未放行，且最终选择轮次上限处的回答。
使用原答案解析器、评测器及 aliases/demonyms；全部预测重建与主实验一致后才生成报告。
需要 Python 3.9+、NumPy、ujson，以及本目录已有的分析模块。

```bash
python analyse/bad_case_candidates.py
```

默认输出 `result_202608/bad_case_candidates/` 中的 `candidates.jsonl`、`summary.json`
和可离线打开的 `review.html`。阅读报告每页 20 题，可搜索问题/ID/答案，按四类 KGE
检查状态与 F1=0 / F1>0 筛选；默认 F1、题目 ID 升序。EM/F1 使用 0–1 单位。
三元组分数对应每份 `wait4merge` 中的 `old_llm_response`；默认 θ₀=0.2、c=128。
严格小于阈值才通过，等于阈值继续；无有效参考分数记为无法评分，不填入虚构高分。
提前接受的回答排除，gold 映射或任一 EL 来源记录缺失的题目也不作为零覆盖候选。

passage 级 local check 和 triple score **仅补充阅读材料，不参与粗筛**。可关闭补充：

```bash
python analyse/bad_case_candidates.py \
  --skip-reference-checks \
  --output-dir result_202608/bad_case_candidates_without_checks
```

默认仍从两份 Llama 文件读取实体；有独立实体文件时，可用 `--el-files /path/el1.jsonl
/path/el2.jsonl` 替换。每个 EL 来源须覆盖待筛题目，文件内允许同题多个快照并取并集。
缓存输入使用 JSONL（允许 `.json` 后缀），gold 与最终预测使用历史 JSON 格式。
其他路径通过 `--turn0-input`、`--turn1-input`、`--prediction-file`、`--gold-file`、
`--alias-file`、`--dataset-test-dir` 覆盖；该版本只处理 2Wiki。使用另一组实验时须同时
设置对应的 `--theta0`、`--c`、`--backbone` 和全部输入。已有输出需显式 `--overwrite`，
始终禁止覆盖输入。HTML 模板 [`bad_case_review.html`](bad_case_review.html) 应与脚本一起保留。

候选只是粗筛：EM=0、高 KGE 分数或实体未命中不能单独证明事实错误或失败原因。
并集不是某轮真实查询；参考未出现不自动等于被过滤；未提供的检查保留未知。
报告分别保留原始回答、实际评分三元组、gold 证据与来源行号，并留出人工分析说明。
人工整理的错误事实可以包含系统未抽到的关系，但须与实测三元组和分数分开。

```bash
python -m unittest discover -s tests -p 'test_bad_case_candidates.py' -v
```

## Bad case 二次筛选：同名错链线索

[`bad_case_shortlist.py`](bad_case_shortlist.py) 只读取已生成的粗筛 `candidates.jsonl`，
复用 `el_accuracy.py` 的规范化、名称对齐和位置有效性判断，寻找问题中完整出现的
直接 mention 与同题唯一 gold QID 不一致的线索。先严格同名，再允许 gold 名称
去掉末尾括号；歧义匹配、扩展实体、无效位置和不在问题中的 mention 不作为线索。
位置只用于区分直接链接，不按字符下标截取原问题。无需第三方 Python 包或模型调用。

```bash
python analyse/bad_case_shortlist.py
python -m unittest discover -s tests -p 'test_bad_case_shortlist.py' -v
```

默认输出到 `result_202608/bad_case_shortlist/`，只有 `candidates.jsonl` 和 `summary.json`。
可用 `--input`、`--output-dir` 替换路径；重复生成需显式 `--overwrite`，禁止覆盖输入。
不重跑粗筛、答案评测或 KGE 评分，不处理 passage 检查，不生成网页。

每条输出完整保留原候选，新增 `second_pass`：两档匹配结果、原始 mention、预测 QID、
目标名称与描述、对应 gold QID 与名称、原记录及来源行号、身份信息完整度、来源冲突、
实体数量、排序键与顺序号。严格档只需至少一条严格线索；同题的去括号线索仍完整保留。
目标信息来自各自原始记录，绝不跨来源拼接。`manual_analysis` 原样保留，默认仍待人工阅读。

排序为严格档优先、身份信息齐全优先、无跨来源冲突优先、错链 mention 数升序、直接
mention 数升序、F1 升序、题目 ID 升序。身份信息齐全要求**每条入选错链记录各自**都有
非空名称和描述；跨来源冲突指不同源文件对同一规范化直接 mention 给出不同 QID。
同一文件内多个快照不算跨来源，记录仍保留。mention 数按规范化名称跨来源去重，直接
mention 数包含所有位置有效记录；记录总数及并集 QID 数另存。数量和元数据仅影响排序，
不设截断、不丢弃相应候选。所有 gold 关系、支撑句、逐轮回答与系统行为留待阅读。

输出是名称对齐后的 QID 不一致线索，**不自动确认事实错误、错链语义或失败原因**。
人工错误事实分析与系统实际检查行为继续分开；未抽取关系不附加系统原始评分。

## EL 准确率与支撑实体召回率

使用 [`el_accuracy.py`](el_accuracy.py) 离线评估 2Wiki、HotpotQA、PopQA 的两档
mention 对齐 QID 一致率，以及直接链接和完整 `query_entity` 的支撑实体召回率。
无需模型推理或答案评估；映射不完整题单列。三套命令及指标边界见
[`el_accuracy.md`](el_accuracy.md)。

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
  --gold-file /path/test.json --alias-file /path/id_aliases.json \
  --output-dir result/popqa_sensitivity
```

PopQA 使用 `popqa/popqa.py`，需要 gold 中的 `answer_id` 及完整 aliases 文件。
此前误用单答案评估器的 PopQA 结果需离线重评；修正后的敏感性和 EL 覆盖脚本
都会使用答案别名，并在汇总中记录评估器路径及 `answer_aliases_enabled`。

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
- `logs/*.log`：完整评估命令、stdout、stderr；仅汇总答案评估指标。
- `details/*.jsonl`：仅 `--save-details` 时生成，记录 `selected_source`、
  最终完整回复（默认 `llm_response`）、短答案 `prediction` 和评分。

解析严格调用相应数据集 `phrase_ans.py` 的 `phrase_answer()`；保留历史规则，
包括未匹配句式时返回整段回复。不会执行该文件主程序中的写死路径，也不会改动它。
实际评估入口分别为 `hotpot/hotpot_evaluate_v1.py`、
`2wikimultihop/2wikimultihop_evaluate_v1.1.py`、`popqa/popqa.py`，
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

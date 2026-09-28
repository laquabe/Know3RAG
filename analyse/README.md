# 参数扫描与最终答案融合

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

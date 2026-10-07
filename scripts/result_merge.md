# 按动态阈值融合答案

`code/utils.py process_by_line --func result_merge` 读取 JSONL，每条记录须同时
包含旧答案、新答案及**旧答案**的三元组评分列表。不读取 `local_check`。

阈值按论文 Eq. (3) 计算：

```text
theta_t = theta0 * (c / (1 + exp(1 - theta0))) ** t
```

`--theta` 是 theta0，`--c` 默认 128，`--T` 是当前被评估的旧答案轮次，
从 0 开始，不是最大迭代预算：turn0 → turn1 用 0，turn1 → turn2 用 1。
每次调用使用相同的 theta0 和 c，只改变 T。

评分沿用已有 `score_feature`：对每个有参考分数的三元组计算
`abs(triple_score - mean(ref_score))`，再对有效三元组取均值。
这与修稿清单的均值约定一致；原稿 Eq. (1) 的求和符号尚待修正。
分数严格小于阈值时选旧答案；大于等于阈值或没有有效三元组时选新答案。
没有有效评分时继续使用新答案是本代码的明确回退规则。

示例（theta=1 仅为运行示例，实际按实验设置指定）：

```bash
python code/utils.py process_by_line \
  --input_file turn01_input.jsonl --output_file turn01_merged.jsonl \
  --func result_merge \
  --old-response-key turn0_response --new-response-key llm_response \
  --triple-score-key turn0_triple_score --theta 1 --c 128 --T 0

python code/utils.py process_by_line \
  --input_file turn12_input.jsonl --output_file turn12_merged.jsonl \
  --func result_merge \
  --old-response-key old_llm_response --new-response-key llm_response \
  --triple-score-key llm_triple_score --theta 1 --c 128 --T 1
```

输出保留原字段，默认将选中的答案写入 `llm_response`；可通过
`--merge-output-key merged_response` 保留原始新答案并另存选择结果。
输入和输出必须是不同文件。字段缺失会报错，不自动猜测字段或评分归属。

每次调用只比较给定的两轮答案。多轮融合时，前面已通过阈值而停止的样本
应保留其已选答案，不应被后续一次独立融合覆盖。

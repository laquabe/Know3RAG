# QA 答案评估

`eval_hotpot.py` 直接读取融合结果 JSONL，默认用 `llm_response` 预测、`answer`
标准答案和 `id` 标识。也支持 JSON 数组。依赖现有 `numpy`，不加载模型。

```bash
python scripts/eval_hotpot.py \
  --input result/merged.jsonl \
  --answer-key llm_response \
  --output result/metrics.json \
  --details result/per_question.jsonl
```

EM、token F1、Precision、Recall 遵循
[HotpotQA 官方脚本](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)
的答案指标定义：小写、删除 ASCII 标点、删除 a/an/the、合并空白；
yes/no/noanswer 与标准答案不一致时 F1/P/R 为零。逐题平均，数值为 0–1，
乘以 100 为百分数。不计算支持事实或 joint 指标，不按第一个句点截断答案。

答案提取属于本项目的额外预处理，不是官方评估器的一部分：

1. 若输出中存在含字符串 `answer` / `Answer` 的 JSON 对象，读取该字段。
2. 复用 `utils/data_io.py`：识别 The best answer is、Best answer is、
   The answer is、Answer is、Final answer:、Final answer is、Answer:。
   忽略大小写，取最后出现的标记后的第一个非空行，清理外围标记和末尾标点。
3. 无明确标记时沿用末尾短答案 / 整段文本回退；此类记录标为 `fallback`。

不会利用标准答案辅助提取。不会从推理文本中搜索 gold 来判对，也不会自动
删除答案后的同一行解释。例如 `yes, because ...` 可能不匹配 `yes`，应检查
逐题结果，而不是用 gold 修正解析。已提取好的短答案可用 `--extraction raw`。

汇总包含提取方式计数、缺失预测数和评估题数；逐题文件包含原回复、提取答案、
标准答案和各项指标，方便审计。默认只评估输入中存在的题目，不能发现整题缺失。
正式完整集评估可提供官方 gold 文件：

```bash
python scripts/eval_hotpot.py \
  --input result/merged.jsonl --gold-file datasets/hotpot_dev_distractor_v1.json \
  --gold-id-key _id --output result/metrics.json --details result/per_question.jsonl
```

提供 gold 文件后按 ID 对齐，缺失预测计零并保留在分母，额外预测不参与评分，
重复 ID 报错。标准答案目前必须是字符串；多别名列表需另行定义评估协议，
不能直接将本脚本当作 2Wiki / PopQA 官方完整评估器。

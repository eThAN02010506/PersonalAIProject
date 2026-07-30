# Qwopus-Agent 评估与真实案例

本文定义发布前的确定性回归和环境相关真实案例。评估结果只用于发现退化，不能替代用户
对最终内容是否有用的判断。

## 1. 确定性 P0 回归

运行固定的 P0 接受测试：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest \
  tests.test_p0_acceptance \
  tests.test_graph_multiformat_realcase \
  tests.test_conversation_knowledge \
  tests.test_saved_documents_api
```

该集合覆盖：

- 大型多语言文档尾部事实经过切片、持久化和重启后仍可检索；
- PDF、DOCX、TXT、XLSX 的真实文件解析和跨格式四跳图路径；
- 多表、非英文 Excel 的区域识别；
- 会话私库、账号 Global 聚合和删除边界；
- 多份已保存文档的附加、检索和访问校验。

## 2. 检索质量

`qwopus_agent.evaluation.evaluate_retrieval` 使用显式参考来源计算：

- **Source recall**：期望来源中有多少被召回；
- **Source precision**：返回来源中有多少属于期望集合；
- **Reciprocal rank**：第一个相关结果是否足够靠前；
- **Forbidden hits**：不应跨范围出现的来源是否泄漏。

每个新检索案例应包含稳定查询、期望来源、禁止来源和最低阈值。来源级指标用于检测文件
漏召回、噪声和 ACL 泄漏；不能只断言结果列表非空。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest tests.test_retrieval_evaluation
```

## 3. 回答质量

`qwopus_agent.evaluation.evaluate_answer` 复用生产 `AnswerQualityEvaluator`，并增加参考概念
覆盖与禁用内容检查：

- concise 简单答案不要求人为扩写；
- balanced 复杂答案必须达到基本结构和深度；
- detailed 复杂答案必须覆盖任务概念，并包含因果、证据、条件、示例、风险或验证；
- Tool Observation、内部 Thought 和无来源实证声明不能成为最终答案。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest \
  tests.test_answer_quality \
  tests.test_answer_evaluation
```

## 4. 环境相关真实案例

以下测试依赖本机模型、凭据或浏览器，不放入默认 CI。执行时记录日期、模型、endpoint、
输入文件、耗时和结果。

| 能力 | 输入 | 通过条件 |
| --- | --- | --- |
| 模型聊天 | 中英文各一个简单问题 | 服务实时发现模型名，返回用户语言的最终答案 |
| Detailed | 一个需要机制、风险、替代和验证的复杂问题 | 内容连贯且覆盖四类要求，无固定字数填充 |
| MinerU OCR | 一张含标题、段落和表格的 PNG/JPEG | Markdown 包含可读标题、正文和表格信息 |
| 扫描 PDF | 至少三页扫描件 | 页码可追踪，末页事实可被检索和引用 |
| Tavily | 一个中文查询和一个英文查询 | 返回相同语言的综合答案和真实 URL |
| Browser | 一个公开动态网页 | 私网阻断仍生效，公开正文可读取 |
| 模型故障 | 断开 endpoint 或返回 503 | 有界时间内失败，Debug 保存分类、尝试和耗时 |

真实模型回答应保存为测试记录，而不是直接加入包含用户文档或凭据的 Git fixture。

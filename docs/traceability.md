# Qwopus-Agent 需求追踪矩阵

> 本文把 [requirements.md](requirements.md) 中的功能需求映射到生产代码和自动化验收。
> “已验证”表示列出的测试覆盖当前主要验收条件，不表示对应能力没有边界。

## 使用规则

1. 修改 `FR-*` 行为时，必须同步更新对应测试和本表。
2. 需求只有在生产模块存在、自动化验收通过且边界已记录时，才能标记为已验证。
3. 真实模型、MinerU OCR 和外部搜索等环境相关能力，应同时保留离线合同测试和人工真实案例。
4. 一个测试可以覆盖多项需求，但每项需求必须至少有一个明确的自动化验收入口。

## 功能需求

| 需求 | 生产责任 | 自动化验收 | 证据状态 | 已知边界 |
| --- | --- | --- | --- | --- |
| FR-LLM-01 | `llm/base.py`、`llm/registry.py`、`llm/openai_compatible.py` | `test_llm_contract.py`、`test_llm_registry.py` | 已验证 | 非 OpenAI-compatible 后端仍需独立 Adapter |
| FR-LLM-02 | `integrations/smolagents_model.py`、`api/model_runtime.py` | `test_smolagents_runtime.py`、`test_model_runtime.py` | 已验证 | 服务端需提供可识别的 `/models` 响应 |
| FR-AGT-01 | `agents/planner.py`、`agents/executor.py` | `test_agents_architecture.py` | 已验证 | Planner 计划质量仍受模型和规则影响 |
| FR-AGT-02 | `agents/router.py`、`agents/multi_agent/`、`services/agent_orchestrator.py` | `test_agents_architecture.py`、`test_multi_agent.py`、`test_agent_orchestrator.py` | 已验证 | 复杂任务增加模型调用和延迟 |
| FR-AGT-03 | `services/answer_pipeline.py`、`services/agent_orchestrator.py`、`integrations/smolagents_runtime.py` | `test_answer_pipeline.py`、`test_agent_orchestrator.py`、`test_smolagents_runtime.py` | 已验证 | 弱模型可能触发一次有界修正 |
| FR-SKL-01 | `skills/registry.py`、`skills/base.py` | `test_skill_registry.py`、`test_skill_discovery.py` | 已验证 | 只加载符合 BaseSkill 工厂合同的模块 |
| FR-SKL-02 | `skills/workflow.py`、`skills/catalog.py`、`services/skill_growth_service.py` | `test_skill_growth.py`、`test_skill_catalog.py`、`test_skill_authoring.py` | 已验证 | Promotion 保持人工确认，不自动部署模型生成代码 |
| FR-DOC-01 | `documents/parser.py`、`documents/mineru.py`、`analysis/document_analysis.py` | `test_document_analysis.py`、`test_graph_multiformat_realcase.py` | 已验证 | PNG/JPEG OCR 需要 MinerU pipeline 和本地模型 |
| FR-DOC-02 | `documents/structure.py`、`documents/chunker.py`、`documents/document_store.py` | `test_document_structure.py`、`test_saved_documents_api.py` | 已验证 | 极差 OCR 标题仍可能降低章节识别质量 |
| FR-XLS-01 | `analysis/workbook_profile.py`、`analysis/excel_processing.py`、`analysis/pandas_sandbox.py` | `test_workbook_profile.py`、`test_document_analysis.py`、`test_pandas_sandbox.py` | 已验证 | 旧 `.xls` 依赖可用的 pandas Excel engine |
| FR-KNW-01 | `memory/minirag.py`、`memory/knowledge_store.py` | `test_minirag.py` | 已验证 | 使用 MiniRAG 的 NanoVectorDB，不调用完整上游 query pipeline |
| FR-KNW-02 | `memory/conversation_knowledge.py`、`api/repository.py` | `test_conversation_knowledge.py`、`test_accounts.py`、`test_saved_documents_api.py` | 已验证 | Global 仅聚合当前账号的聊天知识 |
| FR-KNW-03 | `memory/knowledge_graph.py`、`memory/graph_backend.py`、`memory/graph_extraction.py` | `test_knowledge_graph.py`、`test_graph_backend.py`、`test_graph_extraction.py`、`test_graph_multiformat_realcase.py` | 已验证 | 弱模型抽取失败时退回规则抽取，普通自然语言关系可能缺失 |
| FR-WEB-01 | `integrations/tavily.py`、`skills/web_search.py` | `test_web_search_skill.py`、`test_smolagents_tools.py` | 已验证 | 真实搜索需要有效 Tavily key 和网络 |
| FR-WEB-02 | `integrations/playwright_browser.py`、`skills/browser.py` | `test_playwright_browser.py`、`test_browser_skill.py` | 已验证 | 只读公开 HTTP(S)，不复用个人浏览器会话 |
| FR-RPT-01 | `reports/generator.py`、`reports/charts.py` | `test_reports.py` | 已验证 | PDF 为基础排版，不是出版级编辑器 |
| FR-AUTH-01 | `api/auth.py`、`api/repository.py`、`api/routes/auth.py` | `test_accounts.py`、`test_lan_auth.py` | 已验证 | 本地账号，不含外部身份提供商 |
| FR-AUTH-02 | `api/repository.py`、`api/routes/conversations.py`、`api/routes/documents.py`、`api/routes/reports.py` | `test_accounts.py`、`test_api.py`、`test_saved_documents_api.py` | 已验证 | 共享粒度是完整聊天及其附件 |
| FR-DBG-01 | `api/routes/debug.py`、`utils/debug_store.py`、`frontend/src/components/DebugConsole.tsx` | `test_debug_store.py`、`test_api.py` | 已验证 | 仅回环地址和管理员可访问；不能读取模型隐藏思维链 |
| FR-UI-01 | `frontend/src/App.tsx`、`frontend/src/components/` | TypeScript build、ESLint、Python API tests | 已验证 | 当前没有独立端到端浏览器 CI |

## 非功能需求

| 目标 | 验收证据 | 发布门槛 |
| --- | --- | --- |
| 模型无关 | BaseLLM 合同、Provider Registry、运行时模型发现测试 | LLM 合同和 Registry 测试通过 |
| 数据安全 | 账号 ACL、会话知识隔离、Pandas 沙箱、Browser 私网阻断测试 | 对应安全测试全部通过 |
| 可维护性 | strict mypy、Ruff、模块边界测试 | `ruff` 与 `mypy` 零错误 |
| 可观察性 | ProcessEvent、Debug record、运行日志和阶段指标 | Debug 持久化测试通过 |
| 回答质量 | AnswerContract、AnswerQualityEvaluator、回答基准案例 | concise/balanced/detailed 基准通过 |
| 检索质量 | 来源 recall、precision、首个相关来源排名和隔离回归 | 检索基准不低于数据集阈值 |

## 发布检查

合并到发布分支前至少执行：

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests
.venv/bin/ruff check src tests
MYPY_CACHE_DIR=/tmp/qwopus-mypy .venv/bin/mypy src/qwopus_agent
cd frontend && pnpm run lint && pnpm run build
```

涉及模型、OCR、搜索或浏览器真实环境的改动，还应执行
[evaluation.md](evaluation.md) 中对应的真实案例清单并记录环境与结果。

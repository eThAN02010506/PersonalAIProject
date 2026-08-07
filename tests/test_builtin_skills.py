import asyncio
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from qwopus_agent.skills.base import SkillRequest
from qwopus_agent.skills.document_parser import DocumentParserSkill
from qwopus_agent.skills.excel_analysis import ExcelAnalysisSkill
from qwopus_agent.skills.excel_modeling import ExcelModelingSkill
from qwopus_agent.skills.excel_schema import ExcelSchemaSkill
from qwopus_agent.skills.excel_statistics import ExcelStatisticsSkill
from qwopus_agent.skills.graph_search import create_skill as create_graph_search_skill
from qwopus_agent.skills.rag_search import RagSearchSkill
from tests.minirag_fakes import make_test_minirag


class BuiltinSkillTests(unittest.TestCase):
    def test_document_parser_skill_returns_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.md"
            path.write_text("# Notes\n\nProject Alpha", encoding="utf-8")

            response = asyncio.run(
                DocumentParserSkill().run(
                    SkillRequest(query="parse", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        self.assertIn("Project Alpha", response.content)
        self.assertEqual(response.data["metadata"]["source_type"], "markdown")

    def test_excel_schema_skill_returns_safe_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [10, 20],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelSchemaSkill().run(
                    SkillRequest(query="inspect", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        sheet = response.data["sheets"]["Sheet1"]
        self.assertEqual(sheet["column_names"], ["region", "revenue"])
        self.assertEqual(len(sheet["sample_rows"]), 2)

    def test_excel_analysis_skill_reuses_local_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [10, 20],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelAnalysisSkill().run(
                    SkillRequest(query="分析收入", arguments={"file_path": str(path)})
                )
            )

        self.assertTrue(response.success)
        self.assertIn("Spreadsheet Analysis", response.content)
        self.assertIn("Sheet1_schema", response.data["table_names"])

    def test_excel_statistics_skill_finds_iqr_outlier_with_auditable_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.xlsx"
            pd.DataFrame(
                {
                    "student": ["A", "B", "C", "D", "E"],
                    "score": [10, 11, 12, 13, 100],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="outlier 是什么",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "iqr_outliers",
                            "value_columns": ["score"],
                            "label_columns": ["student"],
                            "threshold": 1.5,
                        },
                    )
                )
            )

        # 原因：通用统计 Skill 必须返回可复核阈值和具体记录，而不是只说“存在异常”。
        # 作用：锁定抽象 outlier 请求的确定性本地计算结果。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["outlier_count"], 1)
        self.assertEqual(response.data["rows"][0]["student"], "E")
        self.assertIn("upper_bound", response.content)
        self.assertIn("| E |", response.content)

    def test_excel_statistics_rejects_explicitly_incompatible_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "households.xlsx"
            pd.DataFrame(
                {
                    "area": ["A", "B"],
                    "Households (thousands)": [10, 20],
                    "Households (per cent)": [25.0, 50.0],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="outlier 是什么",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "iqr_outliers",
                            "value_columns": [
                                "Households (thousands)",
                                "Households (per cent)",
                            ],
                            "label_columns": ["area"],
                        },
                    )
                )
            )

        # 原因：不同量纲直接平均会生成可计算但没有业务含义的“异常值”。
        # 作用：确认错误在确定性 Skill 边界被拒绝，促使 Agent 重选字段。
        self.assertFalse(response.success)
        self.assertIn("incompatible explicit units", response.content)

    def test_excel_statistics_scopes_entities_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gdp.xlsx"
            with pd.ExcelWriter(path) as writer:
                pd.DataFrame(
                    {
                        "name": ["World", "Alpha", "Beta", "Gamma", "Delta"],
                        "code": ["WLD", "AAA", "BBB", "CCC", "DDD"],
                        "2024": [1000, 10, 11, 12, 100],
                    }
                ).to_excel(writer, sheet_name="Data", index=False)
                pd.DataFrame(
                    {
                        "Country Code": ["WLD", "AAA", "BBB", "CCC", "DDD"],
                        "Region": [None, "R1", "R1", "R2", "R2"],
                    }
                ).to_excel(writer, sheet_name="Countries", index=False)

            unscoped_response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="outlier 是什么",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Data",
                            "method": "iqr_outliers",
                            "value_columns": ["2024"],
                            "label_columns": ["name", "code"],
                        },
                    )
                )
            )
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="outlier 是什么",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Data",
                            "method": "iqr_outliers",
                            "value_columns": ["2024"],
                            "label_columns": ["name", "code"],
                            "scope_table_name": "Countries",
                            "scope_data_key": "code",
                            "scope_lookup_key": "Country Code",
                            "scope_required_columns": ["Region"],
                        },
                    )
                )
            )

        # 原因：汇总行会扭曲国家级分布，必须先依据元数据建立同层级比较总体。
        # 作用：确认 Skill 通过通用键匹配排除 World，而不是硬编码特定数据源。
        self.assertTrue(unscoped_response.success)
        self.assertEqual(
            unscoped_response.data["details"]["scope_mode"],
            "auto-detected from workbook classification metadata",
        )
        self.assertNotIn("World", unscoped_response.content)
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["rows_before_scope"], 5)
        self.assertEqual(response.data["details"]["rows_after_scope"], 4)
        self.assertNotIn("World", response.content)

    def test_excel_statistics_supports_common_reviewed_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.xlsx"
            pd.DataFrame(
                {
                    "group": ["A", "A", "B", "B", "B"],
                    "sales": [10, 12, 14, 16, 100],
                    "cost": [5, 6, 7, 8, None],
                }
            ).to_excel(path, index=False)

            base_arguments = {
                "file_path": str(path),
                "table_name": "Sheet1",
                "value_columns": ["sales", "cost"],
            }
            responses = {
                method: asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query=method,
                            arguments={
                                **base_arguments,
                                "method": method,
                                **(
                                    {"group_column": "group"}
                                    if method == "group_summary"
                                    else {}
                                ),
                            },
                        )
                    )
                )
                for method in [
                    "describe",
                    "missing",
                    "group_summary",
                    "correlation",
                    "zscore_outliers",
                ]
            }

        # 原因：Skill 是统计方法库，不应只有 outlier 路径经过执行验证。
        # 作用：锁定描述统计、缺失、分组、相关性和 Z-score 的公共入口均可独立运行。
        self.assertTrue(all(response.success for response in responses.values()))
        self.assertIn("missing_percent", responses["missing"].content)
        self.assertIn("sales_mean", responses["group_summary"].content)
        self.assertIn("Pearson correlation", responses["correlation"].content)
        self.assertIn("absolute Z-score", responses["zscore_outliers"].content)

    def test_excel_statistics_supports_distribution_and_categorical_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "advanced_stats.xlsx"
            pd.DataFrame(
                {
                    "score": [10, 12, 14, 16, 18, 20, 22, 24],
                    "cost": [3, 4, 4, 5, 5, 6, 7, 8],
                    "team": ["A", "A", "A", "A", "B", "B", "B", "B"],
                    "passed": ["yes", "yes", "no", "yes", "yes", "no", "no", "no"],
                }
            ).to_excel(path, index=False)

            base_arguments = {
                "file_path": str(path),
                "table_name": "Sheet1",
                "value_columns": ["score", "cost"],
            }
            responses = {
                "quantiles": asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query="percentiles",
                            arguments={**base_arguments, "method": "quantiles"},
                        )
                    )
                ),
                "covariance": asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query="covariance",
                            arguments={**base_arguments, "method": "covariance"},
                        )
                    )
                ),
                "normality_test": asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query="normality",
                            arguments={**base_arguments, "method": "normality_test"},
                        )
                    )
                ),
                "crosstab": asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query="team by passed",
                            arguments={
                                "file_path": str(path),
                                "table_name": "Sheet1",
                                "method": "crosstab",
                                "category_columns": ["team", "passed"],
                            },
                        )
                    )
                ),
                "chi_square_independence": asyncio.run(
                    ExcelStatisticsSkill().run(
                        SkillRequest(
                            query="team independent from passed",
                            arguments={
                                "file_path": str(path),
                                "table_name": "Sheet1",
                                "method": "chi_square_independence",
                                "category_columns": ["team", "passed"],
                            },
                        )
                    )
                ),
            }

        # 原因：用户的 Excel 问题会从描述统计扩展到分布、矩阵和分类变量关系。
        # 作用：锁定这些通用统计仍由本地确定性 Skill 返回表格，而不是交给模型猜算。
        self.assertTrue(all(response.success for response in responses.values()))
        self.assertIn("p90", responses["quantiles"].content)
        self.assertIn("sample covariance matrix", responses["covariance"].content)
        self.assertIn("decision_at_0.05", responses["normality_test"].content)
        self.assertIn("passed", responses["crosstab"].content)
        self.assertIn("Pearson chi-square", responses["chi_square_independence"].content)

    def test_excel_statistics_describe_and_frequency_match_r_style_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.xlsx"
            pd.DataFrame(
                {
                    "score": [10, 20, 30, 40, None],
                    "team": ["A", "A", "B", None, "B"],
                }
            ).to_excel(path, index=False)
            describe = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="summary",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "describe",
                            "value_columns": ["score"],
                        },
                    )
                )
            )
            frequency = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="table(team)",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "frequency",
                            "value_columns": ["team"],
                        },
                    )
                )
            )

        # 原因：用户需要的是可替代 R summary()/table() 的信息面，而不只是平均值。
        # 作用：锁定离散程度、分布形状、缺失情况与分类频数都由同一 Skill 本地返回。
        self.assertTrue(describe.success)
        summary = describe.data["rows"][0]
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["missing"], 1)
        self.assertAlmostEqual(summary["mean"], 25.0)
        self.assertAlmostEqual(summary["standard_error"], 6.454972, places=5)
        self.assertIn("coefficient_of_variation_percent", summary)
        self.assertIn("excess_kurtosis", summary)
        self.assertTrue(frequency.success)
        counts = {
            row["value"]: row["count"]
            for row in frequency.data["rows"]
        }
        self.assertEqual(counts, {"A": 2, "B": 2, "<missing>": 1})

    def test_excel_statistics_treats_string_null_optional_arguments_as_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.xlsx"
            pd.DataFrame({"score": [10, 20, 30]}).to_excel(path, index=False)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="mean score",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "describe",
                            "value_columns": ["score"],
                            "group_column": "null",
                            "scope_table_name": "none",
                            "scope_data_key": "null",
                            "scope_lookup_key": "none",
                        },
                    )
                )
            )

        # 原因：弱 tool-calling 模型会把 JSON null 错序列化成字符串。
        # 作用：可选参数继续按空值处理，不会被误认为真实列名。
        self.assertTrue(response.success)
        self.assertEqual(response.data["rows"][0]["mean"], 20.0)

    def test_excel_statistics_lookup_answers_single_item_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "character.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Profile"
            sheet.append(["姓名", "Ethan Jiang"])
            sheet.append(["年龄", 20])
            sheet.append([])
            sheet.append(["STR", 40])
            sheet.append(["DEX", 60])
            workbook.save(path)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="年龄是多少",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Profile::key_values",
                            "method": "lookup",
                            "lookup_value": "年龄",
                        },
                    )
                )
            )

        # 原因：用户会问“某一项是什么”，这不是统计汇总，不能靠模型从预览里猜。
        # 作用：锁定 lookup 会从 key_values 结构化帧返回精确字段和值。
        self.assertTrue(response.success)
        rows = response.data["rows"]
        self.assertEqual(rows[0]["key"], "年龄")
        self.assertEqual(rows[0]["value"], 20)

    def test_excel_statistics_lookup_can_match_two_column_header_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "character.xlsx"
            pd.DataFrame(
                {
                    "STR": ["CON", "DEX"],
                    "40": [55, 60],
                }
            ).to_excel(path, index=False)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="STR 是多少",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "lookup",
                            "lookup_value": "STR",
                        },
                    )
                )
            )

        # 原因：二列表单被解析成表格时，首个字段值可能变成列名而不是数据行。
        # 作用：保证 COC 角色卡这类 STR | 40 结构仍能被单项查询命中。
        self.assertTrue(response.success)
        self.assertEqual(response.data["rows"][0]["key"], "STR")
        self.assertEqual(response.data["rows"][0]["value"], "40")
        self.assertIn("column_header_pair", response.content)

    def test_excel_statistics_returns_status_row_when_iqr_outliers_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "clean.xlsx"
            pd.DataFrame({"score": [10, 11, 12, 13, 14]}).to_excel(path, index=False)

            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="outlier",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "iqr_outliers",
                            "value_columns": ["score"],
                        },
                    )
                )
            )

        # 原因：没有异常值也是结论，不能让空结果在最终回答的表格展示中丢失。
        # 作用：保证 Chat/API 可以显示 outlier_count=0 和 IQR 边界。
        self.assertTrue(response.success)
        self.assertEqual(response.data["rows"][0]["outlier_count"], 0)
        self.assertIn("| metric | rule | outlier_count |", response.content)

    def test_excel_statistics_calculates_t_intervals_and_one_sample_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.xlsx"
            pd.DataFrame({"score": [8, 9, 10, 11, 12]}).to_excel(path, index=False)
            base_arguments = {
                "file_path": str(path),
                "table_name": "Sheet1",
                "value_columns": ["score"],
                "confidence_level": 0.95,
            }
            interval = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="95% confidence interval",
                        arguments={
                            **base_arguments,
                            "method": "mean_confidence_interval",
                        },
                    )
                )
            )
            test = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="is the mean different from 9",
                        arguments={
                            **base_arguments,
                            "method": "one_sample_t_test",
                            "hypothesized_mean": 9,
                        },
                    )
                )
            )

        # 原因：弱模型不能可靠计算 t 分位数、p 值和小样本置信区间。
        # 作用：用可手工复核的样本锁定 Skill 的 Student-t 公式和决策语义。
        self.assertTrue(interval.success)
        self.assertAlmostEqual(interval.data["rows"][0]["mean"], 10.0)
        self.assertAlmostEqual(interval.data["rows"][0]["ci_lower"], 8.036757, places=5)
        self.assertAlmostEqual(interval.data["rows"][0]["ci_upper"], 11.963243, places=5)
        self.assertTrue(test.success)
        self.assertAlmostEqual(test.data["rows"][0]["t_statistic"], 1.414214, places=5)
        self.assertAlmostEqual(test.data["rows"][0]["p_value"], 0.2302, places=4)
        self.assertEqual(
            test.data["rows"][0]["decision"],
            "do not reject null hypothesis",
        )

    def test_excel_statistics_calculates_two_sample_welch_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "groups.xlsx"
            pd.DataFrame(
                {
                    "group": ["A", "A", "A", "B", "B", "B"],
                    "score": [10, 12, 14, 20, 22, 24],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="are A and B different",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "two_sample_t_test",
                            "value_columns": ["score"],
                            "group_column": "group",
                            "group_values": ["A", "B"],
                            "confidence_level": 0.95,
                        },
                    )
                )
            )

        # 原因：业务表通常比较两个独立群组，等方差假设不应由模型凭空决定。
        # 作用：默认 Welch 检验并返回差值区间，让 Agent 可以回答显著性与实际方向。
        self.assertTrue(response.success)
        row = response.data["rows"][0]
        self.assertAlmostEqual(row["mean_difference"], -10.0)
        self.assertAlmostEqual(row["t_statistic"], -6.123724, places=5)
        self.assertAlmostEqual(row["p_value"], 0.003602, places=5)
        self.assertEqual(row["decision"], "reject equal-means null hypothesis")
        self.assertIn("| score | A |", response.content)

    def test_excel_statistics_pivots_two_categorical_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sales.xlsx"
            pd.DataFrame(
                {
                    "region": ["East", "West", "East", "West"],
                    "product": ["a", "a", "b", "b"],
                    "revenue": [10, 20, 30, 40],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="pivot revenue by region and product",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "pivot",
                            "row_column": "region",
                            "column_column": "product",
                            "value_column": "revenue",
                            "agg": "sum",
                        },
                    )
                )
            )

        # 原因：pivot 是数据整理请求，结果必须是可复核的聚合单元格而非宽表占位。
        # 作用：锁定 East×a=10、West×b=40 等单元格值来自本地 pivot_table。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["pivot_rows"], 2)
        rows = {row["region"]: row for row in response.data["rows"]}
        self.assertEqual(rows["East"]["a"], 10)
        self.assertEqual(rows["East"]["b"], 30)
        self.assertEqual(rows["West"]["b"], 40)
        self.assertIn("| East |", response.content)

    def test_excel_statistics_extracts_date_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dates.xlsx"
            pd.DataFrame(
                {
                    "order_date": ["2024-01-05", "2024-06-20", "2023-12-01"],
                    "amount": [10, 20, 30],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="extract year and month",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "date_extract",
                            "date_column": "order_date",
                            "parts": ["year", "month"],
                        },
                    )
                )
            )

        # 原因：日期提取需要输出可用的时间分量和解析统计，而非原始字符串。
        # 作用：锁定 year/month 列与 min_year/max_year 来自本地解析。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["min_year"], 2023)
        self.assertEqual(response.data["details"]["max_year"], 2024)
        self.assertEqual(response.data["details"]["parsed_count"], 3)
        rows = response.data["rows"]
        self.assertEqual(rows[0]["order_date_year"], 2024)
        self.assertEqual(rows[0]["order_date_month"], 1)
        self.assertEqual(rows[2]["order_date_year"], 2023)
        self.assertIn("order_date_year", response.content)

    def test_excel_statistics_deduplicates_rows_with_auditable_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dupes.xlsx"
            pd.DataFrame(
                {
                    "id": [1, 1, 2, 2, 3],
                    "name": ["A", "A", "B", "B", "C"],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="remove duplicate rows",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "deduplicate",
                            "columns": ["id", "name"],
                            "keep": "first",
                        },
                    )
                )
            )

        # 原因：去重必须给出保留/删除计数，用户才能核对是否有合法行被误删。
        # 作用：锁定 dropped_count 与 keep 语义来自本地 drop_duplicates。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["total_rows"], 5)
        self.assertEqual(response.data["details"]["kept_rows"], 3)
        self.assertEqual(response.data["details"]["dropped_count"], 2)
        self.assertEqual(len(response.data["rows"]), 3)
        self.assertEqual(response.data["rows"][0]["id"], 1)
        self.assertIn("dropped_count", response.content)

    def test_excel_statistics_ranks_and_bins_numeric_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.xlsx"
            pd.DataFrame(
                {
                    "student": ["A", "B", "C", "D"],
                    "score": [10, 20, 30, 40],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="rank scores",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "rank",
                            "value_column": "score",
                            "rank_method": "rank",
                            "label_columns": ["student"],
                        },
                    )
                )
            )

        # 原因：排名请求输出每行名次，avg 秩对应 R rank() 默认。
        # 作用：锁定 10→1、40→4 的名次来自本地 Series.rank。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["rank_method"], "rank")
        rows = {row["student"]: row["rank"] for row in response.data["rows"]}
        self.assertEqual(rows["A"], 1)
        self.assertEqual(rows["D"], 4)

    def test_excel_statistics_ranks_into_ntile_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scores.xlsx"
            pd.DataFrame(
                {
                    "student": ["A", "B", "C", "D"],
                    "score": [10, 20, 30, 40],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="split scores into quartiles",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "rank",
                            "value_column": "score",
                            "rank_method": "ntile",
                            "into_bins": 2,
                            "label_columns": ["student"],
                        },
                    )
                )
            )

        # 原因：分位分组把连续值映射为 1..N 组，便于分层解读。
        # 作用：锁定两桶时 10/20→1、30/40→2。
        self.assertTrue(response.success)
        rows = {row["student"]: row["rank_ntile"] for row in response.data["rows"]}
        self.assertEqual(rows["A"], 1)
        self.assertEqual(rows["B"], 1)
        self.assertEqual(rows["C"], 2)
        self.assertEqual(rows["D"], 2)

    def test_excel_statistics_rank_ntile_all_equal_degrades_to_single_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "equal.xlsx"
            pd.DataFrame(
                {"student": ["A", "B", "C"], "score": [10, 10, 10]}
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="split into quartiles",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "rank",
                            "value_column": "score",
                            "rank_method": "ntile",
                            "into_bins": 4,
                            "label_columns": ["student"],
                        },
                    )
                )
            )

        # 原因：全相等数据会让 qcut 折叠到单一边界并返回全 NaN。
        # 作用：锁定退化为单一桶（全部 rank 1），而不是输出不可读的 NaN 排名。
        self.assertTrue(response.success)
        self.assertEqual(response.data["details"]["into_bins"], 1)
        for row in response.data["rows"]:
            self.assertEqual(row["rank_ntile"], 1)

    def test_excel_statistics_pivot_count_keeps_non_numeric_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed.xlsx"
            pd.DataFrame(
                {"region": ["a", "a", "a"], "product": ["c1", "c1", "c2"], "value": [5, "N/A", 7]}
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="pivot count",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "pivot",
                            "row_column": "region",
                            "column_column": "product",
                            "value_column": "value",
                            "agg": "count",
                        },
                    )
                )
            )

        # 原因：count 语义是“该组合有多少个非空单元”，非数值文本也是有效计数。
        # 作用：锁定文本单元被计数而非静默当作缺失。
        self.assertTrue(response.success)
        row = response.data["rows"][0]
        self.assertEqual(row["c1"], 2)
        self.assertEqual(row["c2"], 1)

    def test_excel_statistics_kruskal_groups_match_filtered_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kruskal.xlsx"
            pd.DataFrame(
                {"g": ["A", "A", "B", "B", "C", "C"], "v": [float("nan"), float("nan"), 2, 3, 4, 5]}
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelStatisticsSkill().run(
                    SkillRequest(
                        query="kruskal wallis",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "kruskal_wallis",
                            "value_columns": ["v"],
                            "group_column": "g",
                        },
                    )
                )
            )

        # 原因：空数值组（A）被过滤后，报告的组列表必须与实际统计使用的组一致。
        # 作用：锁定 group_count 与 groups 都与过滤后的样本对齐，避免误导性组列表。
        self.assertTrue(response.success)
        row = response.data["rows"][0]
        self.assertEqual(row["group_count"], 2)
        self.assertEqual(row["groups"], "B, C")

    def test_excel_modeling_logistic_rejects_fractional_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fractional.xlsx"
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelModelingSkill().run(
                    SkillRequest(
                        query="logistic regression",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "logistic_regression",
                            "outcome_column": "y",
                            "predictor_columns": ["x"],
                            "confidence_level": 0.95,
                        },
                    )
                )
            )

        # 原因：分数 outcome（0.5/0.7）违反 Logit endog∈(0,1) 约束，会产生伪 R² 为负等垃圾结果。
        # 作用：锁定拒绝而不是返回 success=True 的静默错误结果。
        self.assertFalse(response.success)
        self.assertIn("fractional", response.content)

    def test_excel_modeling_logistic_codes_numeric_1_2_as_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "binary.xlsx"
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "y": [1, 1, 1, 2, 1, 2, 2, 2, 2, 2],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelModelingSkill().run(
                    SkillRequest(
                        query="logistic regression",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "logistic_regression",
                            "outcome_column": "y",
                            "predictor_columns": ["x"],
                            "confidence_level": 0.95,
                        },
                    )
                )
            )

        # 原因：1/2 数值编码是合法的二分类，应映射为 0/1 而非报错。
        # 作用：锁定非 (0,1) 区间的两个 distinct 数值能正常建模。
        self.assertTrue(response.success)
        self.assertIn("Model summary", response.data["tables"])
        summary = response.data["tables"]["Model summary"][0]
        self.assertGreater(summary["pseudo_r_squared"], 0)

    def test_excel_modeling_matches_r_style_linear_model_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regression.xlsx"
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6],
                    "y": [3.1, 4.9, 7.2, 8.8, 11.1, 12.9],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelModelingSkill().run(
                    SkillRequest(
                        query="summary(lm(y ~ x))",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "linear_regression",
                            "outcome_column": "y",
                            "predictor_columns": ["x"],
                            "confidence_level": 0.95,
                        },
                    )
                )
            )

        # 原因：回归 Skill 要提供与 R summary(lm()) 对应的完整模型与系数信息。
        # 作用：锁定 OLS 系数、R²、整体 F 检验和残差摘要均来自本地拟合。
        self.assertTrue(response.success)
        coefficients = response.data["tables"]["Coefficients"]
        slope = next(row for row in coefficients if row["term"] == "x")
        self.assertAlmostEqual(slope["estimate"], 1.977143, places=5)
        self.assertLess(slope["p_value"], 0.001)
        model = response.data["tables"]["Model summary"][0]
        self.assertGreater(model["r_squared"], 0.99)
        self.assertIn("Residual summary", response.data["tables"])
        self.assertIn("f_statistic", response.content)

    def test_excel_modeling_aligns_categorical_predictors_with_excel_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "categorical_regression.xlsx"
            pd.DataFrame(
                {
                    "x": [1, 2, 3, 4, 5, 6],
                    "group": ["A", "A", "A", "B", "B", "B"],
                    "y": [2, 4, 6, 9, 11, 13],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelModelingSkill().run(
                    SkillRequest(
                        query="summary(lm(y ~ x + group))",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "linear_regression",
                            "outcome_column": "y",
                            "predictor_columns": ["x", "group"],
                        },
                    )
                )
            )

        # 原因：Excel 行号从 1 开始，分类 dummy 的默认索引从 0 开始，拼接时可能静默错位。
        # 作用：确认混合预测量模型保留六条观测且能返回分类系数，不产生 SVD 错误。
        self.assertTrue(response.success)
        self.assertEqual(
            response.data["tables"]["Model summary"][0]["observations"],
            6,
        )
        terms = {
            row["term"]
            for row in response.data["tables"]["Coefficients"]
        }
        self.assertIn("group_B", terms)

    def test_excel_modeling_returns_anova_effect_and_tukey_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "anova.xlsx"
            pd.DataFrame(
                {
                    "group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                    "score": [10, 12, 14, 20, 22, 24, 11, 13, 15],
                }
            ).to_excel(path, index=False)
            response = asyncio.run(
                ExcelModelingSkill().run(
                    SkillRequest(
                        query="aov(score ~ group)",
                        arguments={
                            "file_path": str(path),
                            "table_name": "Sheet1",
                            "method": "one_way_anova",
                            "outcome_column": "score",
                            "group_column": "group",
                            "confidence_level": 0.95,
                            "include_posthoc": True,
                        },
                    )
                )
            )

        # 原因：显著 ANOVA 本身不能指出哪些组不同，也不能表达差异的实际大小。
        # 作用：确认 Skill 同时返回组统计、F 检验、效应量、Levene 和 Tukey 调整后比较。
        self.assertTrue(response.success)
        anova = response.data["tables"]["ANOVA"][0]
        self.assertAlmostEqual(anova["f_statistic"], 22.75, places=4)
        self.assertLess(anova["p_value"], 0.01)
        self.assertGreater(response.data["details"]["eta_squared"], 0.8)
        self.assertEqual(len(response.data["tables"]["Tukey HSD"]), 3)
        self.assertIn("levene_p_value", response.content)
        self.assertIn("equal_variance_supported", response.content)
        self.assertIn("Tukey HSD assumes", response.content)

    def test_rag_search_skill_uses_injected_minirag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            minirag = make_test_minirag(Path(tmpdir) / "documents.jsonl")
            minirag.insert("Project Alpha revenue increased.")

            # 原因：Skill 测试需要隔离持久化知识库，不能污染真实 storage/minirag。
            # 作用：通过依赖注入验证 rag_search 只依赖 MiniRAG.insert/search 公共接口。
            response = asyncio.run(
                RagSearchSkill(minirag=minirag).run(SkillRequest(query="revenue"))
            )

        self.assertTrue(response.success)
        self.assertEqual(response.data["results"], ["Project Alpha revenue increased."])

    def test_rag_search_skill_reports_empty_results_as_failure(self) -> None:
        class EmptyMiniRAG:
            def search(self, _query, **_kwargs):
                return []

        response = asyncio.run(
            RagSearchSkill(minirag=EmptyMiniRAG()).run(
                SkillRequest(query="missing evidence")
            )
        )

        # 原因：零命中若标记成功，Executor 或 Tool Agent 会把提示文本当作真实证据。
        # 作用：锁定无结果不会继续进入成功链路，同时保留结构化空列表供调用方判断。
        self.assertFalse(response.success)
        self.assertEqual(response.data["results"], [])
        self.assertEqual(response.content, "No relevant MiniRAG results.")

    def test_discovered_graph_skill_requires_explicit_scope_injection(self) -> None:
        skill = create_graph_search_skill()

        response = asyncio.run(
            skill.run(SkillRequest(query="Company A to Company B"))
        )

        # 原因：自动发现的 Skill 若直接打开默认图谱，会绕过会话隔离和 Global 授权。
        # 作用：锁定占位实例只能声明能力，获准图谱必须由运行时显式注入。
        self.assertFalse(response.success)
        self.assertIn("requires a KnowledgeGraphIndex", response.content)


if __name__ == "__main__":
    unittest.main()

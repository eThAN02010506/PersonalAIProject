"""Recipe contract tests for the grounded report composer.

These tests verify the guarantees of the merged generic recipe:

1. ``DEFAULT_RECIPE`` treats every parser file as one independent source slot
   and composes a deterministic grounded report without any domain vocabulary.
2. Lesson-named files (``第N课`` / ``lesson N``) are ordered by lesson number,
   scripture references are captured and validated against each source's
   allowed verse range, and the report still uses the generic wording.
3. A minimal custom recipe can be built from the generic recipe and drives the
   same shared composer without changing pipeline code.
"""

from __future__ import annotations

import unittest

from qwopus_agent.reports.grounded import (
    DEFAULT_RECIPE,
    _render_deterministic_grounded_report,
)


class LessonNumeralTests(unittest.TestCase):
    def test_chinese_integer_parses_common_lesson_numbers(self) -> None:
        from qwopus_agent.reports.grounded_facts import _chinese_integer

        cases = {
            "一": 1,
            "十": 10,
            "十二": 12,
            "二十": 20,
            "二十五": 25,
            "一百": 100,
            "一百零五": 105,
            "一百一十": 110,
            "二百零三": 203,
            "一千": 1000,
            "一千零一": 1001,
            "一万": 10000,
            "一万二千": 12000,
            "〇": 0,
            "两": 2,
        }
        for numeral, expected in cases.items():
            self.assertEqual(
                _chinese_integer(numeral),
                expected,
                f"{numeral!r} should parse to {expected}",
            )

    def test_chinese_integer_rejects_unrecognized_characters(self) -> None:
        from qwopus_agent.reports.grounded_facts import _chinese_integer

        self.assertIsNone(_chinese_integer("abc"))
        self.assertIsNone(_chinese_integer("一a二"))


def _generic_prompt() -> str:
    return (
        "Please read all files and output the following sections:\n"
        "## 1. Document understanding\n"
        "## 2. Overall writing strategy\n"
        "## 3. Detailed writing outline\n"
        "## 4. Paragraph guidance\n"
        "## 5. Concrete examples\n"
        "## 6. Complete report draft\n"
        "## 7. Draft review\n"
        "## 8. Final checklist"
    )


def _generic_collection() -> str:
    return (
        'QWOPUS_SOURCE_COVERAGE=["report-a.pdf","report-b.pdf"]\n\n'
        "QWOPUS_EXPLICIT_RUBRIC_FOUND=false\n\n"
        "# File: report-a.pdf\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Quarterly sales report\n"
        "- topic_line: Topic: Regional sales growth\n"
        "- quote_line: 引用：第一季度营收 12.4 亿元\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=a-q]:\n"
        "东部区域贡献了过半增长，主要由新品拉动。\n"
        "APPLICATION_EVIDENCE [chunk_id=a-a]:\n"
        "建议把资源向增长最快的中部渠道倾斜。\n\n"
        "# File: report-b.pdf\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Customer churn analysis\n"
        "- topic_line: Topic: Support responsiveness\n"
        "- quote_line: 引用：次日响应率提升 18 个百分点\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=b-q]:\n"
        "响应速度与续费率高度相关，等待超过一天的用户流失率明显更高。\n"
        "APPLICATION_EVIDENCE [chunk_id=b-a]:\n"
        "建议为高价值客户保留人工回访通道。"
    )


def _bible_collection() -> str:
    return (
        'QWOPUS_SOURCE_COVERAGE=["lesson-21.docx","lesson-22.docx"]\n\n'
        "QWOPUS_EXPLICIT_RUBRIC_FOUND=true\n\n"
        "# File: lesson-21.docx\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Lesson 21\n"
        "- topic_line: Title: Active humility\n"
        "- quote_line: 经文：腓立比书2章8节\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=21-q]:\n"
        "材料不是把卑微解释为自我贬低，而是主动走向低处。\n"
        "APPLICATION_EVIDENCE [chunk_id=21-a]:\n"
        "请辨认一次表面让步、内里压抑的关系处境，并说明动机。\n\n"
        "# File: lesson-22.docx\n"
        "SOURCE_FACTS:\n"
        "- document_heading: Lesson 22\n"
        "- topic_line: Title: Free obedience\n"
        "- quote_line: 经文：腓立比书2章8节\n"
        "QUERY_RELEVANT_EVIDENCE [chunk_id=22-q]:\n"
        "材料说明顺服不是盲从，而是理解之后仍然选择信靠和回应。\n"
        "APPLICATION_EVIDENCE [chunk_id=22-a]:\n"
        "请分析一次明知更正确却不愿行动的选择及其后果。"
    )


def _requested_sections(prompt: str) -> dict[int, str]:
    from qwopus_agent.reports.grounded_facts import _requested_numbered_sections

    return _requested_numbered_sections(prompt)


def _render(requested: dict[int, str], files: list[str], evidence: str, recipe) -> str:
    from qwopus_agent.reports.grounded import _validated_grounded_collection

    specs = _validated_grounded_collection(
        file_names=files,
        collection_evidence=evidence,
        recipe=recipe,
    )
    return _render_deterministic_grounded_report(
        requested=requested,
        file_names=files,
        collection_evidence=evidence,
        source_specs=specs,
        recipe=recipe,
    )


class DefaultRecipeTests(unittest.TestCase):
    def test_generic_recipe_composes_report_for_arbitrary_pdfs(self) -> None:
        files = ["report-a.pdf", "report-b.pdf"]
        evidence = _generic_collection()
        requested = _requested_sections(_generic_prompt())
        report = _render(requested, files, evidence, DEFAULT_RECIPE)

        # 通用 recipe 不引入任何圣经措辞。
        self.assertNotIn("经文", report)
        self.assertNotIn("生活展开", report)
        self.assertNotIn("本课", report)
        # 每个来源都作为一个独立槽位出现，并按文件顺序推进。
        self.assertIn("report-a", report)
        self.assertIn("report-b", report)
        self.assertIn("### 引言", report)
        self.assertIn("### 综合结论", report)
        # 无显式 rubric 时不得虚构分值。
        self.assertNotIn("总分", report)

    def test_generic_recipe_uses_quote_line_label(self) -> None:
        files = ["report-a.pdf", "report-b.pdf"]
        evidence = _generic_collection()
        specs = DEFAULT_RECIPE.build_grounding_specs(files, evidence)
        self.assertEqual(len(specs), 2)
        self.assertIn("第一季度营收", specs[0].passage_lines[0])
        self.assertEqual(specs[0].allowed_references, frozenset())


class MergedRecipeLessonTests(unittest.TestCase):
    def test_generic_recipe_orders_lessons_and_validates_scripture(self) -> None:
        files = ["lesson-22.docx", "lesson-21.docx"]
        evidence = _bible_collection()
        specs = DEFAULT_RECIPE.build_grounding_specs(files, evidence)
        # 通用 recipe 也按课号排序，忽略输入顺序。
        self.assertEqual([spec.number for spec in specs], [21, 22])
        # allowed_references 只含来源经文的 key。
        self.assertTrue(specs[0].allowed_references)
        book, numbers = next(iter(specs[0].allowed_references))
        self.assertEqual(book, "腓立比书")

    def test_generic_recipe_rejects_out_of_range_scripture(self) -> None:
        from qwopus_agent.reports.grounded_facts import (
            _scripture_reference_is_supported,
            _scripture_reference_key,
        )

        allowed = frozenset({("腓立比书", (2, 8))})
        self.assertTrue(
            _scripture_reference_is_supported(
                _scripture_reference_key("腓立比书2章8节"),
                allowed,
            )
        )
        self.assertFalse(
            _scripture_reference_is_supported(
                _scripture_reference_key("腓立比书2章9节"),
                allowed,
            )
        )

    def test_generic_recipe_top_level_check_rejects_unsourced_scripture(self) -> None:
        """The report-level reference check reads the recipe quote fact key."""
        from qwopus_agent.reports.contract import _report_quality_issues

        files = ["lesson-21.docx"]
        evidence = _bible_collection()
        requested = {1: "文档理解"}
        answer = (
            "## 1. 文档理解\n"
            "材料围绕腓立比书2章9节展开，并解释该经文如何推动生活应用。"
        )
        issues = _report_quality_issues(
            answer=answer,
            requested=requested,
            file_names=files,
            user_question="请逐一阅读所有文件并按以下结构输出。",
            collection_evidence=evidence,
            recipe=DEFAULT_RECIPE,
        )
        messages = issues.get(1, [])
        self.assertTrue(
            any(
                "remove or correct references absent from SOURCE_FACTS"
                in message
                for message in messages
            ),
            f"expected an unsourced-reference issue, got: {messages}",
        )

    def test_generic_recipe_composes_lesson_report_with_generic_wording(self) -> None:
        files = ["lesson-21.docx", "lesson-22.docx"]
        evidence = _bible_collection()
        requested = _requested_sections(_generic_prompt())
        report = _render(requested, files, evidence, DEFAULT_RECIPE)
        self.assertIn("### lesson-21", report)
        self.assertIn("### lesson-22", report)
        # 引用以 passage_lines 形式出现，不再是“材料未单列引用”。
        self.assertIn("腓立比书2章8节", report)
        # 措辞保持通用，不引入圣经专属标签。
        self.assertNotIn("经文与主题", report)
        self.assertIn("表面让步、内里压抑", report)


class CustomRecipeTests(unittest.TestCase):
    def test_custom_recipe_reuses_shared_composer(self) -> None:
        """A minimal recipe overriding one label still drives the shared pipeline."""
        import dataclasses

        custom = dataclasses.replace(
            DEFAULT_RECIPE,
            source_fact_labels=dataclasses.replace(
                DEFAULT_RECIPE.source_fact_labels,
                topic_line=("主题", "subject"),
            ),
        )
        files = ["report-a.pdf", "report-b.pdf"]
        evidence = _generic_collection()
        requested = _requested_sections(_generic_prompt())
        report = _render(requested, files, evidence, custom)
        self.assertIn("report-a", report)
        self.assertIn("### 引言", report)

    def test_recipe_is_frozen_and_shareable(self) -> None:
        import dataclasses

        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_RECIPE.composer_thresholds.min_parser_files = 99


class GroundedComposerGuardTests(unittest.TestCase):
    def test_composer_thresholds_gate_selection(self) -> None:
        import dataclasses

        from qwopus_agent.reports.grounded import (
            should_use_grounded_report_composer,
        )

        strict_recipe = dataclasses.replace(
            DEFAULT_RECIPE,
            composer_thresholds=dataclasses.replace(
                DEFAULT_RECIPE.composer_thresholds,
                min_sections=100,
            ),
        )
        self.assertFalse(
            should_use_grounded_report_composer(
                file_names=["report-a.pdf", "report-b.pdf"],
                spreadsheet_names=[],
                user_question=_generic_prompt(),
                has_collection_summary=True,
                recipe=strict_recipe,
            )
        )


if __name__ == "__main__":
    unittest.main()

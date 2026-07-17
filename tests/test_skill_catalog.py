import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwopus_agent.skills import SkillCatalog, SkillManifest


class SkillCatalogTests(unittest.TestCase):
    def test_skill_catalog_persists_versioned_manifests(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "catalog.json"
            catalog = SkillCatalog(storage_path=catalog_path)
            manifest = SkillManifest(
                name="excel_analysis",
                version="0.1.0",
                description="Analyze spreadsheet summaries.",
                module_path="qwopus_agent.skills.excel_analysis",
            )

            # 原因：技能复用/成长系统需要记录版本，而不是只扫描当前文件夹。
            # 作用：验证 manifest 可持久化，并能由新 catalog 实例读取。
            catalog.register(manifest)
            reloaded = SkillCatalog(storage_path=catalog_path)

            self.assertEqual(reloaded.latest("excel_analysis"), manifest)
            self.assertEqual(len(reloaded.list()), 1)

    def test_skill_catalog_replaces_same_name_and_version(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = SkillCatalog(storage_path=Path(tmpdir) / "catalog.json")
            catalog.register(SkillManifest("demo", "1.0.0", "old", "old.module"))
            catalog.register(SkillManifest("demo", "1.0.0", "new", "new.module"))

            latest = catalog.latest("demo")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.description, "new")
            self.assertEqual(len(catalog.list()), 1)


if __name__ == "__main__":
    unittest.main()

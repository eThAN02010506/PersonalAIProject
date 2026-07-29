import tempfile
import unittest
from pathlib import Path

from qwopus_agent.documents.local_folder import (
    MAX_LOCAL_FOLDER_DEPTH,
    LocalFolderError,
    resolve_selected_files,
    scan_local_folder,
)


class LocalFolderTests(unittest.TestCase):
    def test_scan_returns_supported_files_as_a_filtered_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chapter = root / "chapter"
            chapter.mkdir()
            (root / "notes.md").write_text("# Notes", encoding="utf-8")
            (chapter / "table.xlsx").write_bytes(b"sheet")
            (chapter / "ignore.zip").write_bytes(b"archive")
            (root / ".hidden.txt").write_text("hidden", encoding="utf-8")

            result = scan_local_folder(root)

            self.assertEqual(result.file_count, 2)
            self.assertEqual(
                [node.name for node in result.tree.children],
                ["chapter", "notes.md"],
            )
            self.assertEqual(result.tree.children[0].children[0].name, "table.xlsx")

    def test_scan_skips_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target.txt"
            target.write_text("target", encoding="utf-8")
            (root / "linked.txt").symlink_to(target)

            result = scan_local_folder(root)

            self.assertEqual(result.file_count, 1)
            self.assertEqual(result.tree.children[0].name, "target.txt")

    def test_selection_rejects_traversal_and_symbolic_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            selected = root / "selected.md"
            selected.write_text("# Selected", encoding="utf-8")
            linked = root / "linked.md"
            linked.symlink_to(selected)

            self.assertEqual(
                resolve_selected_files(root, ["selected.md"]),
                (selected.resolve(),),
            )
            with self.assertRaises(LocalFolderError):
                resolve_selected_files(root, ["../outside.md"])
            with self.assertRaises(LocalFolderError):
                resolve_selected_files(root, ["linked.md"])

    def test_scan_rejects_excessive_directory_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            current = root
            for index in range(MAX_LOCAL_FOLDER_DEPTH + 2):
                current = current / f"level-{index}"
                current.mkdir()
            (current / "deep.txt").write_text("deep", encoding="utf-8")

            # 原因：受支持文件数量很少时，旧扫描仍可能递归到 Python 栈限制。
            # 作用：目录深度在进入不安全递归范围前返回稳定的业务错误。
            with self.assertRaisesRegex(LocalFolderError, "nesting exceeds"):
                scan_local_folder(root)


if __name__ == "__main__":
    unittest.main()

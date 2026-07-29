import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qwopus_agent.documents import DocumentStore
from qwopus_agent.documents.storage import save_uploaded_bytes
from qwopus_agent.integrations.smolagents_runtime import SmolagentsModelSettings
from qwopus_agent.memory import (
    ConversationKnowledgeManager,
    conversation_knowledge_path,
)
from qwopus_agent.services.analysis_service import (
    UploadedFileInput,
    analyze_uploaded_files,
)
from tests.minirag_fakes import make_test_minirag


class ConversationKnowledgeTests(unittest.TestCase):
    def test_two_conversations_persist_and_search_only_their_own_documents(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "knowledge"
            manager = ConversationKnowledgeManager(root=root, factory=make_test_minirag)

            with manager.lease("conversation-a") as first:
                first.insert("# File: alpha.txt\n\nalpha-exclusive-9f31 project fact")
            with manager.lease("conversation-b") as second:
                second.insert("# File: beta.txt\n\nbeta-exclusive-7c42 project fact")

            # 原因：conversation_id 若只写在 API 层，底层仍可能复用同一个内存或文件索引。
            # 作用：用互斥事实证明向量记录、图谱文件和重启加载都遵守相同隔离路径。
            self.assertTrue(
                manager.get("conversation-a").search(
                    "alpha exclusive 9f31",
                    min_relevance=0.25,
                )
            )
            second_results = manager.get("conversation-b").search(
                "alpha exclusive 9f31",
                min_relevance=0.25,
            )
            self.assertNotIn("alpha-exclusive-9f31", "\n".join(second_results))
            self.assertTrue(
                all("Source: beta.txt" in result for result in second_results)
            )
            self.assertNotEqual(
                manager.storage_path("conversation-a"),
                manager.storage_path("conversation-b"),
            )

            reloaded = ConversationKnowledgeManager(root=root, factory=make_test_minirag)
            self.assertTrue(
                reloaded.get("conversation-b").search(
                    "beta exclusive 7c42",
                    min_relevance=0.25,
                )
            )

    def test_manager_reuses_one_instance_and_deletes_only_one_scope(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "knowledge"
            manager = ConversationKnowledgeManager(root=root, factory=make_test_minirag)
            first_private = manager.get("conversation-a")
            with manager.lease("conversation-a") as first:
                first.insert("# File: alpha.txt\n\nalpha retained only in this scope")
            with manager.lease("conversation-b") as second:
                second.insert("# File: beta.txt\n\nbeta remains after alpha deletion")

            self.assertIs(first_private, manager.get("conversation-a"))
            manager.delete("conversation-a")

            self.assertFalse((root / "conversation-a").exists())
            self.assertTrue((root / "conversation-b" / "documents.jsonl").exists())
            # 原因：已通过 API 校验的旧上传请求可能晚于删除操作进入知识管理器。
            # 作用：删除 tombstone 阻止旧请求重新创建不可见的会话目录。
            with self.assertRaisesRegex(RuntimeError, "was deleted"):
                manager.get("conversation-a")
            with (
                self.assertRaisesRegex(RuntimeError, "was deleted"),
                manager.lease("conversation-a"),
            ):
                self.fail("Deleted conversation must not grant a knowledge lease.")
            global_memory = make_test_minirag(manager.global_storage_path)
            global_results = "\n".join(
                global_memory.search("beta remains", min_relevance=0.25)
            )
            # 原因：删除私库但保留其全局镜像会让 Global 权限看到已删除聊天的数据。
            # 作用：证明删除只清除 conversation-a 的镜像，conversation-b 仍可全局检索。
            self.assertNotIn("alpha retained", global_results)
            self.assertIn("beta remains", global_results)

    def test_global_aggregate_connects_evidence_from_separate_conversations(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "knowledge"
            manager = ConversationKnowledgeManager(root=root, factory=make_test_minirag)

            with manager.lease("conversation-a") as first:
                first.insert(
                    "# File: ownership.md\n\n"
                    "[[Aurora Holdings|Company]] -[owns]-> [[Lumen Labs|Company]]"
                )
            with manager.lease("conversation-b") as second:
                second.insert(
                    "# File: funding.md\n\n"
                    "[[Lumen Labs|Company]] -[funds]-> [[Project Nova|Project]]"
                )

            private_paths = manager.get("conversation-a").graph_index.paths_between(
                "Aurora Holdings",
                "Project Nova",
            )
            global_memory = make_test_minirag(manager.global_storage_path)
            global_paths = global_memory.graph_index.paths_between(
                "Aurora Holdings",
                "Project Nova",
            )

            # 原因：逐库查询无法连接分散在两个聊天中的边，不符合显式 Global 的跨对话语义。
            # 作用：证明私库保持隔离，而授权用聚合图可以返回跨会话两跳路径。
            self.assertEqual(private_paths, [])
            self.assertTrue(global_paths)
            self.assertEqual(
                global_paths[0].entity_names,
                ("Aurora Holdings", "Lumen Labs", "Project Nova"),
            )
            self.assertEqual(
                {item.source for item in global_paths[0].evidence},
                {
                    "conversation:conversation-a/ownership.md",
                    "conversation:conversation-b/funding.md",
                },
            )

    def test_account_global_aggregates_do_not_share_sources(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "knowledge"
            manager = ConversationKnowledgeManager(root=root, factory=make_test_minirag)

            with manager.lease("owner-chat", global_scope="owner-account") as owner:
                owner.insert("# File: owner.md\n\nowner-global-exclusive-31")
            with manager.lease("member-chat", global_scope="member-account") as member:
                member.insert("# File: member.md\n\nmember-global-exclusive-42")

            owner_global = make_test_minirag(
                manager.global_storage_path_for("owner-account")
            )
            member_global = make_test_minirag(
                manager.global_storage_path_for("member-account")
            )
            owner_results = "\n".join(
                owner_global.search("global exclusive", min_relevance=0.25)
            )
            member_results = "\n".join(
                member_global.search("global exclusive", min_relevance=0.25)
            )

            # 原因：会话私库隔离后若仍共用一个 Global 文件，开关会绕过账号 ACL。
            # 作用：证明每个账号的跨聊天聚合拥有独立 JSONL、向量索引与图谱目录。
            self.assertIn("owner-global-exclusive-31", owner_results)
            self.assertNotIn("member-global-exclusive-42", owner_results)
            self.assertIn("member-global-exclusive-42", member_results)
            self.assertNotIn("owner-global-exclusive-31", member_results)

    def test_scope_path_rejects_directory_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported characters"):
            conversation_knowledge_path("../another-conversation")

    def test_real_upload_pipeline_keeps_two_chat_indexes_separate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = ConversationKnowledgeManager(
                root=root / "knowledge",
                factory=make_test_minirag,
            )
            settings = SmolagentsModelSettings(
                model_id="offline-realcase",
                base_url="http://127.0.0.1:1/v1",
            )
            document_store = DocumentStore(root / "documents")

            def save_in_test_scope(filename: str, content: bytes):
                return save_uploaded_bytes(
                    filename,
                    content,
                    upload_dir=root / "uploads",
                )

            # 原因：真实链路要覆盖保存、解析、结构化和入库，但不应污染项目 storage 或等待模型。
            # 作用：只替换外部路径和在线模型发现，其余代码走生产分析服务与真实 MiniRAG。
            with (
                patch(
                    "qwopus_agent.services.analysis_service.save_uploaded_bytes",
                    side_effect=save_in_test_scope,
                ),
                patch(
                    "qwopus_agent.services.analysis_service.resolve_model_settings",
                    side_effect=lambda current: current,
                ),
            ):
                with manager.lease("conversation-a") as first:
                    first_outcome = analyze_uploaded_files(
                        [UploadedFileInput("alpha.txt", b"alpha-private-31 launch plan")],
                        "",
                        settings,
                        first,
                        document_store=document_store,
                    )
                with manager.lease("conversation-b") as second:
                    second_outcome = analyze_uploaded_files(
                        [UploadedFileInput("beta.txt", b"beta-private-42 budget plan")],
                        "",
                        settings,
                        second,
                        document_store=document_store,
                    )

            self.assertTrue(first_outcome.result.metadata["minirag_inserted"])
            self.assertTrue(second_outcome.result.metadata["minirag_inserted"])
            first_results = "\n".join(
                manager.get("conversation-a").search(
                    "alpha private 31",
                    min_relevance=0.25,
                )
            )
            second_results = "\n".join(
                manager.get("conversation-b").search(
                    "alpha private 31",
                    min_relevance=0.25,
                )
            )
            self.assertIn("alpha-private-31", first_results)
            self.assertNotIn("alpha-private-31", second_results)
            global_results = "\n".join(
                make_test_minirag(manager.global_storage_path).search(
                    "alpha private 31 beta private 42",
                    min_relevance=0.25,
                )
            )
            # 原因：只测试直接 insert 不能证明生产上传服务确实经过镜像入口。
            # 作用：真实保存、解析、结构化链路完成后，两个聊天的内容均进入授权用聚合库。
            self.assertIn("alpha-private-31", global_results)
            self.assertIn("beta-private-42", global_results)


if __name__ == "__main__":
    unittest.main()

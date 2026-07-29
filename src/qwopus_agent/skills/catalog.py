"""Persistent version catalog for reusable and learned skills."""

from __future__ import annotations

import json
import re
from builtins import list as list_type
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

DEFAULT_SKILL_CATALOG_PATH = Path("storage/skills/catalog.json")
SKILL_STATUSES = {"candidate", "active", "archived", "rejected"}
SkillStatus = Literal["candidate", "active", "archived", "rejected"]
SEMANTIC_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class SkillManifest:
    """Versioned metadata for one reusable skill."""

    name: str
    version: str
    description: str
    module_path: str
    checksum: str = ""
    status: SkillStatus = "active"
    spec_path: str | None = None
    created_at: str = ""
    source_run_id: str | None = None
    source_signature: str | None = None


@dataclass
class SkillCatalog:
    """Persist, version, and activate reusable skill manifests."""

    storage_path: Path = DEFAULT_SKILL_CATALOG_PATH

    def register(self, manifest: SkillManifest) -> None:
        """Register or replace one exact skill version."""
        _version_key(manifest.version)
        if manifest.status not in SKILL_STATUSES:
            raise ValueError(f"Unknown skill status: {manifest.status}")
        manifests = self.list()
        key = (manifest.name, manifest.version)
        filtered = [
            current
            for current in manifests
            if (current.name, current.version) != key
        ]
        filtered.append(manifest)
        # 原因：成长 Skill 必须保留候选、激活和归档状态，不能只依赖运行时对象。
        # 作用：持久化可审计版本记录，并使用语义版本顺序稳定输出。
        self._write(sorted(filtered, key=lambda item: (item.name, _version_key(item.version))))

    def list(self) -> list_type[SkillManifest]:
        """List all registered skill manifests."""
        if not self.storage_path.exists():
            return []
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [SkillManifest(**item) for item in payload.get("skills", [])]

    def latest(
        self,
        name: str,
        status: SkillStatus | None = None,
    ) -> SkillManifest | None:
        """Return the newest matching semantic version."""
        candidates = [
            manifest
            for manifest in self.list()
            if manifest.name == name and (status is None or manifest.status == status)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: _version_key(item.version))

    def active(self, name: str) -> SkillManifest | None:
        """Return the active version for one skill."""
        return self.latest(name, status="active")

    def deployed(self) -> list_type[SkillManifest]:
        """Return all active workflow manifests."""
        return [
            manifest
            for manifest in self.list()
            if manifest.status == "active" and manifest.spec_path is not None
        ]

    def next_patch_version(self, name: str) -> str:
        """Return the next patch version, starting learned skills at 0.1.0."""
        latest = self.latest(name)
        if latest is None:
            return "0.1.0"
        major, minor, patch = _version_key(latest.version)
        return f"{major}.{minor}.{patch + 1}"

    def activate(self, name: str, version: str) -> SkillManifest:
        """Activate one version and archive older active versions of the same skill."""
        manifests = self.list()
        target: SkillManifest | None = None
        updated: list_type[SkillManifest] = []
        for manifest in manifests:
            if manifest.name == name and manifest.version == version:
                target = replace(manifest, status="active")
                updated.append(target)
            elif manifest.name == name and manifest.status == "active":
                updated.append(replace(manifest, status="archived"))
            else:
                updated.append(manifest)
        if target is None:
            raise KeyError(f"Unknown skill version: {name}@{version}")
        self._write(sorted(updated, key=lambda item: (item.name, _version_key(item.version))))
        return target

    def reject(self, name: str, version: str) -> SkillManifest:
        """Reject one candidate while preserving its immutable audit record."""
        manifests = self.list()
        target: SkillManifest | None = None
        updated: list_type[SkillManifest] = []
        for manifest in manifests:
            if manifest.name == name and manifest.version == version:
                if manifest.status != "candidate":
                    raise ValueError("Only candidate Skill versions can be rejected.")
                target = replace(manifest, status="rejected")
                updated.append(target)
            else:
                updated.append(manifest)
        if target is None:
            raise KeyError(f"Unknown skill version: {name}@{version}")
        # 原因：删除拒绝记录会失去“为什么这个版本未部署”的审计链。
        # 作用：只改变状态并原子写回，spec 和来源 run id 继续可供 Debug 检查。
        self._write(sorted(updated, key=lambda item: (item.name, _version_key(item.version))))
        return target

    def _write(self, manifests: list_type[SkillManifest]) -> None:
        """Persist manifests atomically to avoid a partial catalog."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [asdict(manifest) for manifest in manifests]}
        temporary_path = self.storage_path.with_suffix(f"{self.storage_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.storage_path)


def _version_key(version: str) -> tuple[int, int, int]:
    match = SEMANTIC_VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"Skill version must use MAJOR.MINOR.PATCH: {version}")
    return tuple(int(value) for value in match.groups())  # type: ignore[return-value]

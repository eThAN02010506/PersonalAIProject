"""Versioned skill catalog for skill reuse and growth."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_SKILL_CATALOG_PATH = Path("storage/skills/catalog.json")


@dataclass(frozen=True)
class SkillManifest:
    """Versioned metadata for one reusable skill."""

    name: str

    version: str

    description: str

    module_path: str


@dataclass
class SkillCatalog:
    """Persist and query reusable skill manifests."""

    storage_path: Path = DEFAULT_SKILL_CATALOG_PATH

    def register(self, manifest: SkillManifest) -> None:
        """Register or replace one skill version."""
        manifests = self.list()
        key = (manifest.name, manifest.version)
        filtered = [
            current
            for current in manifests
            if (current.name, current.version) != key
        ]
        filtered.append(manifest)
        # 原因：技能成长系统需要先有可审计版本记录，不能只依赖 Python 文件存在。
        # 作用：把 skill name/version/module_path 写入本地 catalog，后续可做复用和升级。
        self._write(sorted(filtered, key=lambda item: (item.name, item.version)))

    def list(self) -> list[SkillManifest]:
        """List all registered skill manifests."""
        if not self.storage_path.exists():
            return []
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        return [
            SkillManifest(**item)
            for item in payload.get("skills", [])
        ]

    def latest(self, name: str) -> SkillManifest | None:
        """Return the latest registered version for one skill name."""
        candidates = [manifest for manifest in self.list() if manifest.name == name]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.version)[-1]

    def _write(self, manifests: list[SkillManifest]) -> None:
        """Persist manifests to JSON."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"skills": [asdict(manifest) for manifest in manifests]}
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

"""Host-local Tavily credential persistence and resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import dotenv_values

DEFAULT_TAVILY_KEY_PATH = Path("storage/secrets/tavily.key")
DEFAULT_LEGACY_ENV_PATH = Path(".env.local")


class TavilyCredentialError(ValueError):
    """Raised when a Tavily credential cannot be validated or persisted."""


@dataclass(frozen=True)
class TavilyCredentialStatus:
    """Safe credential metadata that never contains the complete key."""

    configured: bool
    source: Literal["managed", "legacy_local", "environment", "none"]
    masked_key: str | None = None


@dataclass(frozen=True)
class TavilyCredentialStore:
    """Persist one host-wide key outside Git-tracked configuration."""

    path: Path = field(default_factory=lambda: DEFAULT_TAVILY_KEY_PATH)
    legacy_env_path: Path = field(default_factory=lambda: DEFAULT_LEGACY_ENV_PATH)

    def resolve(self, explicit_api_key: str | None = None) -> str:
        """Resolve an explicit, admin-managed, legacy, or environment key."""
        if explicit_api_key and explicit_api_key.strip():
            return explicit_api_key.strip()
        managed = self._managed_key()
        if managed:
            return managed
        legacy = dotenv_values(self.legacy_env_path).get("TAVILY_API_KEY")
        if isinstance(legacy, str) and legacy.strip():
            return legacy.strip()
        return (os.getenv("TAVILY_API_KEY") or "").strip()

    def status(self) -> TavilyCredentialStatus:
        """Return only source and masked state for the settings API."""
        managed = self._managed_key()
        if managed:
            return TavilyCredentialStatus(True, "managed", _mask_key(managed))
        legacy = dotenv_values(self.legacy_env_path).get("TAVILY_API_KEY")
        if isinstance(legacy, str) and legacy.strip():
            return TavilyCredentialStatus(
                True,
                "legacy_local",
                _mask_key(legacy.strip()),
            )
        environment = (os.getenv("TAVILY_API_KEY") or "").strip()
        if environment:
            return TavilyCredentialStatus(
                True,
                "environment",
                _mask_key(environment),
            )
        return TavilyCredentialStatus(False, "none")

    def save(self, api_key: str) -> TavilyCredentialStatus:
        """Atomically replace the admin-managed key with owner-only permissions."""
        normalized = _validate_key(api_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            # 原因：并发保存或进程中断不能留下半个 Key，普通文件默认权限也可能过宽。
            # 作用：以 0600 创建同目录临时文件，再原子替换成正式主机凭据。
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(normalized)
                handle.write("\n")
            temporary_path.replace(self.path)
            os.chmod(self.path, 0o600)
        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            raise TavilyCredentialError("Could not save the Tavily API key.") from exc
        return self.status()

    def delete(self) -> TavilyCredentialStatus:
        """Delete only the admin-managed key and preserve deployment environment values."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise TavilyCredentialError("Could not remove the Tavily API key.") from exc
        return self.status()

    def _managed_key(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TavilyCredentialError("Could not read the Tavily API key.") from exc


def resolve_tavily_api_key(explicit_api_key: str | None = None) -> str:
    """Resolve the process-wide Tavily credential at Tool call time."""
    return TavilyCredentialStore().resolve(explicit_api_key)


def _validate_key(api_key: str) -> str:
    normalized = api_key.strip()
    if not 8 <= len(normalized) <= 512:
        raise TavilyCredentialError("Tavily API key must contain 8 to 512 characters.")
    if any(character.isspace() or ord(character) < 32 for character in normalized):
        raise TavilyCredentialError("Tavily API key must not contain whitespace.")
    return normalized


def _mask_key(api_key: str) -> str:
    """Show enough identity for rotation without exposing a usable credential."""
    if len(api_key) <= 9:
        return "•" * len(api_key)
    return f"{api_key[:5]}{'•' * 8}{api_key[-4:]}"

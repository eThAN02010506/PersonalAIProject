"""Restricted Playwright browser provider for Agent read operations."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit


@dataclass
class PlaywrightBrowserProvider:
    """Render public web pages in one isolated, non-persistent browser context."""

    timeout_ms: int = 15_000
    max_text_chars: int = 20_000
    allow_private_hosts: bool = False
    browser_channel: str | None = "chrome"
    _last_snapshot: str = field(default="", init=False, repr=False)

    def open(self, url: str) -> str:
        """Open one public HTTP(S) page and return its rendered text snapshot."""
        normalized_url = _validate_url(
            url,
            allow_private_hosts=self.allow_private_hosts,
        )
        try:
            from playwright.sync_api import Error as PlaywrightError
            from playwright.sync_api import sync_playwright
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Playwright browser support is not installed. "
                "Install qwopus-agent[browser]."
            ) from exc

        try:
            with sync_playwright() as playwright:
                browser = _launch_browser(playwright, self.browser_channel)
                try:
                    # 原因：Agent 页面不应复用用户 Cookie、本地存储或长期浏览器配置。
                    # 作用：每次 open 都使用无痕 Context，并禁止下载和 Service Worker 绕过路由。
                    context = browser.new_context(
                        accept_downloads=False,
                        service_workers="block",
                    )
                    try:
                        context.route(
                            "**/*",
                            lambda route: _route_request(
                                route,
                                allow_private_hosts=self.allow_private_hosts,
                            ),
                        )
                        page = context.new_page()
                        page.goto(
                            normalized_url,
                            wait_until="domcontentloaded",
                            timeout=self.timeout_ms,
                        )
                        title = page.title()
                        body_text = page.locator("body").inner_text(timeout=self.timeout_ms)
                        current_url = page.url
                    finally:
                        context.close()
                finally:
                    browser.close()
        except PlaywrightError as exc:
            raise RuntimeError(f"Browser navigation failed: {exc}") from exc

        bounded_text = body_text[: self.max_text_chars]
        if len(body_text) > self.max_text_chars:
            bounded_text += "\n\n[Browser snapshot truncated.]"
        self._last_snapshot = (
            f"# Browser Page\n\n- URL: {current_url}\n- Title: {title}\n\n{bounded_text}"
        )
        return self._last_snapshot

    def snapshot(self) -> str:
        """Return the most recent rendered snapshot without reopening the page."""
        if not self._last_snapshot:
            raise RuntimeError("No browser page has been opened yet.")
        return self._last_snapshot


def _launch_browser(playwright: Any, channel: str | None) -> Any:
    """Launch system Chrome when available, then fall back to bundled Chromium."""
    if channel:
        try:
            return playwright.chromium.launch(channel=channel, headless=True)
        except Exception:  # noqa: BLE001 - fallback reports the final Playwright error.
            pass
    return playwright.chromium.launch(headless=True)


def _route_request(route: Any, *, allow_private_hosts: bool) -> None:
    """Abort subrequests that could cross the public-network boundary."""
    try:
        _validate_url(route.request.url, allow_private_hosts=allow_private_hosts)
    except (OSError, ValueError):
        route.abort("blockedbyclient")
    else:
        route.continue_()


def _validate_url(url: str, *, allow_private_hosts: bool) -> str:
    """Validate one HTTP(S) URL and reject private, local, or special IP targets."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Browser URL must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError("Browser URL must not contain embedded credentials.")
    if allow_private_hosts:
        return parsed.geturl()

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = {
            str(item[4][0])
            for item in socket.getaddrinfo(
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise ValueError(f"Browser host could not be resolved: {parsed.hostname}") from exc
    if not addresses:
        raise ValueError(f"Browser host could not be resolved: {parsed.hostname}")
    if any(not _is_public_address(address) for address in addresses):
        raise ValueError("Browser access to private or local network addresses is blocked.")
    return parsed.geturl()


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return bool(address.is_global)

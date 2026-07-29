import importlib.util
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qwopus_agent.integrations.playwright_browser import PlaywrightBrowserProvider


class _DynamicPageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = (
            b"<html><head><title>Qwopus Browser Test</title></head>"
            b"<body><div id='result'>before script</div>"
            b"<script>document.getElementById('result').textContent="
            b"'rendered by javascript';</script></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        del args


class PlaywrightBrowserProviderTests(unittest.TestCase):
    def test_private_network_is_blocked_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "private or local"):
            PlaywrightBrowserProvider().open("http://127.0.0.1:8000")

    @unittest.skipUnless(
        importlib.util.find_spec("playwright") is not None
        and Path("/Applications/Google Chrome.app").is_dir(),
        "Playwright and system Chrome are required for the real browser case.",
    )
    def test_real_browser_returns_javascript_rendered_snapshot(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DynamicPageHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = PlaywrightBrowserProvider(allow_private_hosts=True)
            snapshot = provider.open(
                f"http://127.0.0.1:{server.server_address[1]}/"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        # 原因：只检查静态 HTML 无法证明 Provider 使用了真实浏览器引擎。
        # 作用：JavaScript 改写后的文本和页面标题共同验证 Chrome 渲染链路。
        self.assertIn("Qwopus Browser Test", snapshot)
        self.assertIn("rendered by javascript", snapshot)
        self.assertEqual(provider.snapshot(), snapshot)


if __name__ == "__main__":
    unittest.main()

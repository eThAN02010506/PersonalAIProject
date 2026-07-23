import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // 原因：assistant-ui、Markdown 和 React 默认会合并成超过 500 kB 的首屏文件。
        // 作用：按稳定依赖边界拆包，使浏览器可以并行加载并跨版本复用缓存。
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("@assistant-ui")) return "assistant-ui";
          if (
            id.includes("react-markdown")
            || id.includes("remark-")
            || id.includes("rehype-")
            || id.includes("micromark")
            || id.includes("mdast-")
            || id.includes("hast-")
            || id.includes("/unified/")
          ) {
            return "markdown";
          }
          return "vendor";
        },
      },
    },
  },
  // 原因：开发模式下浏览器与 FastAPI 端口不同，统一代理可避免硬编码服务器地址。
  // 作用：生产构建继续使用同源 /api，调试时自动转发到本地 Python API。
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // 原因：开发模式下浏览器与 FastAPI 端口不同，统一代理可避免硬编码服务器地址。
  // 作用：生产构建继续使用同源 /api，调试时自动转发到本地 Python API。
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});

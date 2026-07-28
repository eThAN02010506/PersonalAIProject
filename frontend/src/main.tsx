import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./styles.css";

// 原因：Debug Console 包含大量原始记录展示逻辑，不应增加普通聊天首屏体积。
// 作用：只有访问 /debug 时才加载诊断页面，正式前端仍保持轻量。
const DebugConsole = lazy(() =>
  import("./components/DebugConsole").then((module) => ({
    default: module.DebugConsole,
  })),
);
const isDebugConsole = window.location.pathname === "/debug";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isDebugConsole ? (
      <Suspense fallback={<div className="workspace-loading">Loading diagnostics...</div>}>
        <DebugConsole />
      </Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);

import type {
  AnalysisResult,
  ChatMessage,
  Conversation,
  Health,
  RunView,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),

  listConversations: () => request<Conversation[]>("/api/conversations"),

  createConversation: (title = "New chat") =>
    request<Conversation>("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),

  deleteConversation: async (conversationId: string): Promise<void> => {
    const response = await fetch(`/api/conversations/${conversationId}`, { method: "DELETE" });
    if (!response.ok) throw new Error(`Delete failed (${response.status})`);
  },

  listMessages: (conversationId: string) =>
    request<ChatMessage[]>(`/api/conversations/${conversationId}/messages`),

  startRun: (
    conversationId: string,
    content: string,
    options: { enableWebSearch: boolean; enableLocalKnowledge: boolean },
  ) =>
    request<{ run_id: string; status: "running" }>(
      `/api/conversations/${conversationId}/runs`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content,
          enable_web_search: options.enableWebSearch,
          enable_local_knowledge: options.enableLocalKnowledge,
        }),
      },
    ),

  getRun: (runId: string) => request<RunView>(`/api/runs/${runId}`),

  cancelRun: (runId: string) =>
    request<RunView>(`/api/runs/${runId}`, { method: "DELETE" }),

  analyze: (files: File[], question: string, generateReport: boolean) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    form.append("question", question);
    form.append("generate_report", String(generateReport));
    return request<AnalysisResult>("/api/analysis", { method: "POST", body: form });
  },
};

export async function waitForRun(
  runId: string,
  onUpdate: (run: RunView) => void,
): Promise<RunView> {
  // 原因：Agent 运行在独立 Python 进程中，HTTP 请求不能一直占用连接。
  // 作用：以短轮询读取真实阶段，并允许页面随时调用取消接口终止任务。
  while (true) {
    const run = await api.getRun(runId);
    onUpdate(run);
    if (run.status !== "running") return run;
    await new Promise((resolve) => window.setTimeout(resolve, 700));
  }
}

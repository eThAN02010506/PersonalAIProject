import type {
  AnalysisResult,
  ChatMessage,
  Conversation,
  DebugOverview,
  DebugRecord,
  Health,
  LocalFolderTree,
  RunView,
  SavedDocument,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const contentType = response.headers.get("content-type") ?? "";
  if (!response.ok) {
    const payload = contentType.includes("application/json")
      ? ((await response.json().catch(() => null)) as { detail?: string } | null)
      : null;
    throw new Error(payload?.detail ?? `Request failed (${response.status})`);
  }
  if (!contentType.includes("application/json")) {
    throw new Error(
      `The API returned a web page for ${path}. Restart the backend so its routes match the frontend.`,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/api/health"),

  debugOverview: (limit = 100, logLines = 500) =>
    request<DebugOverview>(`/api/debug?limit=${limit}&log_lines=${logLines}`),

  debugRecord: (recordId: string) =>
    request<DebugRecord>(`/api/debug/records/${encodeURIComponent(recordId)}`),

  updateModelSettings: (payload: {
    mode: "remote" | "local";
    base_url?: string;
    model_path?: string;
    context_window_tokens: number;
    agent_mode: "tool_calling" | "code";
    supports_structured_output: boolean;
    supports_vision: boolean;
  }) =>
    request<Health>("/api/model-settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

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

  listDocuments: () => request<SavedDocument[]>("/api/documents"),

  attachSavedDocuments: (conversationId: string, documentIds: string[]) =>
    request<{
      conversation_id: string;
      attached_count: number;
      documents: SavedDocument[];
    }>(`/api/conversations/${conversationId}/documents/attach`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: documentIds }),
    }),

  analyzeSavedDocuments: (payload: {
    conversationId: string;
    documentIds: string[];
    question: string;
    generateReport: boolean;
    minSourceRelevance: number;
    analysisMode: "question" | "section" | "full";
    selectedSections: Record<string, string[]>;
  }) =>
    request<AnalysisResult>("/api/documents/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: payload.conversationId,
        document_ids: payload.documentIds,
        question: payload.question,
        generate_report: payload.generateReport,
        min_source_relevance: payload.minSourceRelevance,
        analysis_mode: payload.analysisMode,
        selected_sections: payload.selectedSections,
      }),
    }),

  scanLocalFolder: (path: string) =>
    request<LocalFolderTree>("/api/local-folders/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    }),

  startRun: (
    conversationId: string,
    content: string,
    options: {
      enableWebSearch: boolean;
      enableLocalKnowledge: boolean;
      includeGlobalKnowledge: boolean;
      minSourceRelevance: number;
      responseDetail: "concise" | "balanced" | "detailed";
    },
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
          include_global_knowledge: options.includeGlobalKnowledge,
          min_source_relevance: options.minSourceRelevance,
          response_detail: options.responseDetail,
        }),
      },
    ),

  getRun: (runId: string) => request<RunView>(`/api/runs/${runId}`),

  cancelRun: (runId: string) =>
    request<RunView>(`/api/runs/${runId}`, { method: "DELETE" }),

  analyze: (
    conversationId: string,
    files: File[],
    question: string,
    generateReport: boolean,
    minSourceRelevance: number,
    analysisMode: "question" | "section" | "full",
    selectedSections: Record<string, string[]>,
  ) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    // 原因：上传文件必须进入当前聊天的私有 MiniRAG，而不是进程级共享知识库。
    // 作用：后端以 conversation_id 选择持久化目录，并拒绝不存在的聊天。
    form.append("conversation_id", conversationId);
    form.append("question", question);
    form.append("generate_report", String(generateReport));
    form.append("min_source_relevance", String(minSourceRelevance));
    form.append("analysis_mode", analysisMode);
    form.append("selected_sections", JSON.stringify(selectedSections));
    return request<AnalysisResult>("/api/analysis", { method: "POST", body: form });
  },

  analyzeLocalFolder: (payload: {
    conversationId: string;
    root: string;
    selectedFiles: string[];
    question: string;
    generateReport: boolean;
    analysisMode: "question" | "section" | "full";
    selectedSections: Record<string, string[]>;
  }) =>
    request<AnalysisResult>("/api/local-folders/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: payload.conversationId,
        root: payload.root,
        selected_files: payload.selectedFiles,
        question: payload.question,
        generate_report: payload.generateReport,
        analysis_mode: payload.analysisMode,
        selected_sections: payload.selectedSections,
      }),
    }),
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

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type RunView = {
  run_id: string;
  status: "running" | "completed" | "failed" | "cancelled";
  phase: string;
  answer?: string;
  trace: Record<string, unknown>[];
  citations: Record<string, unknown>[];
  error?: string;
};

export type AnalysisResult = {
  answer: string;
  route: string;
  citations: Record<string, unknown>[];
  trace: Record<string, unknown>[];
  reports: Array<{ kind: string; name: string; url: string }>;
};

export type Health = {
  status: string;
  model_online: boolean;
  message: string;
  model: string;
};

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
  documents: DocumentOutline[];
};

export type DocumentSection = {
  id: string;
  title: string;
  level: number;
  parent_id?: string;
  section_path: string[];
  page_start?: number;
  page_end?: number;
};

export type DocumentOutline = {
  document_id: string;
  source: string;
  sections: DocumentSection[];
};

export type Health = {
  status: string;
  mode: "remote" | "local";
  model_online: boolean;
  message: string;
  model: string;
  base_url: string;
  local_model_path?: string;
};

export type DebugAgentRun = {
  label?: string;
  prompt?: string;
  max_steps?: number;
  state?: string;
  output?: string;
  steps?: Record<string, unknown>[];
};

export type DebugRecord = {
  id: string;
  timestamp?: string;
  source?: string;
  status?: string;
  run_id?: string;
  result?: string;
  trace?: Record<string, unknown>[];
  debug_runs?: DebugAgentRun[];
};

export type DebugRecordSummary = {
  id: string;
  timestamp?: string;
  source: string;
  status: string;
  run_id: string;
  result_preview: string;
  trace_events: number;
  agent_runs: number;
};

export type DebugRuntimeLog = {
  path: string;
  exists: boolean;
  size_bytes: number;
  modified_at?: string;
  total_lines: number;
  lines: string[];
  error?: string;
};

export type DebugOverview = {
  generated_at: string;
  uptime_seconds: number;
  process_id: number;
  python_version: string;
  platform: string;
  model: Health;
  active_runs: number;
  completed_runs: number;
  record_count: number;
  record_storage_bytes: number;
  source_counts: Record<string, number>;
  status_counts: Record<string, number>;
  records: DebugRecordSummary[];
  runtime_log: DebugRuntimeLog;
};

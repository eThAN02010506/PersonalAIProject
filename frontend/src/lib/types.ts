export type UserAccount = {
  id: string;
  username: string;
  display_name: string;
  role: "admin" | "member";
  active: boolean;
  created_at: string;
};

export type AuthStatus = {
  bootstrap_required: boolean;
  user?: UserAccount;
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  owner_user_id?: string;
  owner_username?: string;
  is_owner: boolean;
  shared_count: number;
};

export type ConversationMember = {
  user_id: string;
  username: string;
  display_name: string;
  access: "owner" | "member";
};

export type ChatMessage = {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
};

export type InterpretationMode = "precise" | "contextual" | "exploratory";

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
  spreadsheets: SpreadsheetWorkbook[];
  source_coverage?: SourceCoverage;
  generation_mode?: string;
};

export type SpreadsheetSheet = {
  name: string;
  kind: "empty" | "table" | "multi_table" | "form" | "matrix";
  region_count: number;
  formula_count: number;
  merged_range_count: number;
  chart_count: number;
  image_count: number;
  data_validation_count: number;
  hidden: boolean;
};

export type SpreadsheetTable = {
  name: string;
  source_sheet: string;
  rows: number;
  columns: number;
  column_names: string[];
  columns_truncated: boolean;
};

export type SpreadsheetWorkbook = {
  source: string;
  sheet_count: number;
  formula_count: number;
  merged_range_count: number;
  chart_count: number;
  image_count: number;
  data_validation_count: number;
  sheets: SpreadsheetSheet[];
  tables: SpreadsheetTable[];
};

export type SkillVersion = {
  name: string;
  version: string;
  description: string;
  status: "candidate" | "active" | "archived" | "rejected";
  created_at: string;
  source_run_id?: string;
  source_model?: string;
  intent_examples: string[];
  steps: Array<{ skill_name: string }>;
  spec_valid: boolean;
};

export type SkillCapability = {
  name: string;
  description: string;
};

export type SkillSourceConversation = {
  id: string;
  title: string;
  owner_username?: string;
  updated_at: string;
};

export type SkillSourceRun = {
  run_id: string;
  conversation_id: string;
  objective: string;
  operational_objective: string;
  model_id: string;
  reusable_skills: string[];
  answer_preview: string;
  created_at: string;
};

export type SkillCandidateReview = {
  skill: SkillVersion;
  spec_json: string;
  diff: string;
  checks: Array<{
    name: string;
    passed: boolean;
    detail: string;
  }>;
  model_output?: string;
};

export type SkillCandidateTest = {
  success: boolean;
  output: string;
  steps: Array<{
    skill_name: string;
    query: string;
    argument_keys: string[];
  }>;
};

export type SourceCoverage = {
  required_sources: string[];
  covered_sources: string[];
  missing_sources: string[];
  complete: boolean;
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

export type SavedDocument = {
  document_id: string;
  source: string;
  file_type: string;
  size_bytes: number;
  section_count: number;
  saved_at: string;
  summary_available: boolean;
};

export type LocalFolderNode = {
  name: string;
  relative_path: string;
  kind: "directory" | "file";
  children: LocalFolderNode[];
};

export type LocalFolderTree = {
  root: string;
  file_count: number;
  max_selection: number;
  tree: LocalFolderNode;
};

export type Health = {
  status: string;
  mode: "remote" | "local";
  model_online: boolean;
  message: string;
  model: string;
  base_url: string;
  local_model_path?: string;
  context_window_tokens: number;
  agent_mode: "tool_calling" | "code";
  supports_structured_output: boolean;
  supports_vision: boolean;
  request_timeout_seconds: number;
  max_retries: number;
  run_timeout_seconds: number;
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
  user_id?: string;
  username?: string;
  result?: string;
  trace?: Record<string, unknown>[];
  debug_runs?: DebugAgentRun[];
  metrics?: Record<string, unknown>;
};

export type DebugRecordSummary = {
  id: string;
  timestamp?: string;
  source: string;
  status: string;
  run_id: string;
  user_id?: string;
  username?: string;
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

import {
  Activity,
  ArrowLeft,
  Bot,
  Bug,
  CheckCircle2,
  CircleAlert,
  Clock3,
  Database,
  Download,
  FileJson,
  Layers3,
  RefreshCw,
  Server,
  Terminal,
  Wrench,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "../lib/api";
import type { DebugAgentRun, DebugOverview, DebugRecord } from "../lib/types";
import { SkillWorkspace } from "./SkillWorkspace";

type ConsoleView = "runs" | "logs" | "skills";

export function DebugConsole() {
  const [overview, setOverview] = useState<DebugOverview | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedRecord, setSelectedRecord] = useState<DebugRecord | null>(null);
  const [view, setView] = useState<ConsoleView>("runs");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [loadingRecord, setLoadingRecord] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const refreshInFlight = useRef(false);

  const refresh = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    try {
      const snapshot = await api.debugOverview();
      setOverview(snapshot);
      setSelectedId((current) => {
        if (current && snapshot.records.some((record) => record.id === current)) {
          return current;
        }
        return snapshot.records[0]?.id ?? null;
      });
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load diagnostics.");
    } finally {
      // 原因：setInterval 不会等待上一次网络请求，离线模型探测可能让刷新重叠。
      // 作用：同一标签页始终最多保留一个 Debug overview 请求。
      refreshInFlight.current = false;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!autoRefresh) return;
    // 原因：Agent worker 在后台完成时不会主动推送原始诊断记录。
    // 作用：短周期只读刷新让 Console 跟上任务状态，不占用聊天长连接。
    const timer = window.setInterval(() => void refresh(), 5_000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, refresh]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedRecord(null);
      return;
    }
    let active = true;
    setLoadingRecord(true);
    // 原因：完整 Prompt 和 Observation 可能很大，不能随五秒概览轮询重复传输。
    // 作用：仅在用户选择运行时加载一次完整记录，仍保留全部调试字段。
    void api.debugRecord(selectedId)
      .then((record) => {
        if (active) setSelectedRecord(record);
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : "Unable to load run.");
      })
      .finally(() => {
        if (active) setLoadingRecord(false);
      });
    return () => {
      active = false;
    };
  }, [selectedId]);

  const filteredRecords = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return (overview?.records ?? []).filter((record) => {
      const matchesSource = sourceFilter === "all" || record.source === sourceFilter;
      const matchesStatus = statusFilter === "all" || record.status === statusFilter;
      const searchable = [
        record.run_id,
        record.username ?? "",
        record.source,
        record.status,
        record.result_preview,
      ].join(" ").toLowerCase();
      return matchesSource && matchesStatus && (!normalizedQuery || searchable.includes(normalizedQuery));
    });
  }, [overview, query, sourceFilter, statusFilter]);

  return (
    <main className="debug-console">
      <header className="debug-header">
        <div className="debug-title">
          <a className="icon-button" href="/" title="Back to Qwopus Agent">
            <ArrowLeft size={18} />
          </a>
          <div className="brand-mark"><Bug size={18} /></div>
          <div>
            <h1>Debug Console</h1>
            <p>Local backend diagnostics</p>
          </div>
        </div>
        <div className="debug-header-actions">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.target.checked)}
            />
            Auto refresh
          </label>
          <button className="secondary-button debug-action" onClick={() => void refresh()}>
            <RefreshCw size={15} className={loading ? "spin" : ""} />
            Refresh
          </button>
        </div>
      </header>

      <section className="debug-notice">
        <CircleAlert size={17} />
        <span>
          Local diagnostics may contain complete prompts, document excerpts, Tool Observations,
          errors and model-returned reasoning drafts. Hidden provider reasoning is not available.
        </span>
      </section>

      {error && <div className="error-banner debug-error">{error}</div>}
      {overview && <RuntimeSummary overview={overview} />}

      <nav className="debug-tabs" aria-label="Debug views">
        <button className={view === "runs" ? "active" : ""} onClick={() => setView("runs")}>
          <Activity size={16} /> Agent runs
        </button>
        <button className={view === "logs" ? "active" : ""} onClick={() => setView("logs")}>
          <Terminal size={16} /> Runtime log
        </button>
        <button className={view === "skills" ? "active" : ""} onClick={() => setView("skills")}>
          <Layers3 size={16} /> Skill authoring
        </button>
      </nav>

      {view === "runs" ? (
        <section className="debug-runs">
          <aside className="debug-run-browser">
            <div className="debug-filters">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Filter run id or result"
                aria-label="Filter debug runs"
              />
              <div>
                <select
                  value={sourceFilter}
                  onChange={(event) => setSourceFilter(event.target.value)}
                  aria-label="Filter by source"
                >
                  <option value="all">All sources</option>
                  {Object.keys(overview?.source_counts ?? {}).map((source) => (
                    <option key={source} value={source}>{source}</option>
                  ))}
                </select>
                <select
                  value={statusFilter}
                  onChange={(event) => setStatusFilter(event.target.value)}
                  aria-label="Filter by status"
                >
                  <option value="all">All statuses</option>
                  {Object.keys(overview?.status_counts ?? {}).map((status) => (
                    <option key={status} value={status}>{status}</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="debug-run-list">
              {filteredRecords.map((record) => (
                <button
                  key={record.id}
                  className={record.id === selectedId ? "active" : ""}
                  onClick={() => setSelectedId(record.id)}
                >
                  <span className={`debug-status ${record.status ?? "unknown"}`} />
                  <span>
                    <strong>{record.source ?? "agent"}</strong>
                    <small>{record.username ? `@${record.username}` : "legacy / system"}</small>
                    <small>{record.run_id ?? record.id}</small>
                    <small>{record.trace_events} events · {record.agent_runs} Agent runs</small>
                    <time>{formatDate(record.timestamp)}</time>
                  </span>
                </button>
              ))}
              {!filteredRecords.length && (
                <p className="debug-empty">No diagnostics match these filters.</p>
              )}
            </div>
          </aside>
          <div className="debug-record-detail">
            {loadingRecord ? (
              <div className="debug-empty-detail">
                <RefreshCw size={20} className="spin" />
              </div>
            ) : selectedRecord ? (
              <RecordDetail record={selectedRecord} />
            ) : (
              <div className="debug-empty-detail">
                <Wrench size={22} />
                <p>Run an Agent task to capture diagnostic data.</p>
              </div>
            )}
          </div>
        </section>
      ) : view === "logs" ? (
        <RuntimeLog overview={overview} />
      ) : (
        <section className="debug-skill-view">
          <SkillWorkspace enableAuthoring />
        </section>
      )}
    </main>
  );
}

function RuntimeSummary({ overview }: { overview: DebugOverview }) {
  const metrics = [
    { label: "Model", value: shortModelName(overview.model.model), icon: Bot },
    {
      label: "Connection",
      value: overview.model.model_online ? "Online" : "Offline",
      icon: overview.model.model_online ? CheckCircle2 : CircleAlert,
    },
    { label: "Active runs", value: String(overview.active_runs), icon: Activity },
    { label: "Captured runs", value: String(overview.record_count), icon: FileJson },
    { label: "Uptime", value: formatDuration(overview.uptime_seconds), icon: Clock3 },
    { label: "Trace storage", value: formatBytes(overview.record_storage_bytes), icon: Database },
  ];
  return (
    <>
      <section className="debug-metrics">
        {metrics.map(({ label, value, icon: Icon }) => (
          <div key={label}>
            <Icon size={16} />
            <span>{label}</span>
            <strong title={value}>{value}</strong>
          </div>
        ))}
      </section>
      <section className="debug-runtime-strip">
        <Server size={15} />
        <span>PID {overview.process_id}</span>
        <span>Python {overview.python_version}</span>
        <span>{overview.model.mode} · {overview.model.base_url}</span>
        <span title={overview.platform}>{overview.platform}</span>
        <time>{formatDate(overview.generated_at)}</time>
      </section>
    </>
  );
}

function RecordDetail({ record }: { record: DebugRecord }) {
  return (
    <>
      <header className="debug-record-header">
        <div>
          <span className={`debug-status ${record.status ?? "unknown"}`} />
          <h2>{record.source ?? "agent"} run</h2>
          <code>{record.run_id ?? record.id}</code>
        </div>
        <button
          className="secondary-button debug-action"
          onClick={() => downloadText(
            `qwopus-debug-${record.run_id ?? record.id}.json`,
            JSON.stringify(record, null, 2),
            "application/json",
          )}
        >
          <Download size={15} /> JSON
        </button>
      </header>

      <section className="debug-section">
        <h3>Final result</h3>
        <pre>{record.result || "No final result was recorded."}</pre>
      </section>

      <section className="debug-section">
        <h3>Run metrics</h3>
        <pre>{JSON.stringify(record.metrics ?? {}, null, 2)}</pre>
      </section>

      <section className="debug-section">
        <h3>Orchestration trace</h3>
        <div className="trace-table">
          {(record.trace ?? []).map((event, index) => (
            <div key={index}>
              <span>{index + 1}</span>
              <pre>{JSON.stringify(event, null, 2)}</pre>
            </div>
          ))}
          {!record.trace?.length && <p>No orchestration events were recorded.</p>}
        </div>
      </section>

      <section className="debug-section">
        <h3>Raw Agent runs</h3>
        {(record.debug_runs ?? []).map((run, index) => (
          <RawAgentRun key={`${run.label ?? "run"}-${index}`} run={run} index={index} />
        ))}
        {!record.debug_runs?.length && <p>No raw Agent steps were recorded.</p>}
      </section>
    </>
  );
}

function RawAgentRun({ run, index }: { run: DebugAgentRun; index: number }) {
  const steps = run.steps ?? [];
  return (
    <details className="debug-raw-run" open={index === 0}>
      <summary>
        <span>{run.label ?? `run_${index + 1}`}</span>
        <small>
          state={run.state ?? "unknown"} · max_steps={run.max_steps ?? "?"} ·
          recorded_steps={steps.length}
        </small>
      </summary>
      <div className="debug-raw-body">
        <RawField title="Complete prompt" value={run.prompt} />
        <RawField title="Raw run output" value={run.output} />
        {steps.map((step, stepIndex) => (
          <details className="debug-step" key={stepIndex}>
            <summary>Step {String(step.step_number ?? stepIndex + 1)}</summary>
            <RawField title="Model output / reasoning draft" value={step.model_output} />
            <RawField title="Tool calls / arguments" value={step.tool_calls} json />
            <RawField title="Tool Observation" value={step.observations} />
            <RawField title="Error" value={step.error} />
            <RawField title="All recorded fields" value={step} json />
          </details>
        ))}
      </div>
    </details>
  );
}

function RawField({
  title,
  value,
  json = false,
}: {
  title: string;
  value: unknown;
  json?: boolean;
}) {
  if (value === undefined || value === null || value === "") return null;
  return (
    <div className="debug-raw-field">
      <h4>{title}</h4>
      <pre>{json ? JSON.stringify(value, null, 2) : String(value)}</pre>
    </div>
  );
}

function RuntimeLog({ overview }: { overview: DebugOverview | null }) {
  const runtimeLog = overview?.runtime_log;
  return (
    <section className="debug-log-view">
      <header>
        <div>
          <h2>Runtime log</h2>
          <p>
            {runtimeLog?.path ?? "logs/qwopus_agent.log"} ·
            {runtimeLog?.exists
              ? ` ${runtimeLog.total_lines} lines · ${formatBytes(runtimeLog.size_bytes)}`
              : " not created"}
          </p>
        </div>
        <button
          className="secondary-button debug-action"
          disabled={!runtimeLog?.lines.length}
          onClick={() => downloadText(
            "qwopus_agent.log",
            runtimeLog?.lines.join("\n") ?? "",
            "text/plain",
          )}
        >
          <Download size={15} /> Log tail
        </button>
      </header>
      {runtimeLog?.error && <div className="error-banner">{runtimeLog.error}</div>}
      <pre className="runtime-log-output">
        {runtimeLog?.lines.length
          ? runtimeLog.lines.join("\n")
          : "No runtime log lines are available."}
      </pre>
    </section>
  );
}

function downloadText(filename: string, value: string, mimeType: string) {
  const url = URL.createObjectURL(new Blob([value], { type: mimeType }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function shortModelName(model: string) {
  return model.replaceAll("\\", "/").replace(/\/+$/, "").split("/").at(-1) || model;
}

function formatDate(value?: string) {
  if (!value) return "Unknown time";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatDuration(seconds: number) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3_600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3_600)}h ${Math.floor((seconds % 3_600) / 60)}m`;
}

function formatBytes(bytes: number) {
  if (bytes < 1_024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

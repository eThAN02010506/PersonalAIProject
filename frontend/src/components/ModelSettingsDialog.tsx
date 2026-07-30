import { Globe2, HardDrive, LoaderCircle, X } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { Health } from "../lib/types";

type ModelSettingsDialogProps = {
  open: boolean;
  health: Health | null;
  onClose: () => void;
  onSaved: (health: Health) => void;
};

export function ModelSettingsDialog({
  open,
  health,
  onClose,
  onSaved,
}: ModelSettingsDialogProps) {
  const [mode, setMode] = useState<"remote" | "local">("remote");
  const [baseUrl, setBaseUrl] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [contextWindow, setContextWindow] = useState(32768);
  const [agentMode, setAgentMode] = useState<"tool_calling" | "code">("tool_calling");
  const [requestTimeout, setRequestTimeout] = useState(120);
  const [maxRetries, setMaxRetries] = useState(1);
  const [runTimeout, setRunTimeout] = useState(600);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    // 原因：服务可能在其他标签页被切换，弹窗不能沿用过期的表单状态。
    // 作用：每次打开都以当前后端状态填充地址、模式和本地路径。
    setMode(health?.mode ?? "remote");
    setBaseUrl(health?.base_url ?? "");
    setModelPath(health?.local_model_path ?? "");
    setContextWindow(health?.context_window_tokens ?? 32768);
    setAgentMode(health?.agent_mode ?? "tool_calling");
    setRequestTimeout(health?.request_timeout_seconds ?? 120);
    setMaxRetries(health?.max_retries ?? 1);
    setRunTimeout(health?.run_timeout_seconds ?? 600);
    setError(null);
  }, [health, open]);

  if (!open) return null;

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const updated = await api.updateModelSettings(
        mode === "remote"
          ? {
              mode,
              base_url: baseUrl,
              context_window_tokens: contextWindow,
              agent_mode: agentMode,
              supports_structured_output: health?.supports_structured_output ?? false,
              supports_vision: health?.supports_vision ?? false,
              request_timeout_seconds: requestTimeout,
              max_retries: maxRetries,
              run_timeout_seconds: runTimeout,
            }
          : {
              mode,
              model_path: modelPath,
              context_window_tokens: contextWindow,
              agent_mode: agentMode,
              supports_structured_output: health?.supports_structured_output ?? false,
              supports_vision: health?.supports_vision ?? false,
              request_timeout_seconds: requestTimeout,
              max_retries: maxRetries,
              run_timeout_seconds: runTimeout,
            },
      );
      onSaved(updated);
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Model connection failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <section
        className="model-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-settings-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="dialog-header">
          <h2 id="model-settings-title">Model connection</h2>
          <button className="icon-button" type="button" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </header>

        <form onSubmit={(event) => void submit(event)}>
          <div className="model-mode-switch" role="tablist" aria-label="Model source">
            <button
              type="button"
              className={mode === "remote" ? "active" : ""}
              onClick={() => setMode("remote")}
            >
              <Globe2 size={16} /> Remote API
            </button>
            <button
              type="button"
              className={mode === "local" ? "active" : ""}
              onClick={() => setMode("local")}
            >
              <HardDrive size={16} /> Local MLX
            </button>
          </div>

          {mode === "remote" ? (
            <label className="model-field">
              <span>OpenAI-compatible address</span>
              <input
                type="url"
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder="http://192.168.1.97:8001/v1"
                required
                autoFocus
              />
            </label>
          ) : (
            <label className="model-field">
              <span>Local model directory</span>
              <input
                type="text"
                value={modelPath}
                onChange={(event) => setModelPath(event.target.value)}
                placeholder="/Users/name/Desktop/model/model-name"
                required
                autoFocus
              />
            </label>
          )}

          <div className="model-capability-grid">
            <label className="model-field">
              <span>Context window</span>
              <input
                type="number"
                min="2048"
                step="1024"
                value={contextWindow}
                onChange={(event) => setContextWindow(Number(event.target.value))}
                required
              />
            </label>
            <label className="model-field">
              <span>Agent protocol</span>
              <select
                value={agentMode}
                onChange={(event) =>
                  setAgentMode(event.target.value as "tool_calling" | "code")
                }
              >
                <option value="tool_calling">Tool calling</option>
                <option value="code">Code actions</option>
              </select>
            </label>
          </div>

          <div className="model-capability-grid">
            <label className="model-field">
              <span>Request timeout (seconds)</span>
              <input
                type="number"
                min="1"
                max="600"
                value={requestTimeout}
                onChange={(event) => setRequestTimeout(Number(event.target.value))}
                required
              />
            </label>
            <label className="model-field">
              <span>Transient retries</span>
              <input
                type="number"
                min="0"
                max="3"
                value={maxRetries}
                onChange={(event) => setMaxRetries(Number(event.target.value))}
                required
              />
            </label>
            <label className="model-field">
              <span>Run timeout (seconds)</span>
              <input
                type="number"
                min={requestTimeout}
                max="3600"
                value={runTimeout}
                onChange={(event) => setRunTimeout(Number(event.target.value))}
                required
              />
            </label>
          </div>

          {error && <div className="error-banner model-error">{error}</div>}

          <footer className="dialog-actions">
            <button type="button" className="secondary-button" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="primary-button" disabled={saving}>
              {saving && <LoaderCircle className="spin" size={16} />}
              {mode === "local" ? "Start and use" : "Connect"}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}

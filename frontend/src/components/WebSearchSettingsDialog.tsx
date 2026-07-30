import { ExternalLink, KeyRound, LoaderCircle, Trash2, X } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { WebSearchSettings } from "../lib/types";

type WebSearchSettingsDialogProps = {
  open: boolean;
  settings: WebSearchSettings;
  onClose: () => void;
  onChanged: (settings: WebSearchSettings) => void;
};

export function WebSearchSettingsDialog({
  open,
  settings,
  onClose,
  onChanged,
}: WebSearchSettingsDialogProps) {
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState<"save" | "test" | "delete" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) return;
    // 原因：后端只返回掩码，掩码不能被误当成真实 Key 再次保存。
    // 作用：每次打开都保持密码框为空，只有用户主动输入的新值才会离开浏览器。
    setApiKey("");
    setMessage(null);
    setFailed(false);
  }, [open]);

  if (!open) return null;

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy("save");
    setMessage(null);
    try {
      const updated = await api.updateWebSearchSettings(apiKey);
      onChanged(updated);
      setApiKey("");
      setFailed(false);
      setMessage("Tavily API key saved. New searches use it immediately.");
    } catch (reason) {
      setFailed(true);
      setMessage(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const test = async () => {
    setBusy("test");
    setMessage(null);
    try {
      const result = await api.testWebSearchSettings(apiKey || undefined);
      setFailed(!result.success);
      setMessage(result.message);
    } catch (reason) {
      setFailed(true);
      setMessage(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const remove = async () => {
    setBusy("delete");
    setMessage(null);
    try {
      const updated = await api.deleteWebSearchSettings();
      onChanged(updated);
      setApiKey("");
      setFailed(false);
      setMessage(
        updated.configured
          ? "Managed key removed. Tavily now uses the configured environment value."
          : "Tavily API key removed. Web search is disabled.",
      );
    } catch (reason) {
      setFailed(true);
      setMessage(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <section
        className="model-dialog web-search-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="web-search-settings-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="dialog-header">
          <div>
            <h2 id="web-search-settings-title">Tavily web search</h2>
            <span>Host-wide administrator setting</span>
          </div>
          <button className="icon-button" type="button" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </header>

        <form onSubmit={(event) => void save(event)}>
          <div className="credential-status">
            <span className={`status-dot ${settings.configured ? "online" : ""}`} />
            <div>
              <strong>{settings.configured ? "Configured" : "Not configured"}</strong>
              <small>
                {settings.masked_key
                  ? `${settings.masked_key} · ${formatSource(settings.source)}`
                  : settings.message}
              </small>
            </div>
          </div>

          <p className="credential-help">
            Create a key in the Tavily dashboard, paste it below, then test the
            connection. The key is stored only on this host and is never shown in full
            again.
            {" "}
            <a href="https://app.tavily.com/" target="_blank" rel="noreferrer">
              Open Tavily <ExternalLink size={12} />
            </a>
          </p>

          <label className="model-field">
            <span>{settings.configured ? "Replace API key" : "Tavily API key"}</span>
            <input
              type="password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              placeholder="tvly-..."
              autoComplete="new-password"
              spellCheck={false}
              autoFocus
            />
          </label>

          {message && (
            <div className={failed ? "error-banner model-error" : "credential-message"}>
              {message}
            </div>
          )}

          <footer className="dialog-actions credential-actions">
            {settings.source === "managed" && (
              <button
                type="button"
                className="danger-button"
                onClick={() => void remove()}
                disabled={busy !== null}
              >
                {busy === "delete" ? (
                  <LoaderCircle className="spin" size={15} />
                ) : (
                  <Trash2 size={15} />
                )}
                Remove
              </button>
            )}
            <span />
            <button
              type="button"
              className="secondary-button"
              onClick={() => void test()}
              disabled={busy !== null || (!apiKey && !settings.configured)}
            >
              {busy === "test" && <LoaderCircle className="spin" size={15} />}
              Test
            </button>
            <button
              type="submit"
              className="primary-button"
              disabled={busy !== null || !apiKey}
            >
              {busy === "save" ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <KeyRound size={15} />
              )}
              Save key
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}

function formatSource(source: WebSearchSettings["source"]): string {
  const labels: Record<Exclude<WebSearchSettings["source"], null>, string> = {
    managed: "Admin UI",
    legacy_local: ".env.local",
    environment: "Environment",
    none: "Not configured",
  };
  return source ? labels[source] : "Configured";
}

function toMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Tavily request failed";
}

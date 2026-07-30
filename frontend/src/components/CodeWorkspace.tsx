import {
  Check,
  ChevronRight,
  Code2,
  FileCode2,
  Folder,
  GitPullRequest,
  LoaderCircle,
  MessageSquare,
  Play,
  RotateCcw,
  Search,
  Send,
  ShieldCheck,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { api } from "../lib/api";
import type {
  CodeChange,
  CodeChatReply,
  CodeCommand,
  CodeFile,
  CodeSearchMatch,
  CodeTestResult,
  CodeTreeNode,
  CodeWorkspaceTree,
  CodeWorkspaceMessage,
} from "../lib/types";

export function CodeWorkspace() {
  const [workspacePath, setWorkspacePath] = useState(".");
  const [workspace, setWorkspace] = useState<CodeWorkspaceTree | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [activeFile, setActiveFile] = useState<CodeFile | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<CodeSearchMatch[]>([]);
  const [chatMessages, setChatMessages] = useState<CodeWorkspaceMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatReply, setChatReply] = useState<CodeChatReply | null>(null);
  const [objective, setObjective] = useState("");
  const [changes, setChanges] = useState<CodeChange[]>([]);
  const [activeChangeId, setActiveChangeId] = useState<string | null>(null);
  const [commands, setCommands] = useState<CodeCommand[]>([]);
  const [commandId, setCommandId] = useState("");
  const [testResult, setTestResult] = useState<CodeTestResult | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activeChange = useMemo(
    () => changes.find((change) => change.id === activeChangeId) ?? changes[0] ?? null,
    [activeChangeId, changes],
  );

  useEffect(() => {
    void api.listCodeChanges().then((items) => {
      setChanges(items);
      setActiveChangeId(items[0]?.id ?? null);
    }).catch(() => undefined);
  }, []);

  const scan = async () => {
    setBusy("scan");
    setError(null);
    try {
      const result = await api.scanCodeWorkspace(workspacePath);
      const availableCommands = await api.listCodeCommands(result.root);
      setWorkspace(result);
      setWorkspacePath(result.root);
      setCommands(availableCommands);
      setCommandId(availableCommands[0]?.id ?? "");
      setSelectedFiles(new Set());
      setActiveFile(null);
      setSearchResults([]);
      setChatMessages([]);
      setChatReply(null);
      setObjective("");
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const sendCodeMessage = async () => {
    const message = chatInput.trim();
    if (!workspace || !message) return;
    const priorMessages = chatMessages;
    setChatMessages((current) => [...current, { role: "user", content: message }]);
    setChatInput("");
    setBusy("chat");
    setError(null);
    try {
      const reply = await api.chatAboutCode({
        root: workspace.root,
        message,
        history: priorMessages,
        selectedFiles: [...selectedFiles],
      });
      setChatMessages((current) => [
        ...current,
        { role: "assistant", content: reply.message },
      ]);
      setChatReply(reply);
      if (reply.selected_files.length > 0) {
        setSelectedFiles(new Set(reply.selected_files));
      }
      if (reply.objective) {
        setObjective(reply.objective);
      }
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const openFile = async (path: string) => {
    if (!workspace) return;
    setBusy("read");
    setError(null);
    try {
      setActiveFile(await api.readCodeFile(workspace.root, path));
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const search = async () => {
    if (!workspace || !searchQuery.trim()) return;
    setBusy("search");
    setError(null);
    try {
      setSearchResults(await api.searchCodeWorkspace(workspace.root, searchQuery));
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const propose = async () => {
    if (!workspace || !objective.trim() || selectedFiles.size === 0) return;
    setBusy("propose");
    setError(null);
    setTestResult(null);
    try {
      const proposal = await api.proposeCodeChange({
        root: workspace.root,
        objective,
        selectedFiles: [...selectedFiles],
        // Agent 已读文件提供测试与调用方合同，但未勾选文件不能进入可编辑白名单。
        contextFiles: (chatReply?.inspected_files ?? []).filter(
          (path) => !selectedFiles.has(path),
        ),
      });
      setChanges((current) => [proposal, ...current]);
      setActiveChangeId(proposal.id);
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const changeAction = async (action: "apply" | "reject" | "rollback") => {
    if (!activeChange) return;
    setBusy(action);
    setError(null);
    try {
      const updated = await api.updateCodeChange(activeChange.id, action);
      setChanges((current) =>
        current.map((change) => change.id === updated.id ? updated : change)
      );
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  const runTest = async () => {
    if (!activeChange || !commandId) return;
    setBusy("test");
    setError(null);
    try {
      setTestResult(await api.testCodeChange(activeChange.id, commandId));
    } catch (reason) {
      setError(toMessage(reason));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="code-workspace">
      <header className="workspace-heading">
        <Code2 size={22} />
        <div>
          <h1>Code Workspace</h1>
          <p>Inspect a local Git repository, review model-proposed diffs, then approve explicitly.</p>
        </div>
        <span className="code-safety-label"><ShieldCheck size={14} /> Local admin only</span>
      </header>

      {error && <div className="error-banner code-error">{error}</div>}

      <section className="code-repository-bar">
        <label>
          <span>Git repository root</span>
          <div>
            <input
              value={workspacePath}
              onChange={(event) => setWorkspacePath(event.target.value)}
              placeholder="/Users/name/project"
            />
            <button className="secondary-button" onClick={() => void scan()} disabled={busy !== null}>
              {busy === "scan" ? <LoaderCircle className="spin" size={15} /> : <Folder size={15} />}
              Open
            </button>
          </div>
        </label>
        {workspace && <small>{workspace.file_count} eligible source files</small>}
      </section>

      {workspace && (
        <>
          <div className="code-browser-layout">
            <aside className="code-tree-pane">
              <header>
                <strong>Files</strong>
                <span>{selectedFiles.size}/8 selected</span>
              </header>
              <div className="code-tree-scroll">
                <CodeTree
                  node={workspace.tree}
                  selected={selectedFiles}
                  onToggle={(path, checked) => {
                    setSelectedFiles((current) => {
                      const next = new Set(current);
                      if (checked && next.size < 8) next.add(path);
                      if (!checked) next.delete(path);
                      return next;
                    });
                  }}
                  onOpen={(path) => void openFile(path)}
                />
              </div>
            </aside>

            <section className="code-reader-pane">
              <div className="code-search-row">
                <Search size={15} />
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void search();
                  }}
                  placeholder="Literal source search"
                />
                <button className="secondary-button" onClick={() => void search()}>
                  Search
                </button>
              </div>
              {searchResults.length > 0 && (
                <div className="code-search-results">
                  {searchResults.map((match) => (
                    <button
                      key={`${match.path}:${match.line}:${match.column}`}
                      onClick={() => void openFile(match.path)}
                    >
                      <code>{match.path}:{match.line}</code>
                      <span>{match.preview}</span>
                    </button>
                  ))}
                </div>
              )}
              {activeFile ? (
                <>
                  <header className="code-file-heading">
                    <FileCode2 size={15} />
                    <strong>{activeFile.path}</strong>
                    <span>{activeFile.total_lines} lines</span>
                  </header>
                  <pre className="code-source-preview">{activeFile.content}</pre>
                </>
              ) : (
                <div className="code-empty">Select a source file to inspect it.</div>
              )}
            </section>
          </div>

          <section className="code-conversation">
            <header>
              <MessageSquare size={15} />
              <div>
                <strong>Discuss the change</strong>
                <span>Ask about the code or describe the outcome in your own words.</span>
              </div>
            </header>
            <div className="code-conversation-thread">
              {chatMessages.length === 0 ? (
                <div className="code-chat-empty">
                  The agent will inspect relevant repository files before preparing a change.
                </div>
              ) : chatMessages.map((item, index) => (
                <div
                  className={`code-chat-message ${item.role}`}
                  key={`${item.role}-${index}-${item.content.slice(0, 20)}`}
                >
                  <strong>{item.role === "user" ? "You" : "Agent"}</strong>
                  <p>{item.content}</p>
                </div>
              ))}
              {busy === "chat" && (
                <div className="code-chat-message assistant pending">
                  <LoaderCircle className="spin" size={14} />
                  <span>Inspecting the repository and reasoning about your request...</span>
                </div>
              )}
            </div>
            {chatReply && (
              <div className={`code-chat-state ${chatReply.mode}`}>
                <strong>
                  {chatReply.mode === "ready"
                    ? "Ready for a reviewed diff"
                    : chatReply.mode === "clarify"
                      ? "Needs one decision"
                      : "Repository answer"}
                </strong>
                {chatReply.inspected_files.length > 0 && (
                  <span>Inspected: {chatReply.inspected_files.join(", ")}</span>
                )}
              </div>
            )}
            <div className="code-chat-composer">
              <textarea
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                    event.preventDefault();
                    void sendCodeMessage();
                  }
                }}
                rows={3}
                placeholder="Describe a goal, bug, or abstract requirement. Use Cmd/Ctrl+Enter to send."
              />
              <button
                className="primary-button"
                onClick={() => void sendCodeMessage()}
                disabled={busy !== null || !chatInput.trim()}
                title="Send this message and let the agent inspect relevant code"
              >
                <Send size={15} />
                Send
              </button>
            </div>
          </section>

          <section className="code-proposal-editor">
            <label>
              <span>Implementation objective</span>
              <textarea
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                rows={4}
                placeholder="Discuss the requirement above or enter a precise implementation objective."
              />
            </label>
            <div>
              <span>
                The model can propose exact edits only in the {selectedFiles.size} selected file(s).
              </span>
              <button
                className="primary-button"
                onClick={() => void propose()}
                disabled={busy !== null || !objective.trim() || selectedFiles.size === 0}
              >
                {busy === "propose"
                  ? <LoaderCircle className="spin" size={15} />
                  : <GitPullRequest size={15} />}
                Generate diff
              </button>
            </div>
          </section>
        </>
      )}

      {changes.length > 0 && (
        <section className="code-review">
          <aside className="code-change-list">
            <header>Change history</header>
            {changes.map((change) => (
              <button
                key={change.id}
                className={activeChange?.id === change.id ? "active" : ""}
                onClick={() => {
                  setActiveChangeId(change.id);
                  setTestResult(null);
                }}
              >
                <strong>{change.summary}</strong>
                <span className={`code-status ${change.status}`}>{change.status}</span>
                <small>{change.changed_files.join(", ")}</small>
              </button>
            ))}
          </aside>

          {activeChange && (
            <div className="code-diff-review">
              <header>
                <div>
                  <h2>{activeChange.summary}</h2>
                  <p>{activeChange.reason}</p>
                </div>
                <span className={`code-status ${activeChange.status}`}>
                  {activeChange.status}
                </span>
              </header>
              <pre>{activeChange.unified_diff}</pre>
              <div className="code-verification-plan">
                <strong>Verification plan</strong>
                <ol>
                  {activeChange.verification_plan.map((step) => <li key={step}>{step}</li>)}
                </ol>
              </div>
              <footer>
                {activeChange.status === "proposed" && (
                  <>
                    <button className="secondary-button" onClick={() => void changeAction("reject")}>
                      <X size={15} /> Reject
                    </button>
                    <button className="primary-button" onClick={() => void changeAction("apply")}>
                      <Check size={15} /> Apply change
                    </button>
                  </>
                )}
                {activeChange.status === "applied" && (
                  <>
                    <label>
                      <select value={commandId} onChange={(event) => setCommandId(event.target.value)}>
                        {commands.map((command) => (
                          <option key={command.id} value={command.id}>{command.label}</option>
                        ))}
                      </select>
                    </label>
                    <button className="secondary-button" onClick={() => void runTest()} disabled={!commandId}>
                      <Play size={15} /> Run check
                    </button>
                    <button className="secondary-button" onClick={() => void changeAction("rollback")}>
                      <RotateCcw size={15} /> Roll back
                    </button>
                  </>
                )}
              </footer>
              {testResult && (
                <div className={`code-test-result ${testResult.success ? "passed" : "failed"}`}>
                  <strong>{testResult.success ? "Check passed" : "Check failed"}</strong>
                  <code>{testResult.command.join(" ")}</code>
                  <pre>{testResult.output || "(no output)"}</pre>
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </section>
  );
}

function CodeTree({
  node,
  selected,
  onToggle,
  onOpen,
  depth = 0,
}: {
  node: CodeTreeNode;
  selected: Set<string>;
  onToggle: (path: string, checked: boolean) => void;
  onOpen: (path: string) => void;
  depth?: number;
}) {
  if (node.kind === "file") {
    return (
      <div className="code-tree-file" style={{ paddingLeft: `${10 + depth * 14}px` }}>
        <input
          type="checkbox"
          checked={selected.has(node.relative_path)}
          onChange={(event) => onToggle(node.relative_path, event.target.checked)}
          aria-label={`Select ${node.relative_path}`}
        />
        <button onClick={() => onOpen(node.relative_path)}>
          <FileCode2 size={14} />
          <span>{node.name}</span>
        </button>
      </div>
    );
  }
  return (
    <details className="code-tree-directory" open={depth < 2}>
      <summary style={{ paddingLeft: `${8 + depth * 14}px` }}>
        <ChevronRight size={13} />
        <Folder size={14} />
        <span>{node.name}</span>
      </summary>
      {node.children.map((child) => (
        <CodeTree
          key={child.relative_path}
          node={child}
          selected={selected}
          onToggle={onToggle}
          onOpen={onOpen}
          depth={depth + 1}
        />
      ))}
    </details>
  );
}

function toMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Unexpected Code Workspace error";
}

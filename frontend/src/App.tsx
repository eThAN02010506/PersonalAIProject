import {
  FileSearch,
  Globe2,
  Layers3,
  Menu,
  MessageCircle,
  Monitor,
  Network,
  Search,
  ScanText,
  Share2,
  SlidersHorizontal,
  TextSelect,
  Wrench,
  X,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useState } from "react";

import { AgentRuntimeProvider } from "./components/AgentRuntimeProvider";
import { AuthScreen } from "./components/AuthScreen";
import { ChatThread } from "./components/ChatThread";
import { Sidebar } from "./components/Sidebar";
import { api, waitForRun } from "./lib/api";
import type {
  ChatMessage,
  Conversation,
  Health,
  InterpretationMode,
  RunView,
  UserAccount,
} from "./lib/types";

// 原因：文档、过程和模型设置不是默认聊天首屏的必需代码。
// 作用：仅在用户打开对应功能时加载模块，降低初始 JavaScript 解析成本。
const DocumentWorkspace = lazy(() =>
  import("./components/DocumentWorkspace").then((module) => ({
    default: module.DocumentWorkspace,
  })),
);
const ModelSettingsDialog = lazy(() =>
  import("./components/ModelSettingsDialog").then((module) => ({
    default: module.ModelSettingsDialog,
  })),
);
const RunInspector = lazy(() =>
  import("./components/RunInspector").then((module) => ({
    default: module.RunInspector,
  })),
);
const SkillWorkspace = lazy(() =>
  import("./components/SkillWorkspace").then((module) => ({
    default: module.SkillWorkspace,
  })),
);
const AccountDialog = lazy(() =>
  import("./components/AccountDialog").then((module) => ({
    default: module.AccountDialog,
  })),
);
const ShareDialog = lazy(() =>
  import("./components/ShareDialog").then((module) => ({
    default: module.ShareDialog,
  })),
);

type ViewMode = "chat" | "documents" | "skills";
type ResponseDetail = "concise" | "balanced" | "detailed";

export default function App() {
  const [authLoading, setAuthLoading] = useState(true);
  const [bootstrapRequired, setBootstrapRequired] = useState(false);
  const [user, setUser] = useState<UserAccount | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [mode, setMode] = useState<ViewMode>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [webSearch, setWebSearch] = useState(false);
  const [browserAccess, setBrowserAccess] = useState(false);
  const [localKnowledge, setLocalKnowledge] = useState(false);
  const [globalKnowledge, setGlobalKnowledge] = useState(false);
  const [minSourceRelevance, setMinSourceRelevance] = useState(0.55);
  const [responseDetail, setResponseDetail] = useState<ResponseDetail>("detailed");
  const [interpretationMode, setInterpretationMode] =
    useState<InterpretationMode>("contextual");
  const [showProcess, setShowProcess] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [runView, setRunView] = useState<RunView | null>(null);
  const [phase, setPhase] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [shareOpen, setShareOpen] = useState(false);

  const refreshConversations = useCallback(async () => {
    const items = await api.listConversations();
    setConversations(items);
    return items;
  }, []);

  const loadMessages = useCallback(async (conversationId: string) => {
    setMessages(await api.listMessages(conversationId));
  }, []);

  useEffect(() => {
    let active = true;
    const requireAuthentication = () => {
      setUser(null);
      setConversations([]);
      setActiveId(null);
      setMessages([]);
    };
    window.addEventListener("qwopus:auth-required", requireAuthentication);
    void api.authStatus()
      .then((status) => {
        if (!active) return;
        setBootstrapRequired(status.bootstrap_required);
        setUser(status.user ?? null);
      })
      .catch((reason) => {
        if (active) setError(toMessage(reason));
      })
      .finally(() => {
        if (active) setAuthLoading(false);
      });
    return () => {
      active = false;
      window.removeEventListener("qwopus:auth-required", requireAuthentication);
    };
  }, []);

  useEffect(() => {
    if (!user) return;
    let active = true;
    void (async () => {
      try {
        let items = await refreshConversations();
        if (!items.length) {
          const created = await api.createConversation();
          items = [created];
          setConversations(items);
        }
        if (active) setActiveId(items[0].id);
      } catch (reason) {
        if (active) setError(toMessage(reason));
      }
    })();
    void api.health().then(setHealth).catch(() => setHealth(null));
    return () => {
      active = false;
    };
  }, [refreshConversations, user]);

  useEffect(() => {
    if (!activeId) return;
    let active = true;
    void loadMessages(activeId).catch((reason) => {
      if (active) setError(toMessage(reason));
    });
    return () => {
      active = false;
    };
  }, [activeId, loadMessages]);

  const createConversation = useCallback(async () => {
    try {
      const conversation = await api.createConversation();
      setConversations((current) => [conversation, ...current]);
      setActiveId(conversation.id);
      setMessages([]);
      setMode("chat");
      setGlobalKnowledge(false);
      setSidebarOpen(false);
      setRunView(null);
      setError(null);
    } catch (reason) {
      setError(toMessage(reason));
    }
  }, []);

  const deleteConversation = useCallback(
    async (conversationId: string) => {
      if (isRunning && conversationId === activeId) {
        setError("Stop the current Agent run before deleting this conversation.");
        return;
      }
      try {
        await api.deleteConversation(conversationId);
        const remaining = conversations.filter((item) => item.id !== conversationId);
        setConversations(remaining);
        if (activeId === conversationId) {
          if (remaining.length) {
            setActiveId(remaining[0].id);
          } else {
            await createConversation();
          }
        }
      } catch (reason) {
        setError(toMessage(reason));
      }
    },
    [activeId, conversations, createConversation, isRunning],
  );

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeId || isRunning) return;
      const conversationId = activeId;
      const optimistic: ChatMessage = {
        id: `pending-${Date.now()}`,
        conversation_id: conversationId,
        role: "user",
        content,
        created_at: new Date().toISOString(),
      };
      setMessages((current) => [...current, optimistic]);
      setIsRunning(true);
      setError(null);
      setRunView(null);
      setPhase("Starting Agent");
      try {
        // 原因：正式前端必须经过统一 Orchestrator，不能直接向当前模型发送 Chat Completion。
        // 作用：联网、MiniRAG、图谱和 Multi-Agent 开关由后端规划器决定并留下同一份运行轨迹。
        const started = await api.startRun(conversationId, content, {
          enableWebSearch: webSearch,
          enableBrowser: browserAccess,
          enableLocalKnowledge: localKnowledge,
          includeGlobalKnowledge: globalKnowledge,
          minSourceRelevance,
          responseDetail,
          interpretationMode,
        });
        setRunId(started.run_id);
        await loadMessages(conversationId);
        const completed = await waitForRun(started.run_id, (current) => {
          setRunView(current);
          setPhase(formatPhase(current.phase));
        });
        if (completed.status === "failed") {
          throw new Error(completed.error ?? "Agent run failed");
        }
        await loadMessages(conversationId);
        await refreshConversations();
      } catch (reason) {
        setError(toMessage(reason));
        await loadMessages(conversationId).catch(() => undefined);
      } finally {
        setIsRunning(false);
        setRunId(null);
      }
    },
    [
      activeId,
      browserAccess,
      globalKnowledge,
      isRunning,
      interpretationMode,
      loadMessages,
      localKnowledge,
      minSourceRelevance,
      refreshConversations,
      responseDetail,
      webSearch,
    ],
  );

  const cancelRun = useCallback(async () => {
    if (!runId) return;
    const cancelled = await api.cancelRun(runId);
    setRunView(cancelled);
    setPhase("Cancelled");
    setIsRunning(false);
    setRunId(null);
  }, [runId]);

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } finally {
      setUser(null);
      setConversations([]);
      setActiveId(null);
      setMessages([]);
      setMode("chat");
      setAccountOpen(false);
      setShareOpen(false);
      setGlobalKnowledge(false);
    }
  }, []);

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeId) ?? null;

  if (authLoading) {
    return (
      <div className="workspace-loading app-loading">
        Loading Qwopus Agent
      </div>
    );
  }

  if (!user) {
    return (
      <AuthScreen
        bootstrapRequired={bootstrapRequired}
        onAuthenticated={(authenticated) => {
          setUser(authenticated);
          setBootstrapRequired(false);
          setError(null);
        }}
      />
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        health={health}
        user={user}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onCreate={() => void createConversation()}
        onSelect={(id) => {
          setActiveId(id);
          setMode("chat");
          setGlobalKnowledge(false);
          setSidebarOpen(false);
          setRunView(null);
        }}
        onDelete={(id) => void deleteConversation(id)}
        onConfigureModel={() => setModelSettingsOpen(true)}
        onOpenAccount={() => setAccountOpen(true)}
        onLogout={() => void logout()}
      />
      {sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      <main className="main-area">
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} title="Menu">
            <Menu size={19} />
          </button>
          <div className="mode-switch" role="tablist" aria-label="Workspace">
            <button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>
              <MessageCircle size={16} /> Chat
            </button>
            <button
              className={mode === "documents" ? "active" : ""}
              onClick={() => setMode("documents")}
            >
              <FileSearch size={16} /> Documents
            </button>
            {user.role === "admin" && (
              <button
                className={mode === "skills" ? "active" : ""}
                onClick={() => setMode("skills")}
              >
                <Layers3 size={16} /> Skills
              </button>
            )}
          </div>

          {activeConversation?.is_owner && (
            <button
              className="secondary-button share-chat-button"
              onClick={() => setShareOpen(true)}
              title="Manage chat access"
              type="button"
            >
              <Share2 size={15} />
              Share
              {activeConversation.shared_count > 0 && (
                <span>{activeConversation.shared_count}</span>
              )}
            </button>
          )}
          {activeConversation && !activeConversation.is_owner && (
            <span className="shared-chat-label">
              <Share2 size={14} />
              Shared by {activeConversation.owner_username}
            </span>
          )}

          {mode === "chat" && (
            <div className="capability-toggles">
              <label className="detail-select" title="Control answer depth">
                <TextSelect size={15} />
                <select
                  value={responseDetail}
                  onChange={(event) => setResponseDetail(event.target.value as ResponseDetail)}
                  aria-label="Answer detail"
                >
                  <option value="concise">Concise</option>
                  <option value="balanced">Balanced</option>
                  <option value="detailed">Detailed</option>
                </select>
              </label>
              <label
                className="detail-select"
                title="Control how broadly the Agent interprets your request"
              >
                <ScanText size={15} />
                <select
                  value={interpretationMode}
                  onChange={(event) =>
                    setInterpretationMode(event.target.value as InterpretationMode)
                  }
                  aria-label="Interpretation range"
                >
                  <option value="precise">Precise</option>
                  <option value="contextual">Context</option>
                  <option value="exploratory">Explore</option>
                </select>
              </label>
              <label title="Allow Tavily web search">
                <input type="checkbox" checked={webSearch} onChange={(event) => setWebSearch(event.target.checked)} />
                <Search size={15} /> Web
              </label>
              <label title="Allow isolated browser rendering for public pages">
                <input
                  type="checkbox"
                  checked={browserAccess}
                  onChange={(event) => setBrowserAccess(event.target.checked)}
                />
                <Monitor size={15} /> Browser
              </label>
              <label title="Allow MiniRAG and knowledge graph search">
                <input
                  type="checkbox"
                  checked={localKnowledge}
                  onChange={(event) => {
                    setLocalKnowledge(event.target.checked);
                    if (!event.target.checked) setGlobalKnowledge(false);
                  }}
                />
                <Network size={15} /> Knowledge
              </label>
              <label title="Allow knowledge from your account outside this conversation">
                <input
                  type="checkbox"
                  checked={globalKnowledge}
                  disabled={!localKnowledge}
                  onChange={(event) => setGlobalKnowledge(event.target.checked)}
                />
                <Globe2 size={15} /> Global
              </label>
              <label title="Show Agent execution trace">
                <input
                  type="checkbox"
                  checked={showProcess}
                  onChange={(event) => setShowProcess(event.target.checked)}
                />
                <Wrench size={15} /> Process
              </label>
            </div>
          )}
          {(mode === "documents" || localKnowledge) && (
            <label
              className="source-relevance"
              title="Only use local sources at or above this semantic similarity"
            >
              <SlidersHorizontal size={15} />
              <span>Sources</span>
              <input
                type="range"
                min="25"
                max="95"
                step="5"
                value={Math.round(minSourceRelevance * 100)}
                onChange={(event) => setMinSourceRelevance(Number(event.target.value) / 100)}
                aria-label="Minimum local source relevance"
              />
              <output>{Math.round(minSourceRelevance * 100)}%</output>
            </label>
          )}
        </header>

        {error && (
          <div className="error-banner global-error">
            <span>{error}</span>
            <button className="icon-button" onClick={() => setError(null)} title="Dismiss">
              <X size={16} />
            </button>
          </div>
        )}

        {mode === "chat" ? (
          <section className="chat-workspace">
            <AgentRuntimeProvider
              key={activeId ?? "empty"}
              messages={messages}
              isRunning={isRunning}
              onSend={sendMessage}
              onCancel={cancelRun}
            >
              <ChatThread />
            </AgentRuntimeProvider>
            {showProcess && (
              <Suspense fallback={null}>
                <RunInspector run={runView} phase={phase} visible />
              </Suspense>
            )}
          </section>
        ) : mode === "documents" ? (
          <Suspense fallback={<div className="workspace-loading">Loading documents...</div>}>
            <DocumentWorkspace
              key={activeId ?? "empty"}
              conversationId={activeId}
              minSourceRelevance={minSourceRelevance}
              canUseLocalFolder={user.role === "admin"}
            />
          </Suspense>
        ) : (
          <Suspense fallback={<div className="workspace-loading">Loading Skills...</div>}>
            <SkillWorkspace />
          </Suspense>
        )}
      </main>
      {modelSettingsOpen && (
        <Suspense fallback={null}>
          <ModelSettingsDialog
            open
            health={health}
            onClose={() => setModelSettingsOpen(false)}
            onSaved={setHealth}
          />
        </Suspense>
      )}
      {accountOpen && (
        <Suspense fallback={null}>
          <AccountDialog
            user={user}
            onClose={() => setAccountOpen(false)}
            onUserChanged={setUser}
          />
        </Suspense>
      )}
      {shareOpen && activeConversation?.is_owner && (
        <Suspense fallback={null}>
          <ShareDialog
            conversation={activeConversation}
            onClose={() => setShareOpen(false)}
            onChanged={async () => {
              await refreshConversations();
            }}
          />
        </Suspense>
      )}
    </div>
  );
}

function toMessage(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Unexpected error";
}

function formatPhase(phase: string): string {
  const labels: Record<string, string> = {
    connecting: "Connecting to model",
    planning: "Planning task",
    executing: "Executing skills",
    synthesizing: "Writing final answer",
    completed: "Completed",
  };
  return labels[phase] ?? phase.replaceAll("_", " ");
}

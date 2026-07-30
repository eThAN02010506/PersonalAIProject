import {
  Bot,
  Bug,
  KeyRound,
  LogOut,
  MessageSquare,
  Plus,
  Settings2,
  Share2,
  Trash2,
  UserRound,
  X,
} from "lucide-react";

import type {
  Conversation,
  Health,
  UserAccount,
  WebSearchSettings,
} from "../lib/types";

type SidebarProps = {
  conversations: Conversation[];
  activeId: string | null;
  health: Health | null;
  user: UserAccount;
  webSearchSettings: WebSearchSettings | null;
  open: boolean;
  onClose: () => void;
  onCreate: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onConfigureModel: () => void;
  onConfigureWebSearch: () => void;
  onOpenAccount: () => void;
  onLogout: () => void;
};

export function Sidebar({
  conversations,
  activeId,
  health,
  user,
  webSearchSettings,
  open,
  onClose,
  onCreate,
  onSelect,
  onDelete,
  onConfigureModel,
  onConfigureWebSearch,
  onOpenAccount,
  onLogout,
}: SidebarProps) {
  return (
    <aside className={`sidebar ${open ? "open" : ""}`}>
      <div className="brand-row">
        <div className="brand-mark"><Bot size={19} /></div>
        <strong>Qwopus Agent</strong>
        <button className="icon-button mobile-only" onClick={onClose} title="Close sidebar">
          <X size={18} />
        </button>
      </div>

      <button className="new-chat-button" onClick={onCreate}>
        <Plus size={17} />
        New chat
      </button>

      <nav className="conversation-list" aria-label="Conversations">
        {conversations.map((conversation) => (
          <div
            className={`conversation-row ${activeId === conversation.id ? "active" : ""}`}
            key={conversation.id}
          >
            <button className="conversation-select" onClick={() => onSelect(conversation.id)}>
              {conversation.is_owner ? <MessageSquare size={15} /> : <Share2 size={15} />}
              <span>{conversation.title}</span>
            </button>
            {conversation.is_owner && (
              <button
                className="icon-button conversation-delete"
                onClick={() => onDelete(conversation.id)}
                title="Delete conversation"
              >
                <Trash2 size={14} />
              </button>
            )}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="account-summary">
          <button onClick={onOpenAccount} title="Account settings" type="button">
            <UserRound size={16} />
            <span>
              <strong>{user.display_name}</strong>
              <small>@{user.username}</small>
            </span>
          </button>
          <button className="icon-button" onClick={onLogout} title="Sign out" type="button">
            <LogOut size={15} />
          </button>
        </div>
        <div className="model-status-row">
          <div className="model-status" title={health?.message ?? "Checking model"}>
            <span className={`status-dot ${health?.model_online ? "online" : ""}`} />
            <span>{health?.model ?? "Checking model"}</span>
          </div>
          {user.role === "admin" && (
            <button
              className="icon-button model-settings-button"
              type="button"
              onClick={onConfigureModel}
              title="Model connection"
            >
              <Settings2 size={15} />
            </button>
          )}
        </div>
        {user.role === "admin" && (
          <div className="model-status-row">
            <div
              className="model-status"
              title={webSearchSettings?.message ?? "Checking Tavily"}
            >
              <span
                className={`status-dot ${webSearchSettings?.configured ? "online" : ""}`}
              />
              <span>
                {webSearchSettings?.configured ? "Tavily search" : "Tavily not configured"}
              </span>
            </div>
            <button
              className="icon-button model-settings-button"
              type="button"
              onClick={onConfigureWebSearch}
              title="Web search settings"
            >
              <KeyRound size={15} />
            </button>
          </div>
        )}
        {user.role === "admin" && (
          <a
            className="debug-link"
            href="/debug"
            target="_blank"
            rel="noreferrer"
          >
            <Bug size={15} />
            Debug console
          </a>
        )}
      </div>
    </aside>
  );
}

import { Bot, Bug, MessageSquare, Plus, Trash2, X } from "lucide-react";

import type { Conversation, Health } from "../lib/types";

type SidebarProps = {
  conversations: Conversation[];
  activeId: string | null;
  health: Health | null;
  open: boolean;
  onClose: () => void;
  onCreate: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
};

export function Sidebar({
  conversations,
  activeId,
  health,
  open,
  onClose,
  onCreate,
  onSelect,
  onDelete,
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
              <MessageSquare size={15} />
              <span>{conversation.title}</span>
            </button>
            <button
              className="icon-button conversation-delete"
              onClick={() => onDelete(conversation.id)}
              title="Delete conversation"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status" title={health?.message ?? "Checking model"}>
          <span className={`status-dot ${health?.model_online ? "online" : ""}`} />
          <span>{health?.model ?? "Checking model"}</span>
        </div>
        <a
          className="debug-link"
          href="http://localhost:8502"
          target="_blank"
          rel="noreferrer"
        >
          <Bug size={15} />
          Streamlit debug console
        </a>
      </div>
    </aside>
  );
}

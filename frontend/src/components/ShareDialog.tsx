import { LoaderCircle, Share2, Trash2, UserRound, X } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { Conversation, ConversationMember } from "../lib/types";

type ShareDialogProps = {
  conversation: Conversation;
  onClose: () => void;
  onChanged: () => Promise<void>;
};

export function ShareDialog({
  conversation,
  onClose,
  onChanged,
}: ShareDialogProps) {
  const [members, setMembers] = useState<ConversationMember[]>([]);
  const [username, setUsername] = useState("");
  const [busy, setBusy] = useState<string | null>("load");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setMembers(await api.listConversationMembers(conversation.id));
  }, [conversation.id]);

  useEffect(() => {
    void refresh()
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : "Could not load access");
      })
      .finally(() => setBusy(null));
  }, [refresh]);

  const share = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy("share");
    setError(null);
    try {
      await api.shareConversation(conversation.id, username);
      setUsername("");
      await refresh();
      await onChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not share conversation");
    } finally {
      setBusy(null);
    }
  };

  const remove = async (member: ConversationMember) => {
    setBusy(member.user_id);
    setError(null);
    try {
      await api.unshareConversation(conversation.id, member.user_id);
      await refresh();
      await onChanged();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not revoke access");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="dialog-backdrop" onMouseDown={onClose}>
      <section
        className="model-dialog share-dialog"
        onMouseDown={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="share-title"
      >
        <header className="dialog-header">
          <div>
            <h2 id="share-title">Share chat</h2>
            <span>{conversation.title}</span>
          </div>
          <button className="icon-button" onClick={onClose} title="Close" type="button">
            <X size={18} />
          </button>
        </header>
        <div className="share-dialog-body">
          <form className="share-form" onSubmit={(event) => void share(event)}>
            <label>
              <span>Username</span>
              <input
                autoFocus
                onChange={(event) => setUsername(event.target.value)}
                placeholder="account-name"
                required
                value={username}
              />
            </label>
            <button className="primary-button" disabled={busy !== null} type="submit">
              {busy === "share"
                ? <LoaderCircle className="spin" size={15} />
                : <Share2 size={15} />}
              Share
            </button>
          </form>
          <div className="share-member-list">
            {members.map((member) => (
              <div className="share-member-row" key={member.user_id}>
                <UserRound size={16} />
                <div>
                  <strong>{member.display_name}</strong>
                  <small>@{member.username} · {member.access}</small>
                </div>
                {member.access === "member" && (
                  <button
                    className="icon-button"
                    disabled={busy !== null}
                    onClick={() => void remove(member)}
                    title="Revoke access"
                    type="button"
                  >
                    {busy === member.user_id
                      ? <LoaderCircle className="spin" size={14} />
                      : <Trash2 size={14} />}
                  </button>
                )}
              </div>
            ))}
          </div>
          {error && <div className="error-banner">{error}</div>}
        </div>
      </section>
    </div>
  );
}

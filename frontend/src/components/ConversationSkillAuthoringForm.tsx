import { LoaderCircle, MessagesSquare, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../lib/api";
import type {
  SkillCandidateReview,
  SkillSourceConversation,
  SkillSourceRun,
} from "../lib/types";

type ConversationSkillAuthoringFormProps = {
  onGenerated: (review: SkillCandidateReview) => void;
};

export function ConversationSkillAuthoringForm({
  onGenerated,
}: ConversationSkillAuthoringFormProps) {
  const [conversations, setConversations] = useState<SkillSourceConversation[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [runs, setRuns] = useState<SkillSourceRun[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [requestedName, setRequestedName] = useState("");
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void api.listSkillSourceConversations()
      .then((items) => {
        if (!active) return;
        setConversations(items);
        setConversationId(items[0]?.id ?? "");
      })
      .catch((reason) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load conversations");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!conversationId) {
      setRuns([]);
      setSelectedRunIds([]);
      return;
    }
    let active = true;
    setLoading(true);
    setSelectedRunIds([]);
    void api.listSkillSourceRuns(conversationId)
      .then((items) => {
        if (active) setRuns(items);
      })
      .catch((reason) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load runs");
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [conversationId]);

  const selectedSignature = useMemo(() => {
    const first = runs.find((run) => selectedRunIds.includes(run.run_id));
    return first?.reusable_skills.join(" → ") ?? "";
  }, [runs, selectedRunIds]);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!conversationId || !selectedRunIds.length || generating) return;
    setGenerating(true);
    setError(null);
    try {
      onGenerated(
        await api.generateSkillCandidateFromRuns({
          conversationId,
          runIds: selectedRunIds,
          requestedName: requestedName.trim() || undefined,
        }),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Skill generation failed");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <form className="skill-authoring-form" onSubmit={(event) => void submit(event)}>
      <header>
        <div>
          <h2>Generate from Conversation</h2>
          <span>Sanitized run trace · review required</span>
        </div>
        <MessagesSquare size={18} />
      </header>

      {error && <div className="error-banner">{error}</div>}
      <div className="skill-source-fields">
        <label>
          <span>Conversation</span>
          <select
            value={conversationId}
            onChange={(event) => setConversationId(event.target.value)}
            disabled={loading && !conversations.length}
          >
            {!conversations.length && <option value="">No reusable runs yet</option>}
            {conversations.map((conversation) => (
              <option key={conversation.id} value={conversation.id}>
                {conversation.owner_username
                  ? `@${conversation.owner_username} · `
                  : ""}
                {conversation.title}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Preferred name</span>
          <input
            value={requestedName}
            onChange={(event) => setRequestedName(event.target.value)}
            maxLength={90}
            pattern="[a-zA-Z0-9_]*"
            placeholder="sourced_research"
          />
        </label>
      </div>

      <fieldset className="skill-run-picker" disabled={loading}>
        <legend>Compatible successful runs</legend>
        <div>
          {runs.map((run) => {
            const signature = run.reusable_skills.join(" → ");
            const checked = selectedRunIds.includes(run.run_id);
            const incompatible = Boolean(selectedSignature && selectedSignature !== signature);
            return (
              <label className={incompatible ? "disabled" : ""} key={run.run_id}>
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={incompatible}
                  onChange={(event) => {
                    setSelectedRunIds((current) => (
                      event.target.checked
                        ? [...current, run.run_id]
                        : current.filter((runId) => runId !== run.run_id)
                    ));
                  }}
                />
                <span>
                  <strong>{run.objective}</strong>
                  <code>{signature}</code>
                  <small>{run.answer_preview || run.operational_objective}</small>
                  <small>{run.model_id}</small>
                </span>
              </label>
            );
          })}
          {!loading && !runs.length && (
            <p>No successful runs contain a reusable Skill sequence.</p>
          )}
        </div>
      </fieldset>

      <footer>
        <span>{selectedRunIds.length} run(s) selected</span>
        <button
          className="primary-button"
          disabled={generating || !selectedRunIds.length}
          type="submit"
        >
          {generating ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
          Generate candidate
        </button>
      </footer>
    </form>
  );
}

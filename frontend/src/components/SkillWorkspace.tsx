import {
  Ban,
  Check,
  Layers3,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { api } from "../lib/api";
import type { SkillVersion } from "../lib/types";

type SkillAction = "promote" | "reject" | "rollback";

export function SkillWorkspace() {
  const [skills, setSkills] = useState<SkillVersion[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeAction, setActiveAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSkills(await api.listSkills());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load Skills");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const applyAction = async (
    skill: SkillVersion,
    action: SkillAction,
  ) => {
    const actionId = `${skill.name}@${skill.version}:${action}`;
    setActiveAction(actionId);
    setError(null);
    try {
      if (action === "promote") {
        await api.promoteSkill(skill.name, skill.version);
      } else if (action === "reject") {
        await api.rejectSkill(skill.name, skill.version);
      } else {
        await api.rollbackSkill(skill.name, skill.version);
      }
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Skill action failed");
    } finally {
      setActiveAction(null);
    }
  };

  const ordered = [...skills].sort((left, right) => {
    const statusOrder = { candidate: 0, active: 1, archived: 2, rejected: 3 };
    return (
      statusOrder[left.status] - statusOrder[right.status]
      || left.name.localeCompare(right.name)
      || right.version.localeCompare(left.version, undefined, { numeric: true })
    );
  });

  return (
    <section className="skill-workspace">
      <header className="workspace-heading">
        <div>
          <h1>Reusable Skills</h1>
          <p>Review · promote · version · rollback</p>
        </div>
        <button
          className="secondary-button skill-refresh"
          disabled={loading}
          onClick={() => void refresh()}
          type="button"
        >
          <RefreshCw className={loading ? "spin" : ""} size={15} />
          Refresh
        </button>
      </header>

      {error && <div className="error-banner skill-error">{error}</div>}
      {loading && !skills.length && (
        <div className="workspace-loading">
          <LoaderCircle className="spin" size={18} />
          Loading Skills
        </div>
      )}
      {!loading && !ordered.length && (
        <div className="skill-empty">
          <Layers3 size={22} />
          <strong>No learned Skill versions yet</strong>
          <span>Candidates appear after the same safe Tool workflow succeeds twice.</span>
        </div>
      )}

      <div className="skill-version-list">
        {ordered.map((skill) => {
          const busy = activeAction?.startsWith(`${skill.name}@${skill.version}:`);
          return (
            <article className="skill-version-row" key={`${skill.name}@${skill.version}`}>
              <div className="skill-version-main">
                <div>
                  <strong>{skill.name}</strong>
                  <code>{skill.version}</code>
                </div>
                <p>{skill.description}</p>
              </div>
              <span className={`skill-status ${skill.status}`}>{skill.status}</span>
              <div className="skill-version-actions">
                {skill.status === "candidate" && (
                  <>
                    <button
                      className="primary-button"
                      disabled={busy || !skill.spec_valid}
                      onClick={() => void applyAction(skill, "promote")}
                      type="button"
                    >
                      {busy ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}
                      Promote
                    </button>
                    <button
                      className="secondary-button"
                      disabled={busy}
                      onClick={() => void applyAction(skill, "reject")}
                      type="button"
                    >
                      <Ban size={14} />
                      Reject
                    </button>
                  </>
                )}
                {skill.status === "archived" && (
                  <button
                    className="secondary-button"
                    disabled={busy || !skill.spec_valid}
                    onClick={() => void applyAction(skill, "rollback")}
                    type="button"
                  >
                    {busy ? <LoaderCircle className="spin" size={14} /> : <RotateCcw size={14} />}
                    Roll back
                  </button>
                )}
              </div>
              <details className="skill-version-detail">
                <summary>Review evidence and workflow</summary>
                <dl>
                  <div>
                    <dt>Integrity</dt>
                    <dd>{skill.spec_valid ? "Valid" : "Invalid"}</dd>
                  </div>
                  <div>
                    <dt>Source run</dt>
                    <dd><code>{skill.source_run_id ?? "Unknown"}</code></dd>
                  </div>
                  <div>
                    <dt>Steps</dt>
                    <dd>{skill.steps.map((step) => step.skill_name).join(" → ") || "Unavailable"}</dd>
                  </div>
                </dl>
                {skill.intent_examples.length > 0 && (
                  <div className="skill-intents">
                    <strong>Validated intent examples</strong>
                    <ul>
                      {skill.intent_examples.map((example) => (
                        <li key={example}>{example}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </details>
            </article>
          );
        })}
      </div>
    </section>
  );
}

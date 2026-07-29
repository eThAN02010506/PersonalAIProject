import { LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { api } from "../lib/api";
import type { SkillCandidateReview, SkillCapability } from "../lib/types";

type SkillAuthoringFormProps = {
  onGenerated: (review: SkillCandidateReview) => void;
};

export function SkillAuthoringForm({ onGenerated }: SkillAuthoringFormProps) {
  const [capabilities, setCapabilities] = useState<SkillCapability[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [goal, setGoal] = useState("");
  const [requestedName, setRequestedName] = useState("");
  const [examples, setExamples] = useState("");
  const [loadingCapabilities, setLoadingCapabilities] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void api.listSkillCapabilities()
      .then((items) => {
        if (active) setCapabilities(items);
      })
      .catch((reason) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load capabilities");
        }
      })
      .finally(() => {
        if (active) setLoadingCapabilities(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!goal.trim() || !selected.length || generating) return;
    setGenerating(true);
    setError(null);
    try {
      const review = await api.generateSkillCandidate({
        goal: goal.trim(),
        requestedName: requestedName.trim() || undefined,
        intentExamples: examples
          .split("\n")
          .map((value) => value.trim())
          .filter(Boolean),
        allowedSkills: selected,
      });
      onGenerated(review);
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
          <h2>Generate Workflow Candidate</h2>
          <span>Model-authored · review required</span>
        </div>
        <Sparkles size={18} />
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="skill-authoring-fields">
        <label className="skill-goal-field">
          <span>Goal</span>
          <textarea
            value={goal}
            onChange={(event) => setGoal(event.target.value)}
            maxLength={2_000}
            placeholder="Prepare a sourced report from the selected documents"
            required
          />
        </label>
        <label>
          <span>Preferred name</span>
          <input
            value={requestedName}
            onChange={(event) => setRequestedName(event.target.value)}
            maxLength={90}
            pattern="[a-zA-Z0-9_]*"
            placeholder="document_report"
          />
        </label>
        <label>
          <span>Intent examples</span>
          <textarea
            value={examples}
            onChange={(event) => setExamples(event.target.value)}
            placeholder={"Summarize these files\nPrepare the recurring report"}
          />
        </label>
      </div>

      <fieldset className="skill-capability-picker" disabled={loadingCapabilities}>
        <legend>Allowed Skills</legend>
        <div>
          {capabilities.map((capability) => (
            <label key={capability.name} title={capability.description}>
              <input
                type="checkbox"
                checked={selected.includes(capability.name)}
                onChange={(event) => {
                  setSelected((current) => (
                    event.target.checked
                      ? [...current, capability.name]
                      : current.filter((name) => name !== capability.name)
                  ));
                }}
              />
              <span>
                <strong>{capability.name}</strong>
                <small>{capability.description}</small>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <footer>
        <span>{selected.length} allowed</span>
        <button
          className="primary-button"
          disabled={generating || !goal.trim() || !selected.length}
          type="submit"
        >
          {generating ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
          Generate candidate
        </button>
      </footer>
    </form>
  );
}

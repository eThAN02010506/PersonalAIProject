import {
  CheckCircle2,
  FlaskConical,
  LoaderCircle,
  X,
} from "lucide-react";
import { useState } from "react";

import { api } from "../lib/api";
import type { SkillCandidateReview, SkillCandidateTest } from "../lib/types";

type SkillCandidateReviewProps = {
  review: SkillCandidateReview;
  onClose: () => void;
};

export function SkillCandidateReviewPanel({
  review,
  onClose,
}: SkillCandidateReviewProps) {
  const [query, setQuery] = useState("");
  const [testResult, setTestResult] = useState<SkillCandidateTest | null>(null);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runDryTest = async () => {
    if (!query.trim() || testing) return;
    setTesting(true);
    setError(null);
    try {
      setTestResult(
        await api.testSkillCandidate(
          review.skill.name,
          review.skill.version,
          query.trim(),
        ),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Candidate test failed");
    } finally {
      setTesting(false);
    }
  };

  return (
    <section className="skill-candidate-review">
      <header>
        <div>
          <h2>{review.skill.name}</h2>
          <span>
            {review.skill.version} · {review.skill.source_model ?? "Unknown model"}
          </span>
        </div>
        <button className="icon-button" onClick={onClose} title="Close review" type="button">
          <X size={16} />
        </button>
      </header>

      <div className="skill-checks">
        {review.checks.map((check) => (
          <div key={check.name}>
            <CheckCircle2 size={14} />
            <span>
              <strong>{check.name}</strong>
              <small>{check.detail}</small>
            </span>
          </div>
        ))}
      </div>

      <div className="skill-dry-run">
        <label>
          <span>Dry-run query</span>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            maxLength={2_000}
            placeholder="quarterly sales.xlsx"
          />
        </label>
        <button
          className="secondary-button"
          disabled={!query.trim() || testing}
          onClick={() => void runDryTest()}
          type="button"
        >
          {testing
            ? <LoaderCircle className="spin" size={14} />
            : <FlaskConical size={14} />}
          Test candidate
        </button>
      </div>
      {error && <div className="error-banner">{error}</div>}
      {testResult && (
        <div className={`skill-test-result ${testResult.success ? "passed" : "failed"}`}>
          <strong>{testResult.success ? "Dry run passed" : "Dry run failed"}</strong>
          {testResult.steps.map((step, index) => (
            <div key={`${step.skill_name}-${index}`}>
              <code>{step.skill_name}</code>
              <span>{step.query}</span>
              <small>
                {step.argument_keys.length
                  ? `arguments: ${step.argument_keys.join(", ")}`
                  : "no persisted arguments"}
              </small>
            </div>
          ))}
        </div>
      )}

      <details open>
        <summary>Version diff</summary>
        <pre>{review.diff || "No previous version."}</pre>
      </details>
      <details>
        <summary>Persisted WorkflowSpec</summary>
        <pre>{review.spec_json}</pre>
      </details>
      {review.model_output && (
        <details>
          <summary>Raw model output</summary>
          <pre>{review.model_output}</pre>
        </details>
      )}
    </section>
  );
}

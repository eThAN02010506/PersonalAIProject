import { ChevronDown, ListTree } from "lucide-react";

import type { RunView } from "../lib/types";

type RunInspectorProps = {
  run: RunView | null;
  phase: string;
  visible: boolean;
};

export function RunInspector({ run, phase, visible }: RunInspectorProps) {
  if (!visible || (!run && !phase)) return null;
  return (
    <details className="run-inspector" open={run?.status === "running"}>
      <summary>
        <ListTree size={16} />
        <span>{phase || run?.phase || "Agent process"}</span>
        <ChevronDown size={15} className="details-chevron" />
      </summary>
      <div className="trace-list">
        {(run?.trace ?? []).map((entry, index) => (
          <pre key={index}>{JSON.stringify(entry, null, 2)}</pre>
        ))}
        {run?.citations?.length ? (
          <pre>{JSON.stringify({ citations: run.citations }, null, 2)}</pre>
        ) : null}
        {!run?.trace?.length && <p>{phase}</p>}
      </div>
    </details>
  );
}

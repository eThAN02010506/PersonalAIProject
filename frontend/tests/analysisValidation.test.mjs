import assert from "node:assert/strict";
import test from "node:test";

import { getAnalysisInputError } from "../src/lib/analysisValidation.ts";

test("question analysis requires non-whitespace input", () => {
  assert.match(getAnalysisInputError("question", "", {}), /Enter a question/);
  assert.match(getAnalysisInputError("question", "   ", {}), /Enter a question/);
  assert.equal(getAnalysisInputError("question", " Compare evidence ", {}), null);
});

test("full analysis allows an empty optional question", () => {
  assert.equal(getAnalysisInputError("full", "", {}), null);
});

test("section analysis requires at least one selected section", () => {
  assert.match(getAnalysisInputError("section", "", {}), /Select at least one/);
  assert.match(
    getAnalysisInputError("section", "", { document: [] }),
    /Select at least one/,
  );
  assert.equal(
    getAnalysisInputError("section", "", { document: ["section-1"] }),
    null,
  );
});

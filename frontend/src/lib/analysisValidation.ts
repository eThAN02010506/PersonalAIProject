export type AnalysisMode = "question" | "section" | "full";

export function getAnalysisInputError(
  mode: AnalysisMode,
  question: string,
  selectedSections: Record<string, string[]>,
): string | null {
  // 原因：三种分析模式对空问题的含义不同，统一 required 会误伤全文和章节分析。
  // 作用：前端即时遵循服务端契约；服务端仍是拒绝非法请求的最终边界。
  if (mode === "question" && !question.trim()) {
    return "Enter a question before starting question-based analysis.";
  }
  if (
    mode === "section"
    && !Object.values(selectedSections).some((sectionIds) =>
      sectionIds.some((sectionId) => Boolean(sectionId.trim()))
    )
  ) {
    return "Select at least one document section before starting section analysis.";
  }
  return null;
}

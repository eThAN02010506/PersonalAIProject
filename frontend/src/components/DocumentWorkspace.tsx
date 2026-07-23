import {
  BarChart3,
  Download,
  FileText,
  ListTree,
  LoaderCircle,
  Upload,
  X,
} from "lucide-react";
import { type ChangeEvent, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api } from "../lib/api";
import type { AnalysisResult } from "../lib/types";

export function DocumentWorkspace({ minSourceRelevance }: { minSourceRelevance: number }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [question, setQuestion] = useState("");
  const [generateReport, setGenerateReport] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<"question" | "section" | "full">("question");
  const [selectedSections, setSelectedSections] = useState<Record<string, string[]>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const addFiles = (event: ChangeEvent<HTMLInputElement>) => {
    const incoming = Array.from(event.target.files ?? []);
    setFiles((current) => [...current, ...incoming]);
    setResult(null);
    setSelectedSections({});
    setAnalysisMode("question");
    event.target.value = "";
  };

  const removeFile = (index: number) => {
    // 原因：目录和 section id 绑定当前文件版本，删除文件后继续提交旧 id 会产生错误范围。
    // 作用：文件集合改变时清除派生结果，并要求用户重新解析新集合的目录。
    setFiles((current) => current.filter((_, item) => item !== index));
    setResult(null);
    setSelectedSections({});
    setAnalysisMode("question");
  };

  const analyze = async () => {
    if (!files.length || isRunning) return;
    setIsRunning(true);
    setError(null);
    try {
      // 原因：MinerU、Excel 沙箱、MiniRAG 和报告逻辑属于 Python 业务层。
      // 作用：浏览器只上传原文件并显示结果，避免前端产生另一套不一致的解析结果。
      setResult(
        await api.analyze(
          files,
          question,
          generateReport,
          minSourceRelevance,
          analysisMode,
          selectedSections,
        ),
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <section className="document-workspace">
      <header className="workspace-heading">
        <div>
          <h1>Document analysis</h1>
          <p>MinerU · pandas sandbox · MiniRAG</p>
        </div>
        <button className="primary-button" onClick={() => inputRef.current?.click()}>
          <Upload size={16} />
          Add files
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          accept=".pdf,.docx,.md,.txt,.xlsx,.xls,.csv,.png,.jpg,.jpeg"
          onChange={addFiles}
        />
      </header>

      <div className="document-controls">
        <div className="file-list">
          {files.map((file, index) => (
            <div className="file-row" key={`${file.name}-${index}`}>
              <FileText size={17} />
              <span>{file.name}</span>
              <small>{formatBytes(file.size)}</small>
              <button
                className="icon-button"
                onClick={() => removeFile(index)}
                title="Remove file"
              >
                <X size={15} />
              </button>
            </div>
          ))}
          {!files.length && <div className="empty-files">No files selected</div>}
        </div>

        <label className="field-label" htmlFor="analysis-question">Question</label>
        <div className="analysis-mode" aria-label="Analysis scope">
          {(["question", "section", "full"] as const).map((mode) => (
            <button
              className={analysisMode === mode ? "active" : ""}
              disabled={mode === "section" && !result?.documents.length}
              key={mode}
              onClick={() => setAnalysisMode(mode)}
              type="button"
            >
              {mode === "question" ? "Specific question" : mode === "section" ? "Sections" : "Full summary"}
            </button>
          ))}
        </div>
        <textarea
          id="analysis-question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={
            analysisMode === "question"
              ? "Ask about the uploaded files"
              : "Optional instructions for the summary"
          }
          rows={4}
        />
        <div className="analysis-actions">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={generateReport}
              onChange={(event) => setGenerateReport(event.target.checked)}
            />
            <BarChart3 size={16} />
            Generate report
          </label>
          <button className="primary-button" disabled={!files.length || isRunning} onClick={analyze}>
            {isRunning ? <LoaderCircle className="spin" size={16} /> : <FileText size={16} />}
            {isRunning ? "Analyzing" : "Analyze"}
          </button>
        </div>
        {error && <div className="error-banner">{error}</div>}
      </div>

      {result && (
        <>
          {result.documents.length > 0 && (
            <section className="document-outlines">
              <div className="outline-heading">
                <ListTree size={17} />
                <h2>Document outline</h2>
              </div>
              {result.documents.map((document) => (
                <div className="outline-document" key={document.document_id}>
                  <h3>{document.source}</h3>
                  {document.sections.map((section) => (
                    <label
                      className="outline-row"
                      key={section.id}
                      style={{ paddingLeft: `${Math.max(section.level - 1, 0) * 18}px` }}
                    >
                      <input
                        type="checkbox"
                        checked={(selectedSections[document.document_id] ?? []).includes(section.id)}
                        onChange={(event) => {
                          setSelectedSections((current) => {
                            const selected = new Set(current[document.document_id] ?? []);
                            if (event.target.checked) selected.add(section.id);
                            else selected.delete(section.id);
                            return { ...current, [document.document_id]: Array.from(selected) };
                          });
                        }}
                      />
                      <span>{section.title}</span>
                      {section.page_start && <small>{pageLabel(section.page_start, section.page_end)}</small>}
                    </label>
                  ))}
                </div>
              ))}
            </section>
          )}
          <article className="analysis-result">
            <div className="result-meta">
              <span>{result.route}</span>
              <span>{result.citations.length} citations</span>
            </div>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
            {result.reports.length > 0 && (
              <div className="report-downloads">
                {result.reports.map((report) => (
                  <a href={report.url} key={report.url} download>
                    <Download size={15} />
                    {report.name}
                  </a>
                ))}
              </div>
            )}
          </article>
        </>
      )}
    </section>
  );
}

function pageLabel(start: number, end?: number): string {
  return end && end !== start ? `pp. ${start}-${end}` : `p. ${start}`;
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

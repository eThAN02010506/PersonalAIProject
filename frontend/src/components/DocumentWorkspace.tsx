import {
  BarChart3,
  Database,
  Download,
  Folder,
  FileText,
  ListTree,
  LoaderCircle,
  RefreshCw,
  Search,
  TableProperties,
  Upload,
  X,
} from "lucide-react";
import { type ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api } from "../lib/api";
import { getAnalysisInputError } from "../lib/analysisValidation";
import type {
  AnalysisResult,
  LocalFolderNode,
  LocalFolderTree,
  SavedDocument,
} from "../lib/types";

type DocumentWorkspaceProps = {
  conversationId: string | null;
  minSourceRelevance: number;
  canUseLocalFolder: boolean;
  responseDetail: "concise" | "balanced" | "detailed";
};

export function DocumentWorkspace({
  conversationId,
  minSourceRelevance,
  canUseLocalFolder,
  responseDetail,
}: DocumentWorkspaceProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [sourceMode, setSourceMode] = useState<"upload" | "folder">("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [folderPath, setFolderPath] = useState("");
  const [folderTree, setFolderTree] = useState<LocalFolderTree | null>(null);
  const [selectedLocalFiles, setSelectedLocalFiles] = useState<Set<string>>(new Set());
  const [isScanning, setIsScanning] = useState(false);
  const [question, setQuestion] = useState("");
  const [generateReport, setGenerateReport] = useState(false);
  const [analysisMode, setAnalysisMode] = useState<"question" | "section" | "full">("question");
  const [selectedSections, setSelectedSections] = useState<Record<string, string[]>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedDocuments, setSavedDocuments] = useState<SavedDocument[]>([]);
  const [selectedSavedDocuments, setSelectedSavedDocuments] = useState<Set<string>>(
    new Set(),
  );
  const [isLoadingSaved, setIsLoadingSaved] = useState(true);
  const [isAttachingSaved, setIsAttachingSaved] = useState(false);
  const [savedNotice, setSavedNotice] = useState<string | null>(null);
  const [savedError, setSavedError] = useState<string | null>(null);
  const analysisInputError = getAnalysisInputError(
    analysisMode,
    question,
    selectedSections,
  );

  const loadSavedDocuments = useCallback(async () => {
    setIsLoadingSaved(true);
    setSavedError(null);
    try {
      const documents = await api.listDocuments();
      setSavedDocuments(documents);
      const availableIds = new Set(documents.map((document) => document.document_id));
      setSelectedSavedDocuments(
        (current) => new Set(Array.from(current).filter((id) => availableIds.has(id))),
      );
    } catch (reason) {
      setSavedError(reason instanceof Error ? reason.message : "Could not load saved documents");
    } finally {
      setIsLoadingSaved(false);
    }
  }, []);

  useEffect(() => {
    void loadSavedDocuments();
  }, [loadSavedDocuments]);

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

  const changeSourceMode = (mode: "upload" | "folder") => {
    setSourceMode(mode);
    setResult(null);
    setError(null);
    setSelectedSections({});
    setAnalysisMode("question");
  };

  const scanFolder = async () => {
    if (!folderPath.trim() || isScanning) return;
    setIsScanning(true);
    setError(null);
    setResult(null);
    setSelectedSections({});
    try {
      // 原因：macOS 允许目录名以空格结尾，trim() 会把合法路径改成另一个不存在的路径。
      // 作用：仅用 trim() 判断空输入，提交时完整保留用户给出的本地路径。
      const tree = await api.scanLocalFolder(folderPath);
      setFolderTree(tree);
      // 原因：目录可展示的文件数高于一次 Agent 能可靠覆盖的来源数。
      // 作用：小目录默认全选，大目录只预选后端允许的数量，用户仍可更换具体文件。
      setSelectedLocalFiles(
        new Set(collectFilePaths(tree.tree).slice(0, tree.max_selection)),
      );
    } catch (reason) {
      setFolderTree(null);
      setSelectedLocalFiles(new Set());
      setError(reason instanceof Error ? reason.message : "Could not scan folder");
    } finally {
      setIsScanning(false);
    }
  };

  const toggleFolderNode = (node: LocalFolderNode, checked: boolean) => {
    const paths = collectFilePaths(node);
    const next = new Set(selectedLocalFiles);
    for (const path of paths) {
      if (checked) next.add(path);
      else next.delete(path);
    }
    if (folderTree && next.size > folderTree.max_selection) {
      // 原因：让超限选择进入请求只会在解析前得到后端 422，用户无法判断哪一步失败。
      // 作用：保留原选择并立即显示同一份后端限制，不启动任何文档或模型任务。
      setError(`Select at most ${folderTree.max_selection} documents at once.`);
      return;
    }
    setSelectedLocalFiles(next);
    setError(null);
    setResult(null);
    setSelectedSections({});
  };

  const toggleSavedDocument = (documentId: string, checked: boolean) => {
    setSelectedSavedDocuments((current) => {
      const next = new Set(current);
      if (checked) next.add(documentId);
      else next.delete(documentId);
      return next;
    });
    setSavedNotice(null);
    setResult(null);
    setSelectedSections({});
  };

  const selectAllSavedDocuments = () => {
    setSelectedSavedDocuments((current) =>
      current.size === savedDocuments.length
        ? new Set()
        : new Set(savedDocuments.map((document) => document.document_id)),
    );
    setSavedNotice(null);
    setResult(null);
    setSelectedSections({});
  };

  const attachSavedDocuments = async () => {
    if (
      !conversationId
      || !selectedSavedDocuments.size
      || isAttachingSaved
      || isRunning
    ) return;
    setIsAttachingSaved(true);
    setSavedError(null);
    setSavedNotice(null);
    try {
      const attached = await api.attachSavedDocuments(
        conversationId,
        Array.from(selectedSavedDocuments),
      );
      setSavedNotice(
        `${attached.attached_count} document${
          attached.attached_count === 1 ? "" : "s"
        } attached to this chat.`,
      );
    } catch (reason) {
      setSavedError(
        reason instanceof Error ? reason.message : "Could not attach saved documents",
      );
    } finally {
      setIsAttachingSaved(false);
    }
  };

  const analyzeSavedDocuments = async () => {
    if (
      !conversationId
      || !selectedSavedDocuments.size
      || isRunning
      || analysisInputError
    ) return;
    setIsRunning(true);
    setError(null);
    setSavedError(null);
    setSavedNotice(null);
    try {
      const analysisResult = await api.analyzeSavedDocuments({
        conversationId,
        documentIds: Array.from(selectedSavedDocuments),
        question,
        generateReport,
        minSourceRelevance,
        responseDetail,
        analysisMode,
        selectedSections,
      });
      setResult(analysisResult);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Saved-document analysis failed");
    } finally {
      setIsRunning(false);
    }
  };

  const analyze = async () => {
    const hasInput =
      sourceMode === "upload" ? files.length > 0 : selectedLocalFiles.size > 0;
    if (!conversationId || !hasInput || isRunning || analysisInputError) return;
    setIsRunning(true);
    setError(null);
    try {
      // 原因：MinerU、Excel 沙箱、MiniRAG 和报告逻辑属于 Python 业务层。
      // 作用：浏览器只上传原文件并显示结果，避免前端产生另一套不一致的解析结果。
      const analysisResult =
        sourceMode === "upload"
          ? await api.analyze(
              conversationId,
              files,
              question,
              generateReport,
              minSourceRelevance,
              responseDetail,
              analysisMode,
              selectedSections,
            )
          : await api.analyzeLocalFolder({
              conversationId,
              root: folderTree?.root ?? "",
              selectedFiles: Array.from(selectedLocalFiles),
              question,
              generateReport,
              responseDetail,
              analysisMode,
              selectedSections,
            });
      setResult(analysisResult);
      if (sourceMode === "upload") await loadSavedDocuments();
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
          <p>
            {sourceMode === "upload"
              ? "MinerU · pandas sandbox · MiniRAG"
              : "Local files · direct analysis"}
          </p>
        </div>
        {sourceMode === "upload" && (
          <button className="primary-button" onClick={() => inputRef.current?.click()}>
            <Upload size={16} />
            Add files
          </button>
        )}
        <input
          ref={inputRef}
          type="file"
          multiple
          hidden
          accept=".pdf,.docx,.md,.txt,.xlsx,.xls,.csv,.png,.jpg,.jpeg"
          onChange={addFiles}
        />
      </header>

      {sourceMode === "upload" && (
        <section className="saved-documents">
          <div className="saved-documents-heading">
            <div>
              <Database size={17} />
              <h2>Saved documents</h2>
              <span>{savedDocuments.length}</span>
            </div>
            <div className="saved-document-actions">
              <button
                className="text-button"
                disabled={!savedDocuments.length || isRunning || isAttachingSaved}
                onClick={selectAllSavedDocuments}
                type="button"
              >
                {selectedSavedDocuments.size === savedDocuments.length
                  ? "Clear"
                  : "Select all"}
              </button>
              <button
                className="secondary-button"
                disabled={
                  !conversationId
                  || !selectedSavedDocuments.size
                  || isRunning
                  || isAttachingSaved
                }
                onClick={() => void attachSavedDocuments()}
                type="button"
              >
                {isAttachingSaved ? <LoaderCircle className="spin" size={14} /> : <Database size={14} />}
                {isAttachingSaved ? "Attaching" : "Attach to chat"}
              </button>
              <button
                className="secondary-button"
                disabled={
                  !conversationId
                  || !selectedSavedDocuments.size
                  || isRunning
                  || isAttachingSaved
                  || Boolean(analysisInputError)
                }
                onClick={() => void analyzeSavedDocuments()}
                type="button"
              >
                {isRunning ? <LoaderCircle className="spin" size={14} /> : <FileText size={14} />}
                Analyze selected
              </button>
              <button
                className="icon-button"
                disabled={isLoadingSaved}
                onClick={() => void loadSavedDocuments()}
                title="Refresh saved documents"
                type="button"
              >
                <RefreshCw className={isLoadingSaved ? "spin" : ""} size={15} />
              </button>
            </div>
          </div>
          {savedError && <div className="error-banner">{savedError}</div>}
          {savedNotice && <div className="success-banner">{savedNotice}</div>}
          {!savedError && !isLoadingSaved && savedDocuments.length === 0 && (
            <div className="empty-files">No parsed documents have been saved yet.</div>
          )}
          <div className="saved-document-list">
            {savedDocuments.map((document) => (
              <label className="saved-document-row" key={document.document_id}>
                <input
                  checked={selectedSavedDocuments.has(document.document_id)}
                  onChange={(event) =>
                    toggleSavedDocument(document.document_id, event.target.checked)
                  }
                  type="checkbox"
                />
                <FileText size={17} />
                <div>
                  <strong title={document.source}>{document.source}</strong>
                  <small>
                    {document.file_type.toUpperCase()} · {formatBytes(document.size_bytes)} ·{" "}
                    {document.section_count} sections
                    {document.summary_available ? " · summarized" : ""}
                  </small>
                </div>
                <time dateTime={document.saved_at}>{formatSavedAt(document.saved_at)}</time>
              </label>
            ))}
          </div>
        </section>
      )}

      <div className="document-controls">
        <div className="analysis-mode source-mode-switch" aria-label="Document source">
          <button
            className={sourceMode === "upload" ? "active" : ""}
            onClick={() => changeSourceMode("upload")}
            type="button"
          >
            <Upload size={14} />
            Uploads
          </button>
          {canUseLocalFolder && (
            <button
              className={sourceMode === "folder" ? "active" : ""}
              onClick={() => changeSourceMode("folder")}
              type="button"
            >
              <Folder size={14} />
              Local folder
            </button>
          )}
        </div>

        {sourceMode === "upload" ? (
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
        ) : (
          <div className="folder-picker">
            <label className="field-label" htmlFor="folder-path">Folder path</label>
            <div className="folder-path-row">
              <input
                id="folder-path"
                value={folderPath}
                onChange={(event) => setFolderPath(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") void scanFolder();
                }}
                placeholder="/Users/name/Documents/project"
              />
              <button
                className="secondary-button"
                disabled={!folderPath.trim() || isScanning}
                onClick={() => void scanFolder()}
                type="button"
              >
                {isScanning ? <LoaderCircle className="spin" size={15} /> : <Search size={15} />}
                Scan
              </button>
            </div>
            {folderTree && (
              <div className="folder-tree-panel">
                <div className="folder-tree-heading">
                  <strong>{folderTree.tree.name}</strong>
                  <span>
                    {selectedLocalFiles.size} / {folderTree.file_count} selected
                    {folderTree.file_count > folderTree.max_selection
                      ? ` · max ${folderTree.max_selection}`
                      : ""}
                  </span>
                </div>
                <div className="folder-tree" role="tree">
                  {folderTree.tree.children.map((node) => (
                    <FolderTreeNode
                      key={node.relative_path}
                      node={node}
                      selected={selectedLocalFiles}
                      onToggle={toggleFolderNode}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

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
        {analysisInputError && (
          <div className="error-banner" role="status">{analysisInputError}</div>
        )}
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
          <button
            className="primary-button"
            disabled={
              !conversationId
              || isRunning
              || Boolean(analysisInputError)
              || (sourceMode === "upload" ? !files.length : !selectedLocalFiles.size)
            }
            onClick={analyze}
          >
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
          {result.spreadsheets.length > 0 && (
            <section className="workbook-results">
              <div className="outline-heading">
                <TableProperties size={17} />
                <h2>Workbook structure</h2>
              </div>
              {result.spreadsheets.map((workbook) => (
                <details className="workbook-detail" key={workbook.source}>
                  <summary>
                    <strong>{workbook.source}</strong>
                    <span>
                      {workbook.sheet_count} sheets · {workbook.tables.length} analysis tables
                    </span>
                  </summary>
                  <div className="workbook-metrics">
                    <span>{workbook.formula_count} formulas</span>
                    <span>{workbook.merged_range_count} merged ranges</span>
                    <span>{workbook.chart_count} charts</span>
                    <span>{workbook.image_count} images</span>
                    <span>{workbook.data_validation_count} validations</span>
                  </div>
                  <div className="workbook-sheet-list">
                    {workbook.sheets.map((sheet) => (
                      <div key={sheet.name}>
                        <strong>{sheet.name}</strong>
                        <span>{sheet.kind.replace("_", " ")}</span>
                        <small>{sheet.region_count} regions</small>
                      </div>
                    ))}
                  </div>
                  <div className="workbook-table-scroll">
                    <table className="workbook-table">
                      <thead>
                        <tr>
                          <th>Analysis table</th>
                          <th>Size</th>
                          <th>Columns</th>
                        </tr>
                      </thead>
                      <tbody>
                        {workbook.tables.map((table) => (
                          <tr key={table.name}>
                            <td>{table.name}</td>
                            <td>{table.rows} × {table.columns}</td>
                            <td>
                              {table.column_names.join(", ")}
                              {table.columns_truncated ? ", ..." : ""}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              ))}
            </section>
          )}
          <article className="analysis-result">
            <div className="result-meta">
              <span>{result.route}</span>
              <span>{result.citations.length} citations</span>
              {result.generation_mode && (
                <span>
                  {result.generation_mode === "grounded_composer"
                    ? "grounded evidence composer"
                    : result.generation_mode}
                </span>
              )}
              {result.source_coverage && (
                <span>
                  {result.source_coverage.covered_sources.length}/
                  {result.source_coverage.required_sources.length} sources
                </span>
              )}
            </div>
            {result.source_coverage && (
              <details
                className={`source-coverage ${
                  result.source_coverage.complete ? "complete" : "incomplete"
                }`}
                open={!result.source_coverage.complete}
              >
                <summary>
                  {result.source_coverage.complete
                    ? "All selected sources were inspected"
                    : "Source coverage is incomplete"}
                </summary>
                <ul>
                  {result.source_coverage.required_sources.map((source) => {
                    const covered = result.source_coverage?.covered_sources.includes(source);
                    return (
                      <li className={covered ? "covered" : "missing"} key={source}>
                        <span aria-hidden="true">{covered ? "✓" : "!"}</span>
                        {source}
                      </li>
                    );
                  })}
                </ul>
              </details>
            )}
            <div className="message-markdown analysis-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result.answer}</ReactMarkdown>
            </div>
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

type FolderTreeNodeProps = {
  node: LocalFolderNode;
  selected: Set<string>;
  onToggle: (node: LocalFolderNode, checked: boolean) => void;
  depth?: number;
};

function FolderTreeNode({
  node,
  selected,
  onToggle,
  depth = 0,
}: FolderTreeNodeProps) {
  const filePaths = collectFilePaths(node);
  const selectedCount = filePaths.filter((path) => selected.has(path)).length;
  const checked = filePaths.length > 0 && selectedCount === filePaths.length;
  const partiallyChecked = selectedCount > 0 && !checked;

  return (
    <div role="treeitem" aria-selected={checked}>
      <label
        className="folder-tree-row"
        style={{ paddingLeft: `${8 + depth * 18}px` }}
      >
        <input
          type="checkbox"
          checked={checked}
          ref={(element) => {
            if (element) element.indeterminate = partiallyChecked;
          }}
          onChange={(event) => onToggle(node, event.target.checked)}
        />
        {node.kind === "directory" ? <Folder size={15} /> : <FileText size={15} />}
        <span title={node.relative_path}>{node.name}</span>
        {node.kind === "directory" && <small>{selectedCount} / {filePaths.length}</small>}
      </label>
      {node.children.map((child) => (
        <FolderTreeNode
          depth={depth + 1}
          key={child.relative_path}
          node={child}
          selected={selected}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}

function collectFilePaths(node: LocalFolderNode): string[] {
  if (node.kind === "file") return [node.relative_path];
  return node.children.flatMap(collectFilePaths);
}

function pageLabel(start: number, end?: number): string {
  return end && end !== start ? `pp. ${start}-${end}` : `p. ${start}`;
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / 1024 / 1024).toFixed(1)} MB`;
}

function formatSavedAt(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(value));
}

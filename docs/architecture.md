# Qwopus-Agent Architecture

## Design Principles

- Model-agnostic: all model providers implement `BaseLLM`.
- Interface-first: agents, memory, reflection, and skills communicate through stable contracts.
- Local-first: the first concrete model adapter targets `mlx_lm.server` through its OpenAI-compatible API.
- Incremental: each capability is independently testable and composed at CLI/UI boundaries.

## Runtime Modules

### LLM

`BaseLLM` defines the model contract. `LLMConfig` and `LLMRegistry` create concrete adapters from
configuration, so Planner, Executor, Skills, and CLI/UI never depend on Gemma, Qwopus, Qwen, or any
other model family.

`OpenAICompatibleLLM` supports any local or remote runtime that exposes `/v1/chat/completions`.
`LocalMLXLLM` is only a preset for `mlx_lm.server`, not a model-specific dependency.

### Agent

The agent layer is split into:

- `Planner`: creates a `Plan` from a user objective.
- `Executor`: executes a `Plan` through `SkillRegistry`.
- `AgentRouter`: coordinates planning and execution.
- `ResearchAgent`: reuses `AgentRouter` and reflection for research-style tasks.
- `MultiAgentSupervisor`: delegates dependency-aware tasks, executes independent waves in parallel,
  shares state, runs result debate, and returns one arbitrated final answer.

Production answer orchestration separates evidence from writing. `AnswerPlan` defines the central
goal and task-specific depth questions. Capability workers return typed `EvidencePacket` objects,
which are deduplicated into an `EvidenceLedger`; they never write user-facing answers. The reviewer
returns only structured agreements, conflicts, unsupported claims, and material gaps. For complex
detailed requests, one `gap_fill` task may reuse already authorized search tools when the review
contains a concrete gap. The final Synthesizer is the only role prompted to write natural language
for the user. Weaker compatible models that fail JSON formatting use bounded deterministic
fallbacks, so the architecture remains independent from one model family's structured-output
quality.

### Memory

`MiniRAG` exposes only `insert(document)` and `search(query)`. Original Markdown documents are
persisted as JSONL under `storage/minirag`; MiniRAG's `NanoVectorDBStorage` persists multilingual
semantic embeddings in a separate derived index. Search results include file and page citations
when the normalized Markdown contains that metadata. The embedding model is local and independent
from the currently selected chat model.

The same insert operation also feeds an evidence-constrained graph pipeline. A deterministic
extractor handles explicit relations, while `LLMGraphExtractor` resolves the currently selected
`BaseLLM` at insertion time for ordinary document prose. Invalid chunk ids, unsupported quotes, and
relations whose evidence omits an endpoint are rejected before persistence in a directed
`MultiDiGraph`. Search combines bounded graph paths with complementary vector chunks while
deduplicating source evidence. Document updates, deletion, and rebuilds are owned by
`KnowledgeMaintenanceService`, so the Agent-facing MiniRAG contract remains limited to `insert` and
`search`.

`ConversationKnowledgeManager` supplies two authorization-aware persistence
scopes. Each chat writes to `storage/minirag/conversations/<conversation_id>`,
while its owner's optional cross-chat aggregate is stored at
`storage/minirag/users/<user_id>`. The API resolves both paths from the
authenticated account and conversation ACL; callers cannot submit an arbitrary
knowledge directory.

### Analysis Safety

LLM-generated pandas is parsed through an allowlisted AST in both the parent and worker process. The
worker receives no application secrets, has CPU/file/descriptor limits, and returns inert JSON. On
macOS it additionally runs under Seatbelt with network, file-write, process-fork, and sensitive-path
reads denied. This boundary is intended for generated dataframe analysis expressions, not arbitrary
user Python.

### Web Boundary

Every direct non-loopback HTTP request fails closed unless `QWOPUS_LAN_PASSWORD` is configured,
then requires one shared HTTP Basic credential before React or API access. Application accounts form
the second boundary: Argon2id password hashes, opaque server-side sessions, deny-by-default private
API middleware, and per-conversation/document/report ACL checks isolate users.

The Debug Console is deliberately different from normal account access. It is restricted to
loopback clients with an administrator session and can read diagnostic records for every account,
including prompts, document excerpts, model outputs, and the recorded actor. No LAN override exists.
HTTPS is still required when transport confidentiality matters for the main LAN application.

### Skills

`SkillRegistry.discover()` scans `qwopus_agent.skills` and imports modules with `create_skill()`.
Planner selects skills; Executor runs them through the registry. `graph_search` exposes persistent
multi-hop paths and cross-document evidence without manual registration.

`SkillGrowthService` observes complete successful Agent runs. Repeated traces are converted into
declarative `WorkflowSpec` files, stripped of paths and credentials, integrity checked, assigned a
semantic version in `SkillCatalog`, and loaded into `SkillRegistry`.

`SkillAuthoringService` lets an approved Debug Console user ask the current `BaseLLM` to compose a
candidate from explicitly allowed existing Skills. Pydantic rejects unknown fields, unapproved
capabilities, persistent arguments, and malformed output. The candidate remains outside the runtime
Registry until manual promotion; review exposes its exact spec, checksum checks, version diff, and a
side-effect-free dry run. Model-generated arbitrary Python is not accepted or deployed.

### Reports

`ReportGenerator` is the unified report module. It writes Markdown, Excel, real PNG/SVG charts, and a
PDF artifact from one request.

### Prompts

`qwopus_agent.prompts` owns model-facing task construction, response-depth rules, language policy,
and evidence requirements. smolagents integration code consumes these policies but does not own
them.

### Reflection

`TaskReflectionEvaluator` provides structured quality observations and retry suggestions without
requiring another LLM call.

## Current Boundaries

- Browser automation has a tested Skill and provider contract, but no production browser provider
  is wired into the application.
- Accounts are local to one Qwopus-Agent installation. Sharing grants editor access to one complete
  chat and its attached documents/reports; field-level permissions and external identity providers
  are outside the current runtime.
- Authorization isolates data through the application and SQLite ACLs. Runtime files are not
  encrypted per account on disk, so operating-system access to the host remains trusted.
- The MiniRAG adapter uses upstream persistent vector storage inside Qwopus-owned chunking,
  conversation scoping, graph extraction, and evidence rendering. It does not expose the complete
  upstream `MiniRAG.query` pipeline.
- Model-authored Skills remain declarative workflows and require manual promotion. Arbitrary
  generated Python is not accepted as deployable Skill code.

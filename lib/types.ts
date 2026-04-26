export type JsonObject = Record<string, unknown>;

export interface HealthResponse {
  status: "ok" | string;
  app_name: string;
  app_env: string;
  runtime_backend: string;
  llm_provider: string;
  embedding_provider: string;
  vector_store_provider: string;
  graph_store_provider: string;
  graph_runtime_ready: boolean;
  checkpointer_backend?: string | null;
  session_memory_backend?: string | null;
}

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  unit?: "pixel" | "point" | "relative";
}

export interface TextBlock {
  id: string;
  document_id: string;
  page_id: string;
  page_number: number;
  text: string;
  bbox?: BoundingBox | null;
  block_type?: "paragraph" | "title" | "table_text" | "caption" | "footnote" | "header" | "footer";
  confidence?: number | null;
  metadata: JsonObject;
}

export interface DocumentPage {
  id: string;
  document_id: string;
  page_number: number;
  width?: number | null;
  height?: number | null;
  image_uri?: string | null;
  text_blocks: TextBlock[];
  metadata: JsonObject;
}

export interface ParsedDocument {
  id: string;
  filename: string;
  content_type: string;
  status: "uploaded" | "parsing" | "parsed" | "failed" | string;
  pages: DocumentPage[];
  error_message?: string | null;
  metadata: JsonObject;
}

export interface UploadDocumentResponse {
  document_id: string;
  filename: string;
  status: "uploaded" | "failed" | string;
  storage_uri?: string | null;
  error_message?: string | null;
  metadata?: JsonObject;
}

export interface ParseDocumentRequest {
  file_path: string;
  document_id?: string | null;
  skill_name?: string | null;
}

export interface ParseDocumentResponse {
  document_id: string;
  status: "parsing" | "parsed" | "failed" | string;
  parsed_document?: ParsedDocument | null;
  error_message?: string | null;
}

export interface AxisSchema {
  name?: string | null;
  label?: string | null;
  unit?: string | null;
  scale?: "linear" | "log" | "time" | "categorical" | "unknown";
  min_value?: number | string | null;
  max_value?: number | string | null;
  categories?: string[];
}

export interface SeriesPoint {
  x?: number | string | null;
  y?: number | string | null;
  value?: number | string | null;
  label?: string | null;
  metadata: JsonObject;
}

export interface SeriesSchema {
  name: string;
  chart_role?: "bar" | "line" | "scatter" | "area" | "pie_slice" | "table_cell" | "unknown";
  points: SeriesPoint[];
  unit?: string | null;
  metadata: JsonObject;
}

export interface ChartSchema {
  id: string;
  document_id: string;
  page_id: string;
  page_number: number;
  chart_type?: "bar" | "line" | "scatter" | "pie" | "table" | "mixed" | "unknown" | string;
  title?: string | null;
  caption?: string | null;
  bbox?: BoundingBox | null;
  x_axis?: AxisSchema | null;
  y_axis?: AxisSchema | null;
  series: SeriesSchema[];
  summary?: string | null;
  confidence?: number | null;
  metadata: JsonObject;
}

export interface GraphNode {
  id: string;
  label: string;
  properties: JsonObject;
  source_reference?: Evidence;
}

export interface GraphEdge {
  id: string;
  type: string;
  source_node_id: string;
  target_node_id: string;
  properties: JsonObject;
  source_reference?: Evidence;
}

export interface GraphTriple {
  subject: GraphNode;
  predicate: GraphEdge;
  object: GraphNode;
}

export interface GraphExtractionResult {
  document_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  triples: GraphTriple[];
  status: "succeeded" | "partial" | "failed" | string;
  error_message?: string | null;
  metadata: JsonObject;
}

export interface GraphIndexStats {
  document_id: string;
  status: string;
  node_count: number;
  edge_count: number;
  skipped_node_count?: number;
  skipped_edge_count?: number;
  error_message?: string | null;
  metadata: JsonObject;
}

export interface EmbeddingIndexResult {
  document_id: string;
  status: "indexed" | "partial" | "skipped" | "failed" | string;
  record_count: number;
  skipped_count?: number;
  failed_count?: number;
  record_ids?: string[];
  error_message?: string | null;
  metadata: JsonObject;
}

export interface IndexDocumentRequest {
  parsed_document: ParsedDocument;
  charts: ChartSchema[];
  include_graph: boolean;
  include_embeddings: boolean;
  skill_name?: string | null;
}

export interface DocumentIndexResult {
  document_id?: string;
  graph_extraction?: GraphExtractionResult | null;
  graph_index?: GraphIndexStats | null;
  text_embedding_index?: EmbeddingIndexResult | null;
  page_embedding_index?: EmbeddingIndexResult | null;
  chart_embedding_index?: EmbeddingIndexResult | null;
  graph?: GraphExtractionResult | GraphIndexStats | null;
  text_embeddings?: EmbeddingIndexResult | null;
  page_embeddings?: EmbeddingIndexResult | null;
  chart_embeddings?: EmbeddingIndexResult | null;
  status?: string;
  metadata?: JsonObject;
}

export interface IndexDocumentResponse {
  status: "indexing" | "indexed" | "failed" | string;
  result?: DocumentIndexResult;
  document_id?: string;
  embedding_record_count?: number;
  graph_extraction?: GraphExtractionResult | null;
  error_message?: string | null;
}

export interface AskDocumentRequest {
  question: string;
  doc_id?: string | null;
  document_ids: string[];
  top_k: number;
  session_id?: string | null;
  task_intent?: string | null;
  filters: JsonObject;
  metadata?: JsonObject;
  skill_name?: string | null;
  reasoning_style?: string | null;
}

export interface Evidence {
  id: string;
  document_id?: string | null;
  page_id?: string | null;
  page_number?: number | null;
  source_type: string;
  source_id?: string | null;
  snippet?: string | null;
  score?: number | null;
  graph_node_ids?: string[];
  graph_edge_ids?: string[];
  metadata: JsonObject;
}

export interface EvidenceBundle {
  evidences: Evidence[];
  summary?: string | null;
  metadata: JsonObject;
}

export interface RetrievalQuery {
  query: string;
  document_ids: string[];
  mode?: "vector" | "graph" | "hybrid" | string;
  modalities?: string[];
  top_k: number;
  filters: JsonObject;
  graph_query_mode?: "entity" | "subgraph" | "summary" | "auto" | string;
}

export interface RetrievalHit {
  id: string;
  source_type: string;
  source_id: string;
  document_id?: string | null;
  content?: string | null;
  vector_score?: number | null;
  graph_score?: number | null;
  merged_score?: number | null;
  graph_nodes?: GraphNode[];
  graph_edges?: GraphEdge[];
  graph_triples?: GraphTriple[];
  evidence?: EvidenceBundle | null;
  metadata: JsonObject;
}

export interface RetrievalResult {
  query?: RetrievalQuery;
  hits: RetrievalHit[];
  evidence_bundle?: EvidenceBundle;
  metadata: JsonObject;
}

export interface QAResponse {
  answer: string;
  question: string;
  evidence_bundle: EvidenceBundle;
  retrieval_result?: RetrievalResult | null;
  confidence?: number | null;
  metadata: JsonObject;
}

export interface ToolTrace {
  trace_id?: string;
  node_name?: string;
  tool_name?: string;
  status?: string;
  latency_ms?: number | null;
  metadata?: JsonObject;
}

export interface AskDocumentResponse {
  document_ids: string[];
  qa: QAResponse;
}

export interface AskFusedRequest {
  question: string;
  image_path: string;
  doc_id?: string | null;
  document_ids: string[];
  page_id?: string | null;
  page_number?: number;
  chart_id?: string | null;
  session_id?: string | null;
  top_k?: number;
  filters?: JsonObject;
  metadata?: JsonObject;
  skill_name?: string | null;
  reasoning_style?: string | null;
}

export interface AskFusedResponse {
  document_ids: string[];
  qa: QAResponse;
  chart_answer?: string | null;
  chart_confidence?: number | null;
}

export interface ChartUnderstandRequest {
  image_path: string;
  document_id: string;
  page_id: string;
  page_number: number;
  chart_id: string;
  context: JsonObject;
  skill_name?: string | null;
}

export interface ChartUnderstandingResult {
  chart: ChartSchema;
  graph_text: string;
  metadata: JsonObject;
}

export interface ChartUnderstandResponse {
  status: string;
  result: ChartUnderstandingResult;
}

export interface AskChartRequest {
  image_path: string;
  question: string;
  session_id?: string | null;
  document_id?: string | null;
  page_id?: string | null;
  page_number?: number;
  chart_id?: string | null;
  context?: JsonObject;
}

export interface AskChartResponse {
  status: string;
  answer: string;
  session_id: string;
  confidence?: number | null;
  evidence?: JsonObject;
  metadata: JsonObject;
}

export type ResearchSource = "arxiv" | "openalex" | "semantic_scholar" | "ieee" | "zotero";

export interface ResearchTopicPlan {
  topic: string;
  normalized_topic: string;
  queries: string[];
  days_back: number;
  max_papers: number;
  sources: ResearchSource[];
  metadata: JsonObject;
}

export interface PaperCandidate {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  year?: number | null;
  venue?: string | null;
  source: ResearchSource;
  doi?: string | null;
  arxiv_id?: string | null;
  pdf_url?: string | null;
  url?: string | null;
  citations?: number | null;
  is_open_access?: boolean | null;
  published_at?: string | null;
  relevance_score?: number | null;
  summary?: string | null;
  ingest_status: "not_selected" | "selected" | "ingested" | "unavailable";
  metadata: JsonObject;
}

export interface ResearchCluster {
  name: string;
  paper_ids: string[];
  description?: string | null;
}

export interface ComparisonTableRow {
  dimension: string;
  values: Record<string, string>;
}

export interface ComparePapersResult {
  table: ComparisonTableRow[];
  summary: string;
}

export interface RecommendedPaper {
  paper_id: string;
  title: string;
  reason: string;
  source?: ResearchSource | string | null;
  year?: number | null;
  url?: string | null;
}

export interface RecommendPapersResult {
  recommendations: RecommendedPaper[];
}

export interface PaperAnalysisResult {
  answer: string;
  focus: string;
  key_points: string[];
  recommended_paper_ids: string[];
}

export interface ResearchVisualAnchor {
  image_path: string | null;
  page_id: string | null;
  page_number: number | null;
  chart_id: string | null;
  figure_id: string | null;
  anchor_rationale?: string | null;
}

export interface ResearchQATraceSummary {
  route: string | null;
  confidence: number | null;
  rationale: string | null;
  runtime: string | null;
  anchorRationale: string | null;
  anchor: Record<string, string | number>;
}

export interface ResearchQAPaperScope {
  paper_ids: string[];
  scope_mode: string;
}

export interface ContextCompressionSummary {
  paper_count: number;
  summary_count: number;
  levels: string[];
  compressed_paper_ids?: string[];
}

export interface ResearchAdvancedStrategy {
  action: ResearchAdvancedAction;
  comparison_dimensions: string[];
  recommendation_goal?: string | null;
  recommendation_top_k: number;
  force_context_compression: boolean;
}

export interface ResearchWorkspaceState {
  objective: string;
  current_stage: "discover" | "ingest" | "qa" | "document" | "chart" | "complete" | string;
  research_questions: string[];
  hypotheses: string[];
  key_findings: string[];
  evidence_gaps: string[];
  must_read_paper_ids: string[];
  ingest_candidate_ids: string[];
  document_ids: string[];
  next_actions: string[];
  stop_reason?: string | null;
  status_summary: string;
  metadata: JsonObject;
}

export interface ResearchReport {
  report_id: string;
  task_id?: string | null;
  topic: string;
  generated_at: string;
  markdown: string;
  paper_count: number;
  source_counts: Record<string, number>;
  highlights: string[];
  clusters: ResearchCluster[];
  gaps: string[];
  workspace: ResearchWorkspaceState;
  metadata: JsonObject;
}

export interface ResearchTodoItem {
  todo_id: string;
  content: string;
  rationale?: string | null;
  status: "open" | "done" | "dismissed" | string;
  priority: "high" | "medium" | "low" | string;
  created_at: string;
  question?: string | null;
  source: "qa_follow_up" | "evidence_gap" | string;
  metadata: JsonObject;
}

export interface UpdateResearchTodoRequest {
  status: "open" | "done" | "dismissed" | string;
}

export interface ResearchTodoActionRequest {
  max_papers: number;
  include_graph: boolean;
  include_embeddings: boolean;
  skill_name?: string | null;
  conversation_id?: string | null;
}

export interface SearchPapersRequest {
  topic: string;
  days_back: number;
  max_papers: number;
  sources: ResearchSource[];
}

export interface SearchPapersResponse {
  plan: ResearchTopicPlan;
  papers: PaperCandidate[];
  report: ResearchReport;
  warnings: string[];
}

export interface ResearchTask {
  task_id: string;
  topic: string;
  status: "created" | "running" | "completed" | "failed" | string;
  created_at: string;
  updated_at: string;
  days_back: number;
  max_papers: number;
  sources: ResearchSource[];
  paper_count: number;
  imported_document_ids: string[];
  todo_items: ResearchTodoItem[];
  report_id?: string | null;
  workspace: ResearchWorkspaceState;
  metadata: JsonObject;
}

export interface CreateResearchTaskRequest extends SearchPapersRequest {
  run_immediately: boolean;
}

export interface ResearchTaskResponse {
  task: ResearchTask;
  papers: PaperCandidate[];
  report?: ResearchReport | null;
  warnings: string[];
}

export interface ImportPapersRequest {
  task_id?: string | null;
  paper_ids: string[];
  papers: PaperCandidate[];
  include_graph: boolean;
  include_embeddings: boolean;
  skill_name?: string | null;
  conversation_id?: string | null;
  question?: string | null;
  top_k?: number;
  reasoning_style?: string | null;
}

export interface ImportedPaperResult {
  paper_id: string;
  title: string;
  status: "imported" | "skipped" | "failed" | string;
  document_id?: string | null;
  storage_uri?: string | null;
  parsed: boolean;
  indexed: boolean;
  error_message?: string | null;
  metadata: JsonObject;
}

export interface ImportPapersResponse {
  results: ImportedPaperResult[];
  imported_count: number;
  skipped_count: number;
  failed_count: number;
}

export interface ResearchPaperFigurePreview {
  figure_id: string;
  paper_id: string;
  document_id: string;
  page_id: string;
  page_number: number;
  chart_id: string;
  title?: string | null;
  caption?: string | null;
  source: "chart_candidate" | "page_fallback" | string;
  bbox?: BoundingBox | null;
  image_path?: string | null;
  preview_data_url?: string | null;
  metadata: JsonObject;
}

export interface ResearchPaperFigureListResponse {
  task_id: string;
  paper_id: string;
  document_id: string;
  figures: ResearchPaperFigurePreview[];
  warnings: string[];
}

export interface AnalyzeResearchPaperFigureRequest {
  figure_id?: string | null;
  page_id: string;
  chart_id: string;
  image_path?: string | null;
  question?: string | null;
}

export interface AnalyzeResearchPaperFigureResponse {
  task_id: string;
  paper_id: string;
  figure: ResearchPaperFigurePreview;
  chart: ChartSchema;
  graph_text: string;
  answer: string;
  key_points: string[];
  metadata: JsonObject;
}

export interface ResearchTaskAskRequest {
  question: string;
  top_k: number;
  conversation_id?: string | null;
  paper_ids?: string[];
  document_ids?: string[];
  image_path?: string | null;
  page_id?: string | null;
  page_number?: number | null;
  chart_id?: string | null;
  return_citations?: boolean;
  min_length?: number;
  skill_name?: string | null;
  reasoning_style?: string | null;
  metadata?: JsonObject;
}

export interface ResearchTaskAskResponse {
  task_id: string;
  paper_ids: string[];
  document_ids: string[];
  scope_mode: "all_imported" | "selected_papers" | "selected_documents" | "metadata_only" | string;
  qa: QAResponse;
  report?: ResearchReport | null;
  todo_items: ResearchTodoItem[];
  warnings: string[];
}

export interface ResearchTodoActionResponse {
  task: ResearchTask;
  todo: ResearchTodoItem;
  papers: PaperCandidate[];
  report?: ResearchReport | null;
  warnings: string[];
  import_result?: ImportPapersResponse | null;
}

export interface ResearchMessage {
  message_id: string;
  role: "assistant" | "user" | "system" | string;
  kind: string;
  title: string;
  content: string;
  meta?: string | null;
  created_at: string;
  citations: string[];
  payload: JsonObject;
}

export interface ResearchConversationSnapshot {
  topic: string;
  days_back: number;
  max_papers: number;
  sources: ResearchSource[];
  composer_mode: "research" | "qa" | string;
  advanced_strategy: ResearchAdvancedStrategy;
  selected_paper_ids: string[];
  active_paper_ids: string[];
  workspace: ResearchWorkspaceState;
  search_result?: SearchPapersResponse | null;
  task_result?: ResearchTaskResponse | null;
  import_result?: ImportPapersResponse | null;
  ask_result?: ResearchTaskAskResponse | null;
  last_error?: string | null;
  last_notice?: string | null;
  active_job_id?: string | null;
  context_summary?: ResearchContextSummary | null;
  recent_events?: ResearchRuntimeEvent[] | null;
}

export interface ResearchStatusMetadata {
  lifecycle_status:
    | "queued"
    | "running"
    | "waiting_input"
    | "completed"
    | "failed"
    | "cancelled"
    | string;
  started_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  error_code?: string | null;
  error_message?: string | null;
  retry_count: number;
  correlation_id?: string | null;
}

export interface ResearchContextSummary {
  summary_version: number;
  objective: string;
  current_stage: string;
  topic?: string | null;
  paper_count: number;
  imported_document_count: number;
  selected_paper_count: number;
  key_findings: string[];
  evidence_gaps: string[];
  next_actions: string[];
  status_summary: string;
  last_user_message?: string | null;
  last_updated_at?: string | null;
}

export interface ResearchRuntimeEvent {
  event_id: string;
  event_type:
    | "agent_started"
    | "agent_routed"
    | "tool_called"
    | "tool_succeeded"
    | "tool_failed"
    | "memory_updated"
    | "task_completed"
    | "task_failed"
    | string;
  task_id?: string | null;
  conversation_id?: string | null;
  correlation_id?: string | null;
  timestamp: string;
  payload: JsonObject;
}

export interface ResearchConversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  task_id?: string | null;
  message_count: number;
  last_message_preview?: string | null;
  snapshot: ResearchConversationSnapshot;
  status_metadata?: ResearchStatusMetadata | null;
  metadata: JsonObject;
}

export interface ResearchConversationResponse {
  conversation: ResearchConversation;
  messages: ResearchMessage[];
}

export interface CreateResearchConversationRequest {
  title?: string | null;
  topic?: string | null;
  days_back?: number;
  max_papers?: number;
  sources?: ResearchSource[];
}

export interface ResearchJob {
  job_id: string;
  kind: "paper_import" | "todo_import" | string;
  status: "queued" | "running" | "completed" | "failed" | string;
  created_at: string;
  updated_at: string;
  task_id?: string | null;
  conversation_id?: string | null;
  progress_message?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
  error_message?: string | null;
  output: JsonObject;
  status_metadata?: ResearchStatusMetadata | null;
  metadata: JsonObject;
}

export type ResearchAgentMode = "auto" | "research" | "qa" | "import" | "document" | "chart";
export type ResearchAdvancedAction = "discover" | "analyze" | "compare" | "recommend";

export interface ResearchAgentRunRequest {
  message: string;
  mode?: ResearchAgentMode;
  task_id?: string | null;
  conversation_id?: string | null;
  days_back?: number;
  max_papers?: number;
  sources?: ResearchSource[];
  selected_paper_ids?: string[];
  selected_document_ids?: string[];
  advanced_action?: ResearchAdvancedAction | null;
  comparison_dimensions?: string[];
  recommendation_goal?: string | null;
  recommendation_top_k?: number;
  force_context_compression?: boolean;
  auto_import?: boolean;
  import_top_k?: number;
  include_graph?: boolean;
  include_embeddings?: boolean;
  top_k?: number;
  skill_name?: string | null;
  reasoning_style?: string | null;
  document_file_path?: string | null;
  document_id?: string | null;
  chart_image_path?: string | null;
  page_id?: string | null;
  page_number?: number;
  chart_id?: string | null;
  metadata?: JsonObject;
}

export interface ResearchAgentTraceStep {
  step_index: number;
  agent: string;
  thought: string;
  action_name: string;
  phase: "observe" | "plan" | "act" | "reflect" | "commit" | string;
  action_input: JsonObject;
  status: "planned" | "succeeded" | "failed" | "skipped" | string;
  observation: string;
  rationale: string;
  estimated_gain?: number | null;
  estimated_cost?: number | null;
  stop_signal: boolean;
  workspace_summary?: string | null;
  metadata: JsonObject;
}

export interface UnifiedActionOutput extends JsonObject {
  unified_input_adapter: string;
}

export interface UnifiedExecutionEntry extends JsonObject {
  task_type: string;
  agent_name?: string | null;
  agent_to?: string | null;
  status?: string;
  action_output?: UnifiedActionOutput | null;
  execution_mode?: string | null;
  preferred_skill_name?: string | null;
}

export interface ResearchAgentRuntimeMetadata extends JsonObject {
  unified_supervisor_mode?: string | null;
  manager_decision_count?: number | null;
  supervisor_action_trace_count?: number | null;
  agent_result_count?: number | null;
  unified_delegation_plan?: UnifiedExecutionEntry[];
  unified_agent_results?: UnifiedExecutionEntry[];
}

export interface ResearchAgentRunResponse {
  status: "succeeded" | "partial" | "failed" | string;
  task?: ResearchTask | null;
  papers: PaperCandidate[];
  report?: ResearchReport | null;
  import_result?: ImportPapersResponse | null;
  qa?: QAResponse | null;
  parsed_document?: ParsedDocument | null;
  document_index_result?: JsonObject | null;
  chart?: ChartSchema | null;
  chart_graph_text?: string | null;
  messages: ResearchMessage[];
  trace: ResearchAgentTraceStep[];
  warnings: string[];
  next_actions: string[];
  workspace: ResearchWorkspaceState;
  metadata: ResearchAgentRuntimeMetadata;
}

export type RequestState = "idle" | "loading" | "success" | "error";

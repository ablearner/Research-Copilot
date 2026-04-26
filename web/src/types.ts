export interface PaperCandidate {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  year: number | null;
  venue: string | null;
  source: string;
  doi: string | null;
  arxiv_id: string | null;
  pdf_url: string | null;
  url: string | null;
  citations: number | null;
  is_open_access: boolean | null;
  published_at: string | null;
  relevance_score: number | null;
  summary: string | null;
  ingest_status: string;
  metadata: Record<string, unknown>;
}

export interface ResearchReport {
  report_id: string;
  task_id: string | null;
  topic: string;
  generated_at: string;
  markdown: string;
  paper_count: number;
  source_counts: Record<string, number>;
  highlights: string[];
  gaps: string[];
  metadata: Record<string, unknown>;
}

export interface ResearchMessage {
  message_id: string;
  role: 'assistant' | 'user' | 'system';
  kind: string;
  title: string;
  content: string;
  meta: string | null;
  created_at: string;
  citations: string[];
  payload: Record<string, unknown>;
}

export interface EvidenceItem {
  text: string;
  document_id: string | null;
  page_number: number | null;
  score: number | null;
}

export interface EvidenceBundle {
  evidences: EvidenceItem[];
}

export interface QAResponse {
  answer: string;
  question: string;
  evidence_bundle: EvidenceBundle;
  confidence: number | null;
  metadata: Record<string, unknown>;
}

export interface ResearchWorkspaceState {
  objective: string;
  current_stage: string;
  research_questions: string[];
  key_findings: string[];
  evidence_gaps: string[];
  next_actions: string[];
  status_summary: string;
}

export interface ResearchTask {
  task_id: string;
  topic: string;
  status: string;
  created_at: string;
  updated_at: string;
  paper_count: number;
  imported_document_ids: string[];
}

export interface ResearchAgentTraceStep {
  step_index: number;
  agent: string;
  thought: string;
  action_name: string;
  phase: string;
  action_input: Record<string, unknown>;
  status: string;
  observation: string;
  rationale: string;
  stop_signal: boolean;
  workspace_summary: string | null;
  metadata: Record<string, unknown>;
}

export interface ResearchAgentRunResponse {
  status: 'succeeded' | 'partial' | 'failed';
  task: ResearchTask | null;
  papers: PaperCandidate[];
  report: ResearchReport | null;
  import_result: {
    imported_count: number;
    skipped_count: number;
    failed_count: number;
  } | null;
  qa: QAResponse | null;
  chart: unknown | null;
  chart_graph_text: string | null;
  messages: ResearchMessage[];
  trace: ResearchAgentTraceStep[];
  warnings: string[];
  next_actions: string[];
  workspace: ResearchWorkspaceState;
  metadata: Record<string, unknown>;
}

export interface ResearchConversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  task_id: string | null;
  message_count: number;
  last_message_preview: string | null;
}

export interface ResearchConversationResponse {
  conversation: ResearchConversation;
  messages: ResearchMessage[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  papers?: PaperCandidate[];
  report?: ResearchReport;
  qa?: QAResponse;
  trace?: ResearchAgentTraceStep[];
  warnings?: string[];
  notices?: string[];
  nextActions?: string[];
  workspace?: ResearchWorkspaceState;
  backendMessages?: ResearchMessage[];
  status?: 'succeeded' | 'partial' | 'failed';
  isError?: boolean;
}

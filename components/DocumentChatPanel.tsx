"use client";

import { Copy, FileText, RefreshCcw, Send, UserRound } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { DOCUMENT_SKILL_OPTIONS, findAgentOption, REASONING_STYLE_OPTIONS } from "@/lib/agent-config";
import { normalizeMathDelimiters } from "@/lib/markdown";
import type { AskDocumentResponse, RequestState } from "@/lib/types";

export interface DocumentChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  pending?: boolean;
}

interface Props {
  documentId?: string | null;
  result?: AskDocumentResponse | null;
  state: RequestState;
  error?: string | null;
  disabledReason?: string | null;
  messages: DocumentChatMessage[];
  skillName: string;
  reasoningStyle: string;
  onSkillChange: (value: string) => void;
  onReasoningStyleChange: (value: string) => void;
  onAsk: (question: string) => void;
  onRetry?: (prompt: string) => void;
  onClear?: () => void;
}

function Bubble({ message, onRetry }: { message: DocumentChatMessage; onRetry?: (prompt: string) => void }) {
  const isUser = message.role === "user";
  const normalizedContent = normalizeMathDelimiters(message.content);
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`chat-pop flex max-w-[88%] gap-3 xl:max-w-[82%] ${isUser ? "flex-row-reverse" : ""}`}>
        <span className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl ${isUser ? "bg-indigo-100 text-accent" : "bg-slate-100 text-slate-600"}`}>
          {isUser ? <UserRound className="h-4 w-4" /> : <FileText className="h-4 w-4" />}
        </span>
        <div className={`min-w-0 rounded-2xl px-4 py-3 text-[13px] leading-6 ${isUser ? "rounded-br-sm bg-blue-600 text-white" : "rounded-bl-sm border border-gray-200 bg-white text-gray-800"}`}>
          <div className={`chart-markdown break-words whitespace-pre-wrap ${isUser ? "prose-invert" : ""}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {normalizedContent}
            </ReactMarkdown>
          </div>
          <div className={`mt-2 text-[11px] ${isUser ? "text-white/75" : "text-slate-400"}`}>{message.pending ? "思考中..." : message.createdAt}</div>
          {!isUser && !message.pending ? (
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
              <button type="button" onClick={() => navigator.clipboard.writeText(message.content).catch(() => undefined)} className="btn-ghost px-2 py-1">
                <span className="inline-flex items-center gap-1"><Copy className="h-3.5 w-3.5" />复制</span>
              </button>
              {onRetry ? (
                <button type="button" onClick={() => onRetry(`请基于文档证据，进一步细化这段回答：${message.content}`)} className="btn-ghost px-2 py-1">
                  <span className="inline-flex items-center gap-1"><RefreshCcw className="h-3.5 w-3.5" />延展</span>
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function DocumentChatPanel({
  documentId,
  result,
  state,
  error,
  disabledReason,
  messages,
  skillName,
  reasoningStyle,
  onSkillChange,
  onReasoningStyleChange,
  onAsk,
  onRetry,
  onClear
}: Props) {
  const [question, setQuestion] = useState("");
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const isLoading = state === "loading";
  const activeSkill = findAgentOption(DOCUMENT_SKILL_OPTIONS, skillName);
  const activeReasoning = findAgentOption(REASONING_STYLE_OPTIONS, reasoningStyle);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const suggestedQuestions = [
    "这篇文档的核心结论是什么？",
    "文档里最重要的实验结果是什么？",
    "作者主要解决了什么问题？",
    "有哪些关键指标或图表值得关注？"
  ];

  function submit() {
    const trimmed = question.trim();
    if (!trimmed || disabledReason || isLoading) return;
    onAsk(trimmed);
    setQuestion("");
  }

  return (
    <section className="flex h-[calc(100vh-170px)] min-h-[680px] max-h-[calc(100vh-170px)] flex-1 flex-col overflow-hidden rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex shrink-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="section-title flex items-center gap-2 text-[17px] tracking-tight"><FileText className="h-5 w-5 shrink-0 text-accent" />文档问答</h2>
          <p className="mt-1 break-words text-[12px] leading-5 text-muted">围绕已解析和已索引的文档进行连续问答，右侧可同步查看 evidence、trace 和 retrieval hits。</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="badge-muted max-w-full break-all">doc: {documentId ?? "none"}</span>
          {messages.length ? <button onClick={onClear} className="btn-ghost">清空</button> : null}
        </div>
      </div>

      {disabledReason ? <p className="mt-4 rounded-xl bg-amber-50 p-3 text-[12px] leading-5 text-amber-800">{disabledReason}</p> : null}
      {error ? <p className="mt-4 rounded-xl bg-red-50 p-3 text-[12px] leading-5 text-red-700">{error}</p> : null}

      <div className="mt-4 grid gap-3 rounded-2xl border border-line bg-white/90 p-3 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <label className="min-w-0">
          <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Skill</span>
          <select value={skillName} onChange={(event) => onSkillChange(event.target.value)} className="input-base w-full">
            {DOCUMENT_SKILL_OPTIONS.map((option) => (
              <option key={option.value || "auto"} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="mt-2 text-[11px] leading-5 text-slate-500">{activeSkill.hint}</p>
        </label>
        <label className="min-w-0">
          <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Reasoning</span>
          <select value={reasoningStyle} onChange={(event) => onReasoningStyleChange(event.target.value)} className="input-base w-full">
            {REASONING_STYLE_OPTIONS.map((option) => (
              <option key={option.value || "default"} value={option.value}>{option.label}</option>
            ))}
          </select>
          <p className="mt-2 text-[11px] leading-5 text-slate-500">{activeReasoning.hint}</p>
        </label>
      </div>

      <div ref={scrollRef} className="chat-scroll mt-3 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-lg border border-gray-100 bg-gray-50 p-4">
        {!messages.length ? (
          <div className="flex flex-wrap gap-2">
            {suggestedQuestions.map((item) => (
              <button key={item} type="button" onClick={() => setQuestion(item)} className="rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1.5 text-[12px] font-medium text-indigo-700 transition hover:bg-indigo-100">
                {item}
              </button>
            ))}
          </div>
        ) : null}
        {messages.length ? (
          messages.map((message) => <Bubble key={message.id} message={message} onRetry={onRetry} />)
        ) : (
          <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-indigo-200 bg-white/80 p-8 text-center text-sm leading-6 text-slate-500">
            上传并解析文档后，你可以像聊天一样继续提问，右侧会展示证据、检索结果和 trace。
          </div>
        )}
      </div>

      <div className="mt-3 shrink-0 rounded-lg border border-gray-200 bg-white p-2">
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          rows={3}
          placeholder="输入文档问题，Enter 发送，Shift+Enter 换行"
          className="w-full resize-none rounded-xl border-0 p-3 text-[14px] leading-6 outline-none"
        />
        <div className="flex items-center justify-between px-1 pb-1">
          <span className="min-w-0 flex-1 break-all text-[11px] leading-5 text-muted">当前回答置信度：{result?.qa.confidence ?? "empty"} · evidence：{result?.qa.evidence_bundle.evidences.length ?? 0}</span>
          <button disabled={Boolean(disabledReason) || !question.trim() || isLoading} onClick={submit} className="btn-accent px-5">
            <span className="inline-flex items-center gap-2"><Send className="h-4 w-4" />{isLoading ? "发送中..." : "发送"}</span>
          </button>
        </div>
      </div>
    </section>
  );
}

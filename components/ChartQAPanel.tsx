"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Bot, Copy, Image as ImageIcon, Maximize2, MessageCircle, Plus, RefreshCcw, Send, Sparkles, Trash2, UserRound, X } from "lucide-react";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { CHART_SKILL_OPTIONS, findAgentOption, REASONING_STYLE_OPTIONS } from "@/lib/agent-config";
import { normalizeMathDelimiters } from "@/lib/markdown";
import type { AskChartResponse, RequestState } from "@/lib/types";

export interface ChartChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  pending?: boolean;
}

interface Props {
  imagePath?: string | null;
  documentId?: string | null;
  pageId?: string | null;
  pageNumber?: number;
  chartId?: string | null;
  result?: AskChartResponse | null;
  previewUrl?: string | null;
  chartSummary?: {
    chartType?: string | null;
    summary?: string | null;
    xAxis?: string | null;
    yAxis?: string | null;
    series?: string[];
    confidence?: number | null;
  } | null;
  state: RequestState;
  error?: string | null;
  messages: ChartChatMessage[];
  sessionId?: string | null;
  fusedMode?: boolean;
  skillName: string;
  reasoningStyle: string;
  onFusedModeChange?: (enabled: boolean) => void;
  onSkillChange: (value: string) => void;
  onReasoningStyleChange: (value: string) => void;
  onAsk: (question: string) => void;
  onRetry?: (prompt: string) => void;
  onNewSession?: () => void;
  onClear?: () => void;
}

function ChatBubble({ message, onRetry }: { message: ChartChatMessage; onRetry?: (prompt: string) => void }) {
  const isUser = message.role === "user";
  const normalizedContent = normalizeMathDelimiters(message.content);
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`chat-pop flex max-w-[88%] gap-3 xl:max-w-[82%] ${isUser ? "flex-row-reverse" : ""}`}>
            <span className={`mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-2xl ${isUser ? "bg-indigo-100 text-accent" : "bg-slate-100 text-slate-600"}`}>
              {isUser ? <UserRound className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
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
              <button type="button" onClick={() => onRetry(`请基于同一张图，进一步细化这段回答：${message.content}`)} className="btn-ghost px-2 py-1">
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

export function ChartQAPanel({
  imagePath,
  documentId,
  pageId,
  pageNumber,
  chartId,
  result,
  previewUrl,
  chartSummary,
  state,
  error,
  messages,
  sessionId,
  fusedMode = false,
  skillName,
  reasoningStyle,
  onFusedModeChange,
  onSkillChange,
  onReasoningStyleChange,
  onAsk,
  onRetry,
  onNewSession,
  onClear
}: Props) {
  const [question, setQuestion] = useState("");
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxScale, setLightboxScale] = useState(1);
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const canAsk = Boolean(imagePath?.trim());
  const isLoading = state === "loading";
  const activeSkill = findAgentOption(CHART_SKILL_OPTIONS, skillName);
  const activeReasoning = findAgentOption(REASONING_STYLE_OPTIONS, reasoningStyle);

  useEffect(() => {
    const element = chatScrollRef.current;
    if (!element) return;
    element.scrollTo({ top: element.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!lightboxOpen) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setLightboxOpen(false);
      if (event.key === "0") setLightboxScale(1);
      if (event.key === "+" || event.key === "=") setLightboxScale((value) => Math.min(4, Number((value + 0.2).toFixed(2))));
      if (event.key === "-") setLightboxScale((value) => Math.max(0.4, Number((value - 0.2).toFixed(2))));
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [lightboxOpen]);

  function handleLightboxWheel(event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const delta = event.deltaY > 0 ? -0.12 : 0.12;
    setLightboxScale((value) => Math.min(4, Math.max(0.4, Number((value + delta).toFixed(2)))));
  }

  function openLightbox() {
    setLightboxScale(1);
    setLightboxOpen(true);
  }

  function submit() {
    const trimmed = question.trim();
    if (!trimmed || !canAsk || isLoading) return;
    onAsk(trimmed);
    setQuestion("");
  }

  const suggestedQuestions = [
    "蓝色曲线和红色曲线分别表示什么？",
    "哪条曲线整体更高？",
    "这张图想表达的核心结论是什么？",
    "图中有没有明显峰值或异常点？"
  ];

  return (
    <section className="flex h-[calc(100vh-170px)] min-h-[680px] max-h-[calc(100vh-170px)] flex-1 flex-col overflow-hidden rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex shrink-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="section-title flex items-center gap-2 text-[17px] tracking-tight"><MessageCircle className="h-5 w-5 shrink-0 text-accent" />图表追问</h2>
          <p className="mt-1 break-words text-[12px] leading-5 text-muted">围绕同一张图连续追问，支持公式、Markdown 和流式输出。</p>
          <label className="mt-3 inline-flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1.5 text-[12px] font-medium text-indigo-700">
            <input
              type="checkbox"
              checked={fusedMode}
              onChange={(event) => onFusedModeChange?.(event.target.checked)}
              className="h-3.5 w-3.5 accent-indigo-600"
            />
            联合参考文档问答
          </label>
          <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <label className="min-w-0">
              <span className="mb-1.5 block text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Chart skill</span>
              <select value={skillName} onChange={(event) => onSkillChange(event.target.value)} className="input-base w-full">
                {CHART_SKILL_OPTIONS.map((option) => (
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
              <p className="mt-2 text-[11px] leading-5 text-slate-500">{fusedMode ? activeReasoning.hint : "当前是纯图表追问，推理模式会在开启“联合参考文档问答”后用于 ask/fused。"}</p>
            </label>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="badge-muted max-w-full break-all">{chartId ?? "chart session"}</span>
          <button onClick={onNewSession} className="btn-ghost"><span className="inline-flex items-center gap-1"><Plus className="h-3.5 w-3.5" />新建会话</span></button>
          {messages.length ? (
            <button onClick={onClear} className="btn-ghost"><span className="inline-flex items-center gap-1"><Trash2 className="h-3.5 w-3.5" />清空</span></button>
          ) : null}
        </div>
      </div>

      <div className="mt-4 grid min-h-0 flex-1 overflow-hidden gap-4 lg:grid-cols-[330px_minmax(0,1fr)]">
        <aside className="chart-chat-scroll min-h-0 h-full overflow-y-auto rounded-2xl border border-line bg-surface/70 p-3 text-xs text-slate-600">
          <div className="overflow-hidden rounded-2xl border border-white bg-white shadow-sm">
            {previewUrl ? (
              <button type="button" onClick={openLightbox} className="group relative block w-full bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewUrl} alt="chart preview" className="h-64 w-full object-contain bg-white transition-transform duration-300 group-hover:scale-[1.02]" />
                <span className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full bg-white/90 px-2 py-1 text-[11px] font-medium text-slate-700 shadow-sm ring-1 ring-slate-200 backdrop-blur">
                  <Maximize2 className="h-3.5 w-3.5" /> 点击放大
                </span>
              </button>
            ) : (
              <div className="flex h-64 flex-col items-center justify-center gap-2 text-xs text-gray-400"><ImageIcon className="h-6 w-6" />暂无图片预览</div>
            )}
          </div>
          <details className="mt-3 rounded-2xl border border-white bg-white p-4 shadow-sm" open>
            <summary className="cursor-pointer text-[13px] font-semibold text-ink"><span className="inline-flex items-center gap-2"><Sparkles className="h-4 w-4 text-accent" />图表理解侧栏</span></summary>
            <div className="mt-3 grid gap-2 text-[12px] leading-5 text-slate-600">
              <div className="break-all">image: {imagePath ?? "none"}</div>
              <div className="break-all">document: {documentId ?? "none"}</div>
              <div>page: {pageId ?? "page-1"} / {pageNumber ?? 1}</div>
              <div className="break-all">session: {sessionId ?? result?.session_id ?? "new"}</div>
              <div>chart_type: {chartSummary?.chartType ?? "unknown"}</div>
              <div className="break-words">x_axis: {chartSummary?.xAxis ?? "empty"}</div>
              <div className="break-words">y_axis: {chartSummary?.yAxis ?? "empty"}</div>
              <div className="break-words">series: {(chartSummary?.series ?? []).join(", ") || "empty"}</div>
              <div>confidence: {chartSummary?.confidence ?? "empty"}</div>
              <div>fused_mode: {fusedMode ? "enabled" : "disabled"}</div>
              <div>skill: {activeSkill.label}</div>
              <div>reasoning: {reasoningStyle || "default graph"}</div>
              <div className="rounded-xl bg-surface p-3 text-[12px] leading-6 text-slate-500">{chartSummary?.summary ?? "还没有图表理解结果。"}</div>
            </div>
          </details>
        </aside>

        <div className="flex min-h-0 h-full flex-col overflow-hidden">
          {!canAsk ? (
            <p className="mb-4 rounded-xl bg-amber-50 p-3 text-[12px] leading-5 text-amber-800">请先上传并选择一张图表图片，再开始提问。</p>
          ) : null}

          {error ? <p className="mb-4 rounded-xl bg-red-50 p-3 text-[12px] leading-5 text-red-700">{error}</p> : null}

          <div ref={chatScrollRef} className="chat-scroll flex min-h-0 h-full flex-1 flex-col gap-3 overflow-y-auto rounded-lg border border-gray-100 bg-gray-50 p-4">
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
              messages.map((message) => <ChatBubble key={message.id} message={message} onRetry={onRetry} />)
            ) : (
              <div className="flex flex-1 items-center justify-center rounded-2xl border border-dashed border-indigo-200 bg-white/80 p-8 text-center text-sm leading-6 text-slate-500">
                <span className="inline-flex flex-col items-center gap-3"><MessageCircle className="h-8 w-8 text-accent" />上传图表后，可以直接问：蓝色曲线表示什么？峰值在哪里？两条曲线有什么差异？</span>
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
              placeholder="输入图表问题，Enter 发送，Shift+Enter 换行"
              className="w-full resize-none rounded-xl border-0 p-3 text-[14px] leading-6 outline-none"
            />
            <div className="flex items-center justify-between px-1 pb-1">
              <span className="min-w-0 flex-1 break-all text-[11px] leading-5 text-muted">session: {sessionId ?? result?.session_id ?? "new"} · history: {String(result?.metadata?.history_turn_count ?? messages.filter((item) => item.role === "assistant").length)}</span>
              <button disabled={!canAsk || !question.trim() || isLoading} onClick={submit} className="btn-accent px-5">
                <span className="inline-flex items-center gap-2"><Send className="h-4 w-4" />{isLoading ? "发送中..." : "发送"}</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {lightboxOpen && previewUrl ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm" onClick={() => setLightboxOpen(false)}>
          <div className="relative max-h-[92vh] w-full max-w-6xl rounded-3xl bg-white p-3 shadow-2xl" onClick={(event) => event.stopPropagation()}>
            <div className="mb-3 flex items-center justify-between px-2 pt-1">
              <div>
                <p className="text-sm font-semibold text-slate-800">图表大图预览</p>
                <p className="text-xs text-slate-500">ESC 关闭，滚轮缩放，0 重置</p>
              </div>
              <div className="flex items-center gap-2">
                <span className="badge-muted">{Math.round(lightboxScale * 100)}%</span>
                <button type="button" onClick={() => setLightboxScale((value) => Math.max(0.4, Number((value - 0.2).toFixed(2))))} className="btn-ghost">-</button>
                <button type="button" onClick={() => setLightboxScale(1)} className="btn-ghost">重置</button>
                <button type="button" onClick={() => setLightboxScale((value) => Math.min(4, Number((value + 0.2).toFixed(2))))} className="btn-ghost">+</button>
                <button type="button" onClick={() => setLightboxOpen(false)} className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-line text-slate-600 transition hover:bg-slate-50">
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div onWheel={handleLightboxWheel} className="chart-chat-scroll max-h-[82vh] overflow-auto rounded-2xl bg-slate-50 p-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt="chart preview enlarged"
                className="mx-auto max-h-[78vh] w-auto max-w-full origin-center object-contain transition-transform duration-150"
                style={{ transform: `scale(${lightboxScale})` }}
              />
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

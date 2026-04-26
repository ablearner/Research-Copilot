"use client";

import { useState } from "react";
import type { AskDocumentResponse, RequestState } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";

interface Props {
  documentId?: string | null;
  result?: AskDocumentResponse | null;
  state: RequestState;
  error?: string | null;
  disabledReason?: string | null;
  onAsk: (question: string) => void;
}

export function QAPanel({ documentId, result, state, error, disabledReason, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const isDisabled = Boolean(disabledReason) || !question.trim() || state === "loading";
  return (
    <section className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="card">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-base font-bold text-ink">文档问答</h2>
          <span className="badge-muted max-w-[260px] truncate">doc: {documentId ?? "none"}</span>
        </div>
        <textarea value={question} onChange={(event) => setQuestion(event.target.value)} rows={6} placeholder="输入你的问题，例如：这篇文档的核心结论是什么？" className="input-base mt-4 w-full resize-none" />
        {disabledReason ? <p className="mt-3 rounded-xl bg-amber-50 p-3 text-xs text-amber-800">{disabledReason}</p> : null}
        <button disabled={isDisabled} onClick={() => onAsk(question.trim())} className="btn-accent mt-4">
          {state === "loading" ? "正在提问..." : "提问文档"}
        </button>
        {error ? <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">{error}</p> : null}
      </div>
      <AnswerCard qa={result?.qa} />
    </section>
  );
}

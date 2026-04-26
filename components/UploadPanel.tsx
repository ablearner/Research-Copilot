"use client";

import { useState } from "react";
import { FileUp, UploadCloud } from "lucide-react";
import type { RequestState, UploadDocumentResponse } from "@/lib/types";

interface Props {
  result?: UploadDocumentResponse | null;
  state: RequestState;
  error?: string | null;
  onUpload: (file: File) => void;
  mode: "document" | "chart";
}

export function UploadPanel({ result, state, error, onUpload, mode }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const accept = mode === "chart" ? "image/png,image/jpeg,image/webp,image/tiff" : ".pdf,image/png,image/jpeg,image/webp,image/tiff";
  const title = mode === "chart" ? "上传图表图片" : "上传文档或图片";
  const hint = mode === "chart" ? "选择图表图片，随后可做图表理解与追问。" : "选择 PDF 或图片，随后可做解析、索引与问答。";
  return (
    <section className="card">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-accent"><UploadCloud className="h-4 w-4" /></span>
            <h2 className="break-words text-[15px] font-semibold leading-6 text-ink">{title}</h2>
          </div>
          <p className="mt-1 break-words text-[12px] leading-5 text-muted">{hint}</p>
        </div>
        <span className="badge-info shrink-0">{mode === "chart" ? "Chart" : "Doc"}</span>
      </div>
      <input
        className="mt-4 block w-full rounded-xl border border-dashed border-indigo-200 bg-indigo-50/40 p-3 text-sm text-slate-600 file:mr-3 file:rounded-lg file:border-0 file:bg-accent file:px-3 file:py-2 file:text-sm file:font-semibold file:text-white hover:border-indigo-300"
        type="file"
        accept={accept}
        onChange={(event) => setFile(event.target.files?.[0] ?? null)}
      />
      <button
        className="btn-accent mt-3 w-full"
        disabled={!file || state === "loading"}
        onClick={() => file && onUpload(file)}
      >
        <span className="inline-flex items-center justify-center gap-2"><FileUp className="h-4 w-4" />{state === "loading" ? "Uploading..." : "Upload"}</span>
      </button>
      {file ? <p className="mt-3 truncate text-[12px] leading-5 text-slate-500">Selected: {file.name}</p> : null}
      {result ? <p className="mt-3 rounded-xl bg-emerald-50 px-3 py-2 text-[12px] font-medium leading-5 text-emerald-700 break-words">Uploaded: {result.filename}</p> : null}
      {error ? <p className="mt-3 rounded-xl bg-red-50 p-3 text-[12px] leading-5 text-red-700">{error}</p> : null}
    </section>
  );
}

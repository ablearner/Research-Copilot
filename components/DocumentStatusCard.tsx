"use client";

import type { ParseDocumentResponse, UploadDocumentResponse } from "@/lib/types";

interface Props {
  uploadResult?: UploadDocumentResponse | null;
  parseResult?: ParseDocumentResponse | null;
}

function Row({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="kv-row">
      <dt className="kv-label">{label}</dt>
      <dd className="kv-value font-medium text-slate-800">{value ?? "empty"}</dd>
    </div>
  );
}

export function DocumentStatusCard({ uploadResult, parseResult }: Props) {
  const parsed = parseResult?.parsed_document;
  const textBlocks = parsed?.pages.reduce((sum, page) => sum + page.text_blocks.length, 0) ?? 0;
  return (
    <section className="card">
      <div className="flex items-center justify-between gap-3">
        <h2 className="section-title">当前资源状态</h2>
        <span className="badge-info">Resource</span>
      </div>
      {!uploadResult ? <p className="mt-3 section-subtitle">No uploaded document yet.</p> : null}
      <dl className="mt-3">
        <Row label="document_id" value={uploadResult?.document_id ?? parsed?.id} />
        <Row label="filename" value={uploadResult?.filename ?? parsed?.filename} />
        <Row label="storage_uri" value={uploadResult?.storage_uri} />
        <Row label="upload" value={uploadResult?.status} />
        <Row label="parse" value={parseResult?.status ?? parsed?.status} />
        <Row label="pages" value={parsed?.pages.length} />
        <Row label="text_blocks" value={parsed ? textBlocks : undefined} />
      </dl>
      {uploadResult?.error_message || parseResult?.error_message ? (
        <p className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">{uploadResult?.error_message ?? parseResult?.error_message}</p>
      ) : null}
    </section>
  );
}

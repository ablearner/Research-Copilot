"use client";

interface DebugItem {
  label: string;
  value: unknown;
}

export function JsonDebugPanel({ items }: { items: DebugItem[] }) {
  return (
    <section className="card">
      <h2 className="text-sm font-bold text-ink">Raw JSON</h2>
      <div className="mt-3 space-y-3">
        {items.map((item) => (
          <details key={item.label} className="overflow-hidden rounded-2xl border border-line bg-white shadow-sm">
            <summary className="cursor-pointer bg-surface px-4 py-3 text-xs font-semibold text-slate-700">{item.label}</summary>
            <pre className="max-h-80 overflow-auto bg-slate-950 p-4 text-xs leading-5 text-slate-100">{JSON.stringify(item.value ?? null, null, 2)}</pre>
          </details>
        ))}
      </div>
    </section>
  );
}

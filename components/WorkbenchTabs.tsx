"use client";

interface Props {
  mode: "document" | "chart";
  onChange: (mode: "document" | "chart") => void;
}

const tabStyle = {
  active: "bg-white text-accent shadow-sm ring-1 ring-indigo-100",
  idle: "bg-transparent text-slate-500 hover:bg-white/70 hover:text-ink"
};

export function WorkbenchTabs({ mode, onChange }: Props) {
  return (
    <div className="inline-flex rounded-lg border border-gray-200 bg-white p-1">
      <button
        onClick={() => onChange("document")}
        className={`rounded-xl px-4 py-2 text-sm font-semibold transition-all ${mode === "document" ? tabStyle.active : tabStyle.idle}`}
      >
        文档模式
      </button>
      <button
        onClick={() => onChange("chart")}
        className={`rounded-xl px-4 py-2 text-sm font-semibold transition-all ${mode === "chart" ? tabStyle.active : tabStyle.idle}`}
      >
        图表模式
      </button>
    </div>
  );
}

"use client";

import type { ReactNode } from "react";
import { ChevronDown, type LucideIcon } from "lucide-react";

interface Props {
  title: string;
  subtitle?: string;
  icon?: LucideIcon;
  badge?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function InspectorPanel({ title, subtitle, icon: Icon, badge, defaultOpen = true, children }: Props) {
  return (
    <details className="group overflow-hidden rounded-lg border border-gray-200 bg-white" open={defaultOpen}>
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-4 marker:hidden">
        <div className="flex min-w-0 items-center gap-3">
          {Icon ? (
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-50 text-accent">
              <Icon className="h-4 w-4" />
            </span>
          ) : null}
          <div className="min-w-0">
            <h3 className="truncate text-sm font-bold text-ink">{title}</h3>
            {subtitle ? <p className="mt-0.5 truncate text-xs text-muted">{subtitle}</p> : null}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {badge ? <span className="badge-info">{badge}</span> : null}
          <ChevronDown className="h-4 w-4 text-muted transition-transform group-open:rotate-180" />
        </div>
      </summary>
      <div className="border-t border-line px-5 pb-5 pt-4">{children}</div>
    </details>
  );
}

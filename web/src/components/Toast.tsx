import { useEffect, useState } from 'react';
import { X, CheckCircle, AlertTriangle, Info } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  text: string;
}

const ICONS: Record<ToastType, typeof Info> = {
  success: CheckCircle,
  error: AlertTriangle,
  info: Info,
};

const COLORS: Record<ToastType, string> = {
  success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  error: 'bg-red-50 text-red-700 border-red-200',
  info: 'bg-accent-50 text-accent-700 border-accent-200',
};

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: ToastMessage;
  onDismiss: (id: string) => void;
}) {
  const Icon = ICONS[toast.type];

  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), 4000);
    return () => clearTimeout(timer);
  }, [toast.id, onDismiss]);

  return (
    <div
      className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border shadow-sm text-sm animate-slide-up ${COLORS[toast.type]}`}
    >
      <Icon size={14} className="flex-shrink-0" />
      <span className="flex-1">{toast.text}</span>
      <button
        onClick={() => onDismiss(toast.id)}
        className="p-0.5 rounded hover:bg-black/5 transition-colors"
      >
        <X size={12} />
      </button>
    </div>
  );
}

interface ToastContainerProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-20 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

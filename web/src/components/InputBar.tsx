import { useState, useRef, useEffect } from 'react';
import { ArrowUp, Square } from 'lucide-react';
import { FileUpload, FilePreview } from './FileUpload';

interface InputBarProps {
  onSend: (message: string, file?: File) => void;
  isLoading: boolean;
  onStop?: () => void;
}

export function InputBar({ onSend, isLoading, onStop }: InputBarProps) {
  const [input, setInput] = useState('');
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed && !pendingFile) return;
    if (isLoading) return;
    onSend(trimmed || (pendingFile ? `解析文件: ${pendingFile.name}` : ''), pendingFile || undefined);
    setInput('');
    setPendingFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
    if (e.key === 'Escape' && onStop && isLoading) {
      onStop();
    }
  };

  return (
    <div className="border-t border-paper-200 bg-paper-50/80 backdrop-blur-sm px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {pendingFile && (
          <div className="mb-2">
            <FilePreview file={pendingFile} onRemove={() => setPendingFile(null)} />
          </div>
        )}
        <div className="relative flex items-end rounded-2xl border border-paper-300 bg-white shadow-sm focus-within:ring-2 focus-within:ring-accent-200 focus-within:border-accent-400 transition-all">
          <FileUpload
            onFileSelected={(file) => setPendingFile(file)}
            disabled={isLoading}
          />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Research Copilot…"
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none bg-transparent px-4 py-3 text-sm text-ink-700 placeholder:text-ink-300 focus:outline-none disabled:opacity-50 min-h-[44px] max-h-[200px]"
          />
          {isLoading ? (
            <button
              onClick={onStop}
              className="m-1.5 p-2 rounded-xl bg-ink-700 text-white hover:bg-ink-800 transition-colors"
              title="Stop (Esc)"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!input.trim() && !pendingFile}
              className="m-1.5 p-2 rounded-xl bg-accent-700 text-white hover:bg-accent-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
        <div className="text-center mt-2 text-xs text-ink-300 select-none">
          Research Copilot may produce inaccurate information. Verify important
          findings.
        </div>
      </div>
    </div>
  );
}

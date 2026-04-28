import { useState, useRef, useCallback } from 'react';
import { Paperclip, X, FileText, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export function FileUpload({ onFileSelected, disabled }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (file.type !== 'application/pdf') return;
      onFileSelected(file);
    },
    [onFileSelected],
  );

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => {
          handleFile(e.target.files?.[0]);
          if (inputRef.current) inputRef.current.value = '';
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        className="p-2 rounded-xl text-ink-400 hover:text-accent-600 hover:bg-paper-100 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        title="Upload PDF"
      >
        <Paperclip size={16} />
      </button>
    </>
  );
}

interface FilePreviewProps {
  file: File;
  uploading?: boolean;
  onRemove: () => void;
}

export function FilePreview({ file, uploading, onRemove }: FilePreviewProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-paper-100 rounded-lg text-xs text-ink-500">
      {uploading ? (
        <Loader2 size={12} className="animate-spin text-accent-500" />
      ) : (
        <FileText size={12} className="text-accent-500" />
      )}
      <span className="truncate max-w-[200px]">{file.name}</span>
      <span className="text-ink-300">
        ({(file.size / 1024).toFixed(0)} KB)
      </span>
      {!uploading && (
        <button
          onClick={onRemove}
          className="p-0.5 rounded hover:bg-paper-200 text-ink-300 hover:text-ink-500 transition-colors"
        >
          <X size={12} />
        </button>
      )}
    </div>
  );
}

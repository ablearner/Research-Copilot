import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { PaperCard } from './PaperCard';
import type { ChatMessage } from '../types';
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Lightbulb,
  BookOpen,
  Layers,
  Info,
  ZoomIn,
} from 'lucide-react';

// Convert .data/storage/... paths to serveable /files/ URLs
const IMAGE_PATH_RE = /图片路径[：:]\s*(\.data\/storage\/[^\s]+)/g;
const STORAGE_PREFIX_RE = /^\.data\/storage\//;

function storagePathToUrl(p: string): string {
  return '/api/files/' + p.replace(STORAGE_PREFIX_RE, '');
}

function extractImages(message: ChatMessage): string[] {
  const urls: string[] = [];
  const seen = new Set<string>();

  // From content text
  const matches = message.content.matchAll(IMAGE_PATH_RE);
  for (const m of matches) {
    const url = storagePathToUrl(m[1]);
    if (!seen.has(url)) { seen.add(url); urls.push(url); }
  }

  // From payload.image_path in backend messages
  if (message.backendMessages) {
    for (const bm of message.backendMessages) {
      const imgPath = (bm.payload?.image_path as string) || '';
      if (imgPath && imgPath.startsWith('.data/storage/')) {
        const url = storagePathToUrl(imgPath);
        if (!seen.has(url)) { seen.add(url); urls.push(url); }
      }
    }
  }
  return urls;
}

function stripImagePaths(content: string): string {
  return content.replace(IMAGE_PATH_RE, '').trim();
}

function AssistantContent({ message }: { message: ChatMessage }) {
  const images = useMemo(() => extractImages(message), [message]);
  const cleanContent = useMemo(() => stripImagePaths(message.content), [message.content]);
  return (
    <>
      {cleanContent && (
        <div className="markdown-content">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {cleanContent}
          </ReactMarkdown>
        </div>
      )}
      {images.length > 0 && (
        <div className="mt-3 space-y-3">
          {images.map((url, i) => (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="block group relative rounded-xl overflow-hidden border border-paper-200 hover:border-accent-300 transition-colors max-w-lg"
            >
              <img
                src={url}
                alt={`Figure ${i + 1}`}
                className="w-full h-auto"
                loading="lazy"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
              <div className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/40 text-white opacity-0 group-hover:opacity-100 transition-opacity">
                <ZoomIn size={14} />
              </div>
            </a>
          ))}
        </div>
      )}
    </>
  );
}

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  if (message.role === 'user') {
    return <UserBubble content={message.content} />;
  }
  return <AssistantBubble message={message} />;
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end animate-slide-up">
      <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-md bg-accent-700 text-white text-sm leading-relaxed whitespace-pre-wrap">
        {content}
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: ChatMessage }) {
  const [showPapers, setShowPapers] = useState(false);
  const [showTrace, setShowTrace] = useState(true);
  const [showNotices, setShowNotices] = useState(true);

  const hasPapers = message.papers && message.papers.length > 0;
  const hasTrace = message.trace && message.trace.length > 0;
  const hasWarnings = message.warnings && message.warnings.length > 0;
  const hasNotices = message.notices && message.notices.length > 0;
  const hasNextActions = message.nextActions && message.nextActions.length > 0;

  return (
    <div className="animate-slide-up">
      {/* Main content */}
      <div
        className={`text-sm leading-relaxed ${message.isError ? 'text-red-600' : 'text-ink-700'}`}
      >
        {message.isError ? (
          <div className="flex items-start gap-2 px-4 py-3 bg-red-50 rounded-xl border border-red-100">
            <AlertTriangle
              size={16}
              className="flex-shrink-0 mt-0.5 text-red-400"
            />
            <span>{message.content}</span>
          </div>
        ) : (
          <AssistantContent message={message} />
        )}
      </div>

      {/* Warnings */}
      {hasWarnings && (
        <div className="mt-3 space-y-1.5">
          {message.warnings!.map((w, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs text-amber-700 bg-amber-50 px-3 py-2 rounded-lg"
            >
              <AlertTriangle size={12} className="flex-shrink-0 mt-0.5" />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* Papers */}
      {hasPapers && (
        <div className="mt-4">
          <button
            onClick={() => setShowPapers(!showPapers)}
            className="flex items-center gap-2 text-sm font-medium text-accent-700 hover:text-accent-800 transition-colors"
          >
            <BookOpen size={14} />
            <span>
              {message.papers!.length} paper
              {message.papers!.length > 1 ? 's' : ''} found
            </span>
            {showPapers ? (
              <ChevronDown size={14} />
            ) : (
              <ChevronRight size={14} />
            )}
          </button>
          {showPapers && (
            <div className="mt-3 space-y-2 animate-fade-in">
              {message.papers!.map((paper) => (
                <PaperCard key={paper.paper_id} paper={paper} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Next actions */}
      {hasNextActions && (
        <div className="mt-4">
          <div className="flex items-center gap-1.5 text-xs font-medium text-ink-400 mb-2">
            <Lightbulb size={12} />
            <span>Suggested next steps</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {message.nextActions!.map((action, i) => (
              <span
                key={i}
                className="px-3 py-1.5 text-xs bg-accent-50 text-accent-700 rounded-full border border-accent-100"
              >
                {action}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Notices (internal agent activity) */}
      {hasNotices && (
        <div className="mt-4">
          <button
            onClick={() => setShowNotices(!showNotices)}
            className="flex items-center gap-2 text-xs font-medium text-ink-300 hover:text-ink-500 transition-colors"
          >
            <Info size={12} />
            <span>
              Agent activity ({message.notices!.length} item
              {message.notices!.length > 1 ? 's' : ''})
            </span>
            {showNotices ? (
              <ChevronDown size={12} />
            ) : (
              <ChevronRight size={12} />
            )}
          </button>
          {showNotices && (
            <div className="mt-2 space-y-1 animate-fade-in">
              {message.notices!.map((notice, i) => (
                <div
                  key={i}
                  className="text-xs px-3 py-2 bg-paper-100 rounded-lg text-ink-400"
                >
                  {notice}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Trace */}
      {hasTrace && (
        <div className="mt-4">
          <button
            onClick={() => setShowTrace(!showTrace)}
            className="flex items-center gap-2 text-xs font-medium text-ink-300 hover:text-ink-500 transition-colors"
          >
            <Layers size={12} />
            <span>
              Agent trace ({message.trace!.length} step
              {message.trace!.length > 1 ? 's' : ''})
            </span>
            {showTrace ? (
              <ChevronDown size={12} />
            ) : (
              <ChevronRight size={12} />
            )}
          </button>
          {showTrace && (
            <div className="mt-2 space-y-1 animate-fade-in">
              {message.trace!.map((step, idx) => (
                <div
                  key={`${step.step_index}-${idx}`}
                  className="flex items-start gap-2 text-xs px-3 py-2 bg-paper-100 rounded-lg"
                >
                  <span className="font-mono text-ink-300 flex-shrink-0 w-5 text-right">
                    {step.step_index}
                  </span>
                  <div className="min-w-0">
                    <span className="font-medium text-ink-500">
                      {step.action_name}
                    </span>
                    {step.thought && (
                      <span className="text-ink-400 ml-1">
                        — {step.thought}
                      </span>
                    )}
                    <span
                      className={`ml-1.5 inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${
                        step.status === 'succeeded'
                          ? 'bg-emerald-50 text-emerald-600'
                          : step.status === 'failed'
                            ? 'bg-red-50 text-red-600'
                            : 'bg-paper-200 text-ink-400'
                      }`}
                    >
                      {step.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

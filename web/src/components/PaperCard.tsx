import type { PaperCandidate } from '../types';
import { ExternalLink, Calendar, Quote, Users } from 'lucide-react';

interface PaperCardProps {
  paper: PaperCandidate;
  selectable?: boolean;
  selected?: boolean;
  onToggle?: (paperId: string) => void;
}

export function PaperCard({ paper, selectable, selected, onToggle }: PaperCardProps) {
  const url =
    paper.url ||
    paper.pdf_url ||
    (paper.doi ? `https://doi.org/${paper.doi}` : null);

  return (
    <div className={`px-4 py-3 rounded-xl border transition-colors ${selected ? 'border-accent-400 bg-accent-50/50' : 'border-paper-200 bg-white/60 hover:bg-white'}`}>
      <div className="flex items-start justify-between gap-3">
        {selectable && (
          <input
            type="checkbox"
            checked={selected || false}
            onChange={() => onToggle?.(paper.paper_id)}
            className="mt-1 h-4 w-4 rounded border-paper-300 text-accent-600 focus:ring-accent-500 flex-shrink-0 cursor-pointer"
          />
        )}
        <div className="min-w-0 flex-1">
          <h4 className="font-display text-sm font-semibold text-ink-700 leading-snug">
            {url ? (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-accent-700 transition-colors"
              >
                {paper.title}
              </a>
            ) : (
              paper.title
            )}
          </h4>

          {paper.authors.length > 0 && (
            <div className="flex items-center gap-1 mt-1 text-xs text-ink-400">
              <Users size={10} className="flex-shrink-0" />
              <span className="truncate">
                {paper.authors.slice(0, 3).join(', ')}
                {paper.authors.length > 3 &&
                  ` +${paper.authors.length - 3}`}
              </span>
            </div>
          )}

          <div className="flex items-center gap-3 mt-1.5 text-xs text-ink-300">
            {paper.year && (
              <span className="flex items-center gap-1">
                <Calendar size={10} />
                {paper.year}
              </span>
            )}
            {paper.venue && <span>{paper.venue}</span>}
            {paper.citations != null && paper.citations > 0 && (
              <span className="flex items-center gap-1">
                <Quote size={10} />
                {paper.citations}
              </span>
            )}
            <span className="px-1.5 py-0.5 rounded bg-paper-200 text-ink-400 text-[10px] font-medium uppercase">
              {paper.source}
            </span>
          </div>

          {paper.summary && (
            <p className="mt-2 text-xs text-ink-400 leading-relaxed line-clamp-2">
              {paper.summary}
            </p>
          )}
        </div>

        {url && (
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 p-1.5 rounded-lg hover:bg-paper-200 text-ink-300 hover:text-accent-600 transition-colors"
            title="Open paper"
          >
            <ExternalLink size={14} />
          </a>
        )}
      </div>
    </div>
  );
}

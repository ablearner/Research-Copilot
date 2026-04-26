import { useState, useRef } from 'react';
import {
  BookOpen,
  Search,
  FileText,
  BarChart3,
  ArrowUp,
} from 'lucide-react';

interface WelcomeScreenProps {
  onStartChat: (message: string) => void;
}

const suggestions = [
  {
    icon: Search,
    label: 'Discover papers',
    prompt: 'Find recent papers on large language model reasoning',
  },
  {
    icon: FileText,
    label: 'Research survey',
    prompt: 'Write a survey on retrieval-augmented generation techniques',
  },
  {
    icon: BarChart3,
    label: 'Compare methods',
    prompt:
      'Compare different approaches to knowledge graph construction from text',
  },
  {
    icon: BookOpen,
    label: 'Deep analysis',
    prompt:
      'Analyze the latest advances in multi-agent systems for scientific research',
  },
];

export function WelcomeScreen({ onStartChat }: WelcomeScreenProps) {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    onStartChat(trimmed);
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="h-full flex flex-col items-center justify-center px-6 animate-fade-in">
      <div className="w-full max-w-2xl">
        {/* Hero */}
        <div className="text-center mb-10">
          <h1 className="font-display text-4xl font-semibold text-ink-800 mb-3 tracking-tight">
            What would you like to research?
          </h1>
          <p className="text-ink-400 text-base leading-relaxed max-w-lg mx-auto">
            Ask a research question, discover papers, or explore a topic.
            <br />
            The agent will decide the best approach.
          </p>
        </div>

        {/* Input */}
        <div className="relative mb-8">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about research…"
            rows={3}
            className="w-full resize-none rounded-2xl border border-paper-300 bg-white px-5 py-4 pr-14 text-base text-ink-700 placeholder:text-ink-300 focus:outline-none focus:ring-2 focus:ring-accent-200 focus:border-accent-400 transition-all shadow-sm"
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim()}
            className="absolute right-3 bottom-3 p-2 rounded-xl bg-accent-700 text-white hover:bg-accent-800 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            <ArrowUp size={18} />
          </button>
        </div>

        {/* Suggestions */}
        <div className="grid grid-cols-2 gap-3">
          {suggestions.map((s) => (
            <button
              key={s.label}
              onClick={() => onStartChat(s.prompt)}
              className="flex items-start gap-3 px-4 py-3.5 rounded-xl border border-paper-200 bg-white/60 hover:bg-white hover:border-paper-300 hover:shadow-sm transition-all text-left group"
            >
              <s.icon
                size={16}
                className="flex-shrink-0 mt-0.5 text-accent-500 group-hover:text-accent-600 transition-colors"
              />
              <div>
                <div className="text-sm font-medium text-ink-600 group-hover:text-ink-700">
                  {s.label}
                </div>
                <div className="text-xs text-ink-300 mt-0.5 line-clamp-2">
                  {s.prompt}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

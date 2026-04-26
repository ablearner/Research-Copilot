const mathTokenPattern =
  /(?:[A-Za-z][A-Za-z0-9]*|[A-Za-z]_[A-Za-z0-9{}\\]+|[A-Za-z]\^\{?[^\s()，。；;,.]+\}?|[A-Za-z]+_\{[^\n]+?\}|\\[A-Za-z]+|[α-ωΑ-ΩσμλΩΔ][\w{}^_\\]*)/;

function looksLikeMath(expr: string) {
  const trimmed = expr.trim();
  if (!trimmed || trimmed.length > 120) return false;
  if (/[。！？；，：]/.test(trimmed)) return false;
  if (/\s{2,}/.test(trimmed)) return false;
  if (!mathTokenPattern.test(trimmed)) return false;
  return /[=+\-*/^_\\{}]|\b(?:P|q|d|M|res|sigma|theta|lambda|mu|alpha|beta)\b|[α-ωΑ-ΩσμλΩΔ]/.test(trimmed);
}

export function normalizeMathDelimiters(content: string) {
  let normalized = content
    .replace(/\\\[([\s\S]*?)\\\]/g, "$$$$$1$$$$")
    .replace(/\\\(([\s\S]*?)\\\)/g, (_match, expr: string) => `$${expr.trim()}$`)
    .replace(/(?<!\\)\$([^$\n]+?)\$/g, (_match, expr: string) => `$${expr.trim()}$`)
    .replace(/\(\s*([A-Za-zΑ-Ωα-ω\\][^\n，。；;]{0,120}?)\s*\)/g, (match, expr: string) => {
      const trimmed = expr.trim();
      return looksLikeMath(trimmed) ? `$${trimmed}$` : match;
    });

  const inlineFormulas = [
    /P_\{\\text\{res\}\}\(q\)/g,
    /P_\{res\}\(q\)/g,
    /P\(q\)/g,
    /P_res\(q\)/g,
    /σ\^2_\{?k\}?/g,
  ];

  for (const pattern of inlineFormulas) {
    normalized = normalized.replace(pattern, (match, offset: number, source: string) => {
      const previous = source[Math.max(0, offset - 1)];
      const next = source[offset + match.length];
      if (previous === "$" || next === "$") return match;
      return `$${match}$`;
    });
  }

  return normalized;
}

export function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

export function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

export function buildListKey(
  scope: string,
  value: string | number | null | undefined,
  index: number
): string {
  const normalized =
    typeof value === "string" ? value.trim() || "empty" : String(value ?? "empty");
  return `${scope}:${index}:${normalized}`;
}

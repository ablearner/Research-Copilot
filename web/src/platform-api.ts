const base = "/api/backend/api/v1";
export async function api<T = any>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  const csrf = document.cookie
    .split("; ")
    .find((x) => x.startsWith("rc_csrf="))
    ?.split("=")[1];
  if (csrf) headers.set("X-CSRF-Token", decodeURIComponent(csrf));
  if (options.body && !(options.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  const r = await fetch(base + path, {
    ...options,
    headers,
    credentials: "same-origin",
  });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(
      typeof d.detail === "string"
        ? d.detail
        : d.code || `请求失败 (${r.status})`,
    );
  }
  return r.status === 204 ? (undefined as T) : r.json();
}
export const post = (body: unknown = {}) => ({
  method: "POST",
  body: JSON.stringify(body),
});
export const resourceUrl = (path: string) => base + path;

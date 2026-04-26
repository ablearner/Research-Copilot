import type { ResearchPaperFigurePreview } from "./types";

export function buildResearchFigurePreviewHref(
  figure: ResearchPaperFigurePreview
): string | null {
  if (
    typeof figure.preview_data_url === "string" &&
    figure.preview_data_url.trim()
  ) {
    return figure.preview_data_url.trim();
  }
  if (typeof figure.image_path !== "string" || !figure.image_path.trim()) {
    return null;
  }
  const normalizedPath = figure.image_path.replace(/\\/g, "/").trim();
  const fileName = normalizedPath.split("/").filter(Boolean).pop();
  return fileName ? `/uploads/${encodeURIComponent(fileName)}` : null;
}

export function buildResearchFigureDownloadName(
  figure: ResearchPaperFigurePreview
): string {
  const normalizedPath =
    typeof figure.image_path === "string"
      ? figure.image_path.replace(/\\/g, "/").trim()
      : "";
  const fileName = normalizedPath.split("/").filter(Boolean).pop();
  if (fileName) return fileName;
  return `${figure.figure_id || figure.chart_id || "chart-anchor"}.png`;
}

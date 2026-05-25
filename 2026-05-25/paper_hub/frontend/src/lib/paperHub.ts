export type Lang = "zh" | "en";

export interface Paper {
  id?: number;
  uid: string;
  date: string;
  source: string;
  title: string;
  authors?: string;
  affiliations?: string;
  paper_url?: string;
  pdf_url?: string;
  code_url?: string;
  project_url?: string;
  tags?: string[];
  score?: number;
  score_reason_zh?: string;
  score_reason_en?: string;
  method_zh?: string;
  method_en?: string;
  innovation_zh?: string;
  innovation_en?: string;
  metrics_zh?: string;
  metrics_en?: string;
  thumbnail?: string;
}

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function fetchDates(): Promise<string[]> {
  const data = await apiGet<{ dates: string[] }>("/api/dates");
  return data.dates;
}

export async function fetchPapers(date: string): Promise<Paper[]> {
  const data = await apiGet<{ papers: Paper[] }>(
    `/api/papers?date=${encodeURIComponent(date)}`
  );
  return data.papers;
}

export function tPaper(paper: Paper, key: "method" | "innovation" | "metrics" | "score_reason", lang: Lang): string {
  const suffix = lang === "zh" ? "_zh" : "_en";
  const field = `${key}${suffix}` as keyof Paper;
  const value = paper[field];
  return typeof value === "string" ? value : "";
}

export function resolveThumbnail(paper: Paper): string {
  const url = paper.thumbnail ?? "";
  if (!url) return "";
  if (url.startsWith("http")) return url;
  return `${API_BASE}${url}`;
}

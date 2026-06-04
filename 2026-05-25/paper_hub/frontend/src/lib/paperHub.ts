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

const LOCAL_PREFIX = "paper_hub:v1:";

function hasApi(): boolean {
  return API_BASE.trim().length > 0;
}

async function apiGet<T>(path: string): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) {
    throw new Error(`Request failed: ${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

function normalizePapers(payload: unknown): Paper[] {
  const raw = Array.isArray(payload)
    ? (payload as Paper[])
    : payload && typeof payload === "object" && Array.isArray((payload as { papers?: unknown }).papers)
      ? ((payload as { papers: unknown }).papers as Paper[])
      : [];

  return raw.map((paper) => {
    const anyPaper = paper as unknown as { thumbnail_url?: string; thumbnail?: string };
    return {
      ...paper,
      thumbnail: paper.thumbnail ?? anyPaper.thumbnail_url ?? "",
    };
  });
}

function readCached(date: string): Paper[] | null {
  try {
    const raw = localStorage.getItem(`${LOCAL_PREFIX}${date}`);
    if (!raw) return null;
    const data = JSON.parse(raw) as unknown;
    return normalizePapers(data);
  } catch {
    return null;
  }
}

function writeCached(date: string, papers: Paper[]): void {
  try {
    localStorage.setItem(`${LOCAL_PREFIX}${date}`, JSON.stringify(papers));
  } catch {
    // ignore
  }
}

export async function fetchDates(): Promise<string[]> {
  if (hasApi()) {
    const data = await apiGet<{ dates: string[] }>("/api/dates");
    return data.dates;
  }

  const resp = await fetch("/data/index.json");
  if (!resp.ok) {
    throw new Error(`Failed to load local index: ${resp.status} ${resp.statusText}`);
  }
  const index = (await resp.json()) as { dates?: string[] };
  const dates = new Set<string>(Array.isArray(index.dates) ? index.dates : []);

  for (const key of Object.keys(localStorage)) {
    if (!key.startsWith(LOCAL_PREFIX)) continue;
    const date = key.slice(LOCAL_PREFIX.length);
    if (date) dates.add(date);
  }

  return Array.from(dates).sort((a, b) => b.localeCompare(a));
}

export async function fetchPapers(date: string): Promise<Paper[]> {
  if (hasApi()) {
    const data = await apiGet<{ papers: Paper[] }>(
      `/api/papers?date=${encodeURIComponent(date)}`
    );
    return data.papers;
  }

  const cached = readCached(date);
  if (cached) return cached;

  const resp = await fetch(`/data/seed/${encodeURIComponent(date)}.json`);
  if (!resp.ok) {
    throw new Error(`Failed to load local seed: ${resp.status} ${resp.statusText}`);
  }

  const payload = (await resp.json()) as unknown;
  const papers = normalizePapers(payload);
  writeCached(date, papers);
  return papers;
}

export function tPaper(
  paper: Paper,
  key: "method" | "innovation" | "metrics" | "score_reason",
  lang: Lang
): string {
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

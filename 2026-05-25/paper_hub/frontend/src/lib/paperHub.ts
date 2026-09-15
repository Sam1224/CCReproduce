export type Lang = "zh" | "en";

export interface ScoreBreakdown {
  innovation?: number;
  results?: number;
  exp_quality?: number;
  efficiency?: number;
  generalization?: number;
  relevance?: number;
  total?: number;
  rationale_zh?: string;
  rationale_en?: string;
}

export interface Paper {
  paper_id: string;
  inspection_date: string;
  published?: string;
  source?: string;
  title: string;
  authors?: string;
  affiliations?: string;
  paper_url?: string;
  pdf_url?: string;
  code_url?: string;
  demo_url?: string;
  project_url?: string;
  tags?: string[];
  score_total?: number;
  score_breakdown?: ScoreBreakdown;
  rationale_zh?: string;
  rationale_en?: string;
  method_overview_zh?: string;
  method_overview_en?: string;
  story_zh?: string;
  story_en?: string;
  innovation_zh?: string;
  innovation_en?: string;
  key_metrics_zh?: string;
  key_metrics_en?: string;
  figure_path?: string;
  exp_figure_path?: string;
}

interface AggregatedPayload {
  days?: Array<{ inspection_date: string; paper_count: number }>;
  papers?: unknown[];
}

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
const REPO_RAW_BASE =
  (import.meta.env.VITE_REPO_RAW_BASE as string | undefined) ??
  "https://raw.githubusercontent.com/Sam1224/CCReproduce/main";

let cachedPayload: AggregatedPayload | null = null;

function joinValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.filter((item) => typeof item === "string").join(", ");
  }
  return typeof value === "string" ? value : "";
}

function normalizePaperAsset(url: string | undefined): string {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  return `${REPO_RAW_BASE}/paper_webapp/${url.replace(/^\//, "")}`;
}

function normalizePaper(paper: unknown): Paper {
  const raw = paper as Record<string, unknown>;
  const links = (raw.links ?? {}) as Record<string, string | undefined>;
  const scoreBreakdown = (raw.score_breakdown ?? {}) as ScoreBreakdown;

  return {
    paper_id: String(raw.paper_id ?? raw.id ?? ""),
    inspection_date: String(raw.inspection_date ?? raw.date ?? ""),
    published: typeof raw.published === "string" ? raw.published : "",
    source: typeof raw.source === "string" ? raw.source : "",
    title: String(raw.title ?? ""),
    authors: joinValue(raw.authors),
    affiliations: joinValue(raw.affiliations),
    paper_url: links.abs ?? (typeof raw.paper_url === "string" ? raw.paper_url : ""),
    pdf_url: links.pdf ?? (typeof raw.pdf_url === "string" ? raw.pdf_url : ""),
    code_url: links.code ?? (typeof raw.code_url === "string" ? raw.code_url : ""),
    demo_url: links.demo ?? (typeof raw.demo_url === "string" ? raw.demo_url : ""),
    project_url: typeof raw.reproduce_url === "string" ? raw.reproduce_url : "",
    tags: Array.isArray(raw.tags) ? raw.tags.filter((item): item is string => typeof item === "string") : [],
    score_total: typeof raw.score_total === "number" ? raw.score_total : 0,
    score_breakdown: scoreBreakdown,
    rationale_zh: typeof raw.rationale_zh === "string" ? raw.rationale_zh : scoreBreakdown.rationale_zh ?? "",
    rationale_en: typeof raw.rationale_en === "string" ? raw.rationale_en : scoreBreakdown.rationale_en ?? "",
    method_overview_zh: typeof raw.method_overview_zh === "string" ? raw.method_overview_zh : "",
    method_overview_en: typeof raw.method_overview_en === "string" ? raw.method_overview_en : "",
    story_zh: typeof raw.story_zh === "string" ? raw.story_zh : "",
    story_en: typeof raw.story_en === "string" ? raw.story_en : "",
    innovation_zh: typeof raw.innovation_zh === "string" ? raw.innovation_zh : "",
    innovation_en: typeof raw.innovation_en === "string" ? raw.innovation_en : "",
    key_metrics_zh: typeof raw.key_metrics_zh === "string" ? raw.key_metrics_zh : "",
    key_metrics_en: typeof raw.key_metrics_en === "string" ? raw.key_metrics_en : "",
    figure_path: normalizePaperAsset(typeof raw.figure_path === "string" ? raw.figure_path : ""),
    exp_figure_path: normalizePaperAsset(typeof raw.exp_figure_path === "string" ? raw.exp_figure_path : ""),
  };
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const resp = await fetch(url);
    if (!resp.ok) return null;
    return (await resp.json()) as T;
  } catch {
    return null;
  }
}

async function fetchAggregatedPayload(): Promise<AggregatedPayload> {
  if (cachedPayload) return cachedPayload;

  const localUrl = `${API_BASE || ""}/data/papers.json`;
  const remoteUrl = `${REPO_RAW_BASE}/paper_webapp/data/papers.json`;
  const payload = (await fetchJson<AggregatedPayload>(localUrl)) ?? (await fetchJson<AggregatedPayload>(remoteUrl));

  if (!payload) {
    throw new Error("Failed to load aggregated paper data.");
  }

  cachedPayload = payload;
  return payload;
}

export async function fetchDates(): Promise<string[]> {
  const payload = await fetchAggregatedPayload();
  const dates = Array.isArray(payload.days)
    ? payload.days.map((item) => item.inspection_date)
    : [];
  return dates.sort((a, b) => b.localeCompare(a));
}

export async function fetchPapers(date: string): Promise<Paper[]> {
  const payload = await fetchAggregatedPayload();
  const papers = Array.isArray(payload.papers) ? payload.papers.map(normalizePaper) : [];
  return papers
    .filter((paper) => paper.inspection_date === date)
    .sort((left, right) => (right.score_total ?? 0) - (left.score_total ?? 0));
}

export function tPaper(
  paper: Paper,
  key: "method_overview" | "story" | "innovation" | "key_metrics" | "rationale",
  lang: Lang
): string {
  const suffix = lang === "zh" ? "_zh" : "_en";
  const field = `${key}${suffix}` as keyof Paper;
  const value = paper[field];
  return typeof value === "string" ? value : "";
}

export function resolveAsset(url?: string): string {
  return url ?? "";
}

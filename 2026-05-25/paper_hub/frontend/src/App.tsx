import { useEffect, useMemo, useState } from "react";

import "./App.css";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

import type { Lang, Paper } from "@/lib/paperHub";
import { fetchDates, fetchPapers } from "@/lib/paperHub";

import { PaperCard } from "@/components/PaperCard";

function t(zh: string, en: string, lang: Lang) {
  return lang === "zh" ? zh : en;
}

function App() {
  const [lang, setLang] = useState<Lang>("zh");
  const [dates, setDates] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState<string>("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [minScore, setMinScore] = useState<number>(40);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const ds = await fetchDates();
        if (cancelled) return;
        setDates(ds);
        setSelectedDate(ds[0] ?? "");
      } catch (event) {
        if (!cancelled) setError((event as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedDate) return;
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        setError("");
        const nextPapers = await fetchPapers(selectedDate);
        if (!cancelled) setPapers(nextPapers);
      } catch (event) {
        if (!cancelled) setError((event as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedDate]);

  const filtered = useMemo(
    () => papers.filter((paper) => (paper.score_total ?? 0) >= minScore),
    [papers, minScore]
  );

  const topPaper = filtered[0];

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(186,230,253,0.45),_transparent_32%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_35%,#f8fafc_100%)] text-zinc-950">
      <div className="mx-auto max-w-7xl px-4 py-10">
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.9fr)]">
          <Card className="border-zinc-200/80 bg-white/85 p-6 shadow-sm backdrop-blur">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="secondary" className="rounded-full bg-sky-100 text-sky-700">
                {t("电商内容生态 & 达人治理", "E-commerce & Creator Governance", lang)}
              </Badge>
              <Badge variant="outline" className="rounded-full border-zinc-200 text-zinc-600">
                {t("每日巡检 WebApp", "Daily paper web app", lang)}
              </Badge>
            </div>
            <h1 className="mt-4 text-3xl font-semibold tracking-tight text-zinc-950">
              {t("Daily Paper Radar", "Daily Paper Radar", lang)}
            </h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-600">
              {t(
                "聚合每日论文巡检结果，支持按日期回看、按分数过滤、中英切换，并展示 methodology figure、实验亮点、评分明细与代码链接。",
                "A bilingual paper radar for daily scans with historical dates, score filtering, methodology figures, experiment highlights, scoring breakdowns, and code links.",
                lang
              )}
            </p>
            <div className="mt-5 flex flex-wrap gap-3">
              <Button asChild className="rounded-full">
                <a href="https://github.com/Sam1224/CCReproduce" target="_blank" rel="noreferrer">
                  {t("打开 GitHub 仓库", "Open GitHub repo", lang)}
                </a>
              </Button>
              {topPaper?.project_url ? (
                <Button asChild variant="outline" className="rounded-full">
                  <a href={topPaper.project_url} target="_blank" rel="noreferrer">
                    {t("查看当前最高分复现", "Open top reproduction", lang)}
                  </a>
                </Button>
              ) : null}
            </div>
          </Card>

          <Card className="border-zinc-200/80 bg-white/85 p-5 shadow-sm backdrop-blur">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl bg-zinc-50 p-4">
                <div className="text-xs text-zinc-500">{t("已归档日期", "Archived dates", lang)}</div>
                <div className="mt-1 text-2xl font-semibold text-zinc-950">{dates.length}</div>
              </div>
              <div className="rounded-2xl bg-zinc-50 p-4">
                <div className="text-xs text-zinc-500">{t("当日论文数", "Papers today", lang)}</div>
                <div className="mt-1 text-2xl font-semibold text-zinc-950">{papers.length}</div>
              </div>
              <div className="rounded-2xl bg-zinc-50 p-4">
                <div className="text-xs text-zinc-500">{t("过滤后", "Filtered", lang)}</div>
                <div className="mt-1 text-2xl font-semibold text-zinc-950">{filtered.length}</div>
              </div>
              <div className="rounded-2xl bg-zinc-50 p-4">
                <div className="text-xs text-zinc-500">{t("当前阈值", "Min score", lang)}</div>
                <div className="mt-1 text-2xl font-semibold text-zinc-950">{minScore}</div>
              </div>
            </div>
          </Card>
        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
          <Card className="border-zinc-200/80 bg-white/85 p-4 shadow-sm backdrop-blur">
            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <div className="mb-1 text-xs text-zinc-500">{t("语言", "Language", lang)}</div>
                <div className="flex gap-1 rounded-full border border-zinc-200 bg-white p-1">
                  <Button
                    variant={lang === "zh" ? "secondary" : "ghost"}
                    size="sm"
                    className="h-8 rounded-full px-4 text-xs"
                    onClick={() => setLang("zh")}
                  >
                    中文
                  </Button>
                  <Button
                    variant={lang === "en" ? "secondary" : "ghost"}
                    size="sm"
                    className="h-8 rounded-full px-4 text-xs"
                    onClick={() => setLang("en")}
                  >
                    EN
                  </Button>
                </div>
              </div>

              <div>
                <div className="mb-1 text-xs text-zinc-500">{t("日期", "Date", lang)}</div>
                <Select value={selectedDate} onValueChange={setSelectedDate}>
                  <SelectTrigger className="h-10 rounded-full bg-white">
                    <SelectValue placeholder={t("选择日期", "Select date", lang)} />
                  </SelectTrigger>
                  <SelectContent>
                    {dates.map((date) => (
                      <SelectItem key={date} value={date}>
                        {date}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <div className="mb-1 flex items-center justify-between text-xs text-zinc-500">
                  <span>{t("最低分", "Min score", lang)}</span>
                  <span className="font-medium text-zinc-800">{minScore}</span>
                </div>
                <div className="rounded-2xl border border-zinc-200 bg-white px-4 py-4">
                  <Slider value={[minScore]} min={0} max={100} step={1} onValueChange={(value) => setMinScore(value[0] ?? 0)} />
                </div>
              </div>
            </div>
          </Card>

          <Card className="border-zinc-200/80 bg-white/85 p-4 text-sm text-zinc-700 shadow-sm backdrop-blur">
            <div className="space-y-2">
              <div>
                {t("当前日期：", "Selected date: ", lang)}
                <span className="font-medium text-zinc-900">{selectedDate || "-"}</span>
              </div>
              <div>
                {t("最高分论文：", "Top paper: ", lang)}
                <span className="font-medium text-zinc-900">{topPaper?.title ?? "-"}</span>
              </div>
              {loading ? <Badge>{t("加载中", "Loading", lang)}</Badge> : null}
              {error ? (
                <Badge variant="destructive" className="max-w-full truncate">
                  {error}
                </Badge>
              ) : null}
            </div>
          </Card>
        </div>

        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {filtered.map((paper) => (
            <PaperCard key={`${paper.inspection_date}-${paper.paper_id}`} paper={paper} lang={lang} />
          ))}
        </div>

        {!loading && selectedDate && filtered.length === 0 ? (
          <div className="mt-10 text-center text-sm text-zinc-600">
            {t("没有满足当前分数阈值的论文。", "No papers meet the current score threshold.", lang)}
          </div>
        ) : null}
      </div>

      <footer className="border-t border-zinc-200/70 bg-white/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-xs text-zinc-500 sm:flex-row sm:items-center sm:justify-between">
          <div>
            {t(
              "数据源：arXiv / Hugging Face Daily Papers / OpenReview / GitHub 代码核验；结果按每日巡检归档。",
              "Sources: arXiv / Hugging Face Daily Papers / OpenReview / GitHub code verification; results are archived by daily inspection date.",
              lang
            )}
          </div>
          <div className="text-zinc-400">Paper Radar • 2026</div>
        </div>
      </footer>
    </div>
  );
}

export default App;

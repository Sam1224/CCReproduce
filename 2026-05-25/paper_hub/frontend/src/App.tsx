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
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        setLoading(true);
        const ds = await fetchDates();
        if (cancelled) return;
        setDates(ds);
        const first = ds[0] ?? "";
        setSelectedDate(first);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
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
        const ps = await fetchPapers(selectedDate);
        if (!cancelled) setPapers(ps);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedDate]);

  const filtered = useMemo(
    () => papers.filter((p) => (p.score ?? 0) >= minScore),
    [papers, minScore]
  );

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-950">
      <div className="mx-auto max-w-7xl px-4 py-10">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="text-xl font-semibold tracking-tight">Paper Hub</div>
              <Badge variant="secondary" className="text-xs">
                {t("电商内容生态&达人治理", "E-commerce & Creator Governance", lang)}
              </Badge>
            </div>
            <div className="text-sm text-zinc-600">
              {t(
                "每日巡检结果（支持按日期回看、按分数筛选、中英切换）",
                "Daily paper digest (date history, score filter, bilingual)",
                lang
              )}
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex items-center gap-2">
              <div className="text-xs text-zinc-500">{t("语言", "Lang", lang)}</div>
              <div className="flex gap-1 rounded-lg border border-zinc-200 bg-white p-1">
                <Button
                  variant={lang === "zh" ? "secondary" : "ghost"}
                  size="sm"
                  className="h-8 px-3 text-xs"
                  onClick={() => setLang("zh")}
                >
                  中文
                </Button>
                <Button
                  variant={lang === "en" ? "secondary" : "ghost"}
                  size="sm"
                  className="h-8 px-3 text-xs"
                  onClick={() => setLang("en")}
                >
                  EN
                </Button>
              </div>
            </div>

            <div className="w-full sm:w-56">
              <div className="mb-1 text-xs text-zinc-500">{t("日期", "Date", lang)}</div>
              <Select value={selectedDate} onValueChange={setSelectedDate}>
                <SelectTrigger className="h-9 bg-white">
                  <SelectValue placeholder={t("选择日期", "Select date", lang)} />
                </SelectTrigger>
                <SelectContent>
                  {dates.map((d) => (
                    <SelectItem key={d} value={d}>
                      {d}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="w-full sm:w-56">
              <div className="mb-1 flex items-center justify-between text-xs text-zinc-500">
                <div>{t("最低分", "Min score", lang)}</div>
                <div className="font-medium text-zinc-800">{minScore}</div>
              </div>
              <div className="rounded-lg border border-zinc-200 bg-white px-3 py-3">
                <Slider
                  value={[minScore]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(v) => setMinScore(v[0] ?? 0)}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="mt-6">
          <Card className="border-zinc-200 bg-white p-4 text-sm text-zinc-700">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <div>
                {t("当前日期：", "Selected date: ", lang)}
                <span className="font-medium text-zinc-900">{selectedDate || "-"}</span>
              </div>
              <div>
                {t("论文数：", "Papers: ", lang)}
                <span className="font-medium text-zinc-900">{papers.length}</span>
              </div>
              <div>
                {t("筛选后：", "Filtered: ", lang)}
                <span className="font-medium text-zinc-900">{filtered.length}</span>
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

        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((paper) => (
            <PaperCard key={`${paper.date}-${paper.uid}`} paper={paper} lang={lang} />
          ))}
        </div>

        {!loading && selectedDate && filtered.length === 0 ? (
          <div className="mt-10 text-center text-sm text-zinc-600">
            {t("没有满足分数阈值的论文。", "No papers meet the score threshold.", lang)}
          </div>
        ) : null}
      </div>

      <footer className="border-t border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-xs text-zinc-500 sm:flex-row sm:items-center sm:justify-between">
          <div>
            {t(
              "数据来源：ArXiv / HuggingFace Daily Papers / OpenReview 等（以巡检结果为准）",
              "Sources: arXiv / HuggingFace Daily Papers / OpenReview (per daily scan)",
              lang
            )}
          </div>
          <div className="text-zinc-400">Paper Hub • 2026</div>
        </div>
      </footer>
    </div>
  );
}

export default App;

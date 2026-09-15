import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

import type { Lang, Paper, ScoreBreakdown } from "@/lib/paperHub";
import { resolveAsset, tPaper } from "@/lib/paperHub";

function sectionTitle(zh: string, en: string, lang: Lang): string {
  return lang === "zh" ? zh : en;
}

function scoreItems(lang: Lang, score?: ScoreBreakdown) {
  return [
    { label: sectionTitle("创新", "Innovation", lang), value: score?.innovation ?? 0, total: 30 },
    { label: sectionTitle("指标", "Results", lang), value: score?.results ?? 0, total: 15 },
    { label: sectionTitle("实验", "Exp", lang), value: score?.exp_quality ?? 0, total: 15 },
    { label: sectionTitle("效率", "Eff.", lang), value: score?.efficiency ?? 0, total: 10 },
    { label: sectionTitle("泛化", "Gen.", lang), value: score?.generalization ?? 0, total: 5 },
    { label: sectionTitle("相关性", "Relevance", lang), value: score?.relevance ?? 0, total: 25 },
  ];
}

export function PaperCard({ paper, lang, className }: { paper: Paper; lang: Lang; className?: string }) {
  const [showFigure, setShowFigure] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);

  const method = tPaper(paper, "method_overview", lang);
  const story = tPaper(paper, "story", lang);
  const innovation = tPaper(paper, "innovation", lang);
  const metrics = tPaper(paper, "key_metrics", lang);
  const rationale = tPaper(paper, "rationale", lang);

  const figurePath = useMemo(() => resolveAsset(paper.figure_path), [paper.figure_path]);
  const expFigurePath = useMemo(() => resolveAsset(paper.exp_figure_path), [paper.exp_figure_path]);
  const breakdown = scoreItems(lang, paper.score_breakdown);
  const totalScore = paper.score_total ?? 0;

  return (
    <Card className={cn("h-full overflow-hidden border-zinc-200 bg-white/90 shadow-sm backdrop-blur", className)}>
      <div className="h-1.5 w-full bg-gradient-to-r from-sky-400 via-indigo-400 to-violet-400" />
      <CardHeader className="space-y-3 pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2 text-[11px] text-zinc-500">
              <span>{paper.inspection_date}</span>
              {paper.published ? <span>• {paper.published}</span> : null}
              {paper.source ? <span>• {paper.source}</span> : null}
            </div>
            <CardTitle className="text-sm leading-snug text-zinc-950">
              <a href={paper.paper_url || "#"} target="_blank" rel="noreferrer" className="hover:underline">
                {paper.title}
              </a>
            </CardTitle>
          </div>
          <div className="shrink-0 rounded-xl bg-zinc-950 px-3 py-2 text-right text-white shadow-sm">
            <div className="text-[10px] uppercase tracking-[0.16em] text-zinc-300">Score</div>
            <div className="text-lg font-semibold leading-none">{totalScore}</div>
          </div>
        </div>

        <div className="space-y-1 text-xs text-zinc-600">
          {paper.authors ? <div className="line-clamp-2">{paper.authors}</div> : null}
          {paper.affiliations ? <div className="line-clamp-2 text-zinc-500">{paper.affiliations}</div> : null}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {(paper.tags ?? []).slice(0, 6).map((tag) => (
            <Badge key={tag} variant="secondary" className="rounded-full bg-zinc-100 text-[11px] text-zinc-700">
              {tag}
            </Badge>
          ))}
        </div>
      </CardHeader>

      <CardContent className="space-y-4 text-xs text-zinc-800">
        <div className="grid grid-cols-3 gap-2 rounded-xl border border-zinc-200 bg-zinc-50 p-3">
          {breakdown.map((item) => (
            <div key={item.label} className="rounded-lg bg-white px-2 py-2">
              <div className="text-[10px] text-zinc-500">{item.label}</div>
              <div className="mt-1 font-semibold text-zinc-900">
                {item.value}/{item.total}
              </div>
            </div>
          ))}
        </div>

        {method ? (
          <div>
            <div className="font-semibold text-zinc-900">{sectionTitle("方法概览", "Method overview", lang)}</div>
            <div className="mt-1 line-clamp-6 whitespace-pre-wrap text-zinc-700">{method}</div>
          </div>
        ) : null}

        {innovation ? (
          <div>
            <div className="font-semibold text-zinc-900">{sectionTitle("创新点", "Innovation", lang)}</div>
            <div className="mt-1 line-clamp-5 whitespace-pre-wrap text-zinc-700">{innovation}</div>
          </div>
        ) : null}

        {story ? (
          <div>
            <div className="font-semibold text-zinc-900">{sectionTitle("论文故事", "Paper story", lang)}</div>
            <div className="mt-1 line-clamp-5 whitespace-pre-wrap text-zinc-700">{story}</div>
          </div>
        ) : null}

        {rationale ? (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-zinc-700">
            <div className="font-semibold text-zinc-900">{sectionTitle("评分依据", "Scoring rationale", lang)}</div>
            <div className="mt-1 line-clamp-4 whitespace-pre-wrap">{rationale}</div>
          </div>
        ) : null}

        <Separator />

        <div className="flex flex-wrap items-center gap-2">
          <Collapsible open={showFigure} onOpenChange={setShowFigure}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 rounded-full text-xs">
                {showFigure
                  ? sectionTitle("隐藏 Methodology", "Hide methodology", lang)
                  : sectionTitle("显示 Methodology", "Show methodology", lang)}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              {figurePath ? (
                <img src={figurePath} alt={`${paper.title} methodology`} className="w-full rounded-xl border border-zinc-200" loading="lazy" />
              ) : null}
            </CollapsibleContent>
          </Collapsible>

          <Collapsible open={showExperiments} onOpenChange={setShowExperiments}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 rounded-full text-xs">
                {showExperiments
                  ? sectionTitle("隐藏实验", "Hide experiments", lang)
                  : sectionTitle("显示实验", "Show experiments", lang)}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-3">
              {expFigurePath ? (
                <img src={expFigurePath} alt={`${paper.title} experiments`} className="w-full rounded-xl border border-zinc-200" loading="lazy" />
              ) : null}
              <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-zinc-700">
                <div className="font-semibold text-zinc-900">{sectionTitle("关键指标", "Key metrics", lang)}</div>
                <div className="mt-1 whitespace-pre-wrap">{metrics || "-"}</div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </CardContent>

      <CardFooter className="flex flex-wrap gap-2">
        {paper.paper_url ? (
          <Button asChild size="sm" className="h-8 rounded-full text-xs">
            <a href={paper.paper_url} target="_blank" rel="noreferrer">
              {sectionTitle("论文", "Paper", lang)}
            </a>
          </Button>
        ) : null}
        {paper.pdf_url ? (
          <Button asChild variant="secondary" size="sm" className="h-8 rounded-full text-xs">
            <a href={paper.pdf_url} target="_blank" rel="noreferrer">
              PDF
            </a>
          </Button>
        ) : null}
        {paper.code_url ? (
          <Button asChild variant="outline" size="sm" className="h-8 rounded-full text-xs">
            <a href={paper.code_url} target="_blank" rel="noreferrer">
              {sectionTitle("官方代码", "Official code", lang)}
            </a>
          </Button>
        ) : null}
        {paper.demo_url ? (
          <Button asChild variant="outline" size="sm" className="h-8 rounded-full text-xs">
            <a href={paper.demo_url} target="_blank" rel="noreferrer">
              Demo
            </a>
          </Button>
        ) : null}
        {paper.project_url ? (
          <Button asChild variant="outline" size="sm" className="h-8 rounded-full text-xs">
            <a href={paper.project_url} target="_blank" rel="noreferrer">
              {sectionTitle("复现代码", "Reproduction", lang)}
            </a>
          </Button>
        ) : null}
      </CardFooter>
    </Card>
  );
}

import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

import type { Lang, Paper } from "@/lib/paperHub";
import { resolveThumbnail, tPaper } from "@/lib/paperHub";

function sectionTitle(zh: string, en: string, lang: Lang): string {
  return lang === "zh" ? zh : en;
}

function MiniFigure({ title }: { title: string }) {
  return (
    <div className="w-full overflow-hidden rounded-lg border border-zinc-200 bg-white">
      <svg viewBox="0 0 800 260" className="h-auto w-full">
        <defs>
          <linearGradient id="phbg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#fafafa" />
            <stop offset="100%" stopColor="#f4f4f5" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width="800" height="260" fill="url(#phbg)" />

        <rect x="40" y="70" width="180" height="120" rx="16" fill="#ffffff" stroke="#e4e4e7" />
        <text x="130" y="120" textAnchor="middle" fontSize="16" fill="#18181b">
          Input
        </text>
        <text x="130" y="150" textAnchor="middle" fontSize="12" fill="#52525b">
          (text / image)
        </text>

        <rect x="310" y="55" width="180" height="150" rx="16" fill="#ffffff" stroke="#e4e4e7" />
        <text x="400" y="110" textAnchor="middle" fontSize="16" fill="#18181b">
          Method
        </text>
        <text x="400" y="140" textAnchor="middle" fontSize="12" fill="#52525b">
          (paper pipeline)
        </text>
        <text x="400" y="170" textAnchor="middle" fontSize="12" fill="#52525b">
          {title.slice(0, 22)}
        </text>

        <rect x="580" y="70" width="180" height="120" rx="16" fill="#ffffff" stroke="#e4e4e7" />
        <text x="670" y="120" textAnchor="middle" fontSize="16" fill="#18181b">
          Output
        </text>
        <text x="670" y="150" textAnchor="middle" fontSize="12" fill="#52525b">
          metrics / results
        </text>

        <path d="M220 130 L290 130" stroke="#a1a1aa" strokeWidth="3" />
        <path d="M490 130 L560 130" stroke="#a1a1aa" strokeWidth="3" />
        <polygon points="290,130 280,124 280,136" fill="#a1a1aa" />
        <polygon points="560,130 550,124 550,136" fill="#a1a1aa" />
      </svg>
    </div>
  );
}

export function PaperCard({ paper, lang, className }: { paper: Paper; lang: Lang; className?: string }) {
  const [showFigure, setShowFigure] = useState(false);
  const [showExperiments, setShowExperiments] = useState(false);

  const thumb = useMemo(() => resolveThumbnail(paper), [paper]);

  const method = tPaper(paper, "method", lang);
  const innovation = tPaper(paper, "innovation", lang);
  const metrics = tPaper(paper, "metrics", lang);
  const scoreReason = tPaper(paper, "score_reason", lang);

  const score = paper.score ?? 0;

  return (
    <Card className={cn("h-full", className)}>
      <CardHeader className="space-y-2">
        <div className="flex items-start justify-between gap-3">
          <CardTitle className="text-sm leading-snug">
            <a
              href={paper.paper_url || "#"}
              target="_blank"
              rel="noreferrer"
              className="hover:underline"
              title={paper.title}
            >
              {paper.title}
            </a>
          </CardTitle>
          <div className="shrink-0 text-right">
            <div className="text-xs text-zinc-500">{paper.date}</div>
            <div className="mt-0.5 rounded-md bg-zinc-900 px-2 py-1 text-xs font-semibold text-white">
              {score}
            </div>
          </div>
        </div>

        <div className="text-xs text-zinc-600">
          <div className="line-clamp-2">{paper.authors}</div>
          <div className="mt-1 line-clamp-2 text-zinc-500">{paper.affiliations}</div>
        </div>

        <div className="flex flex-wrap gap-1">
          {(paper.tags ?? []).slice(0, 6).map((tag) => (
            <Badge key={tag} variant="secondary" className="text-[11px]">
              {tag}
            </Badge>
          ))}
          {paper.source ? (
            <Badge variant="outline" className="text-[11px]">
              {paper.source}
            </Badge>
          ) : null}
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <div className="space-y-2 text-xs text-zinc-800">
          {method ? (
            <div>
              <div className="font-semibold text-zinc-900">
                {sectionTitle("方法概述", "Method", lang)}
              </div>
              <div className="mt-1 line-clamp-6 whitespace-pre-wrap text-zinc-700">{method}</div>
            </div>
          ) : null}

          {innovation ? (
            <div>
              <div className="font-semibold text-zinc-900">
                {sectionTitle("创新点", "Innovation", lang)}
              </div>
              <div className="mt-1 line-clamp-6 whitespace-pre-wrap text-zinc-700">{innovation}</div>
            </div>
          ) : null}

          {scoreReason ? (
            <div>
              <div className="font-semibold text-zinc-900">
                {sectionTitle("评分依据", "Score rationale", lang)}
              </div>
              <div className="mt-1 line-clamp-4 whitespace-pre-wrap text-zinc-700">{scoreReason}</div>
            </div>
          ) : null}
        </div>

        <Separator />

        <div className="flex flex-wrap items-center gap-2">
          <Collapsible open={showFigure} onOpenChange={setShowFigure}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 text-xs">
                {showFigure
                  ? sectionTitle("隐藏 Figure", "Hide figure", lang)
                  : sectionTitle("显示 Figure", "Show figure", lang)}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              {thumb ? (
                <a href={paper.paper_url || "#"} target="_blank" rel="noreferrer" title="Open paper">
                  <img
                    src={thumb}
                    alt={paper.title}
                    className="w-full rounded-lg border border-zinc-200"
                    loading="lazy"
                  />
                </a>
              ) : (
                <MiniFigure title={paper.uid || paper.title} />
              )}
            </CollapsibleContent>
          </Collapsible>

          <Collapsible open={showExperiments} onOpenChange={setShowExperiments}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 text-xs">
                {showExperiments
                  ? sectionTitle("隐藏 实验", "Hide experiments", lang)
                  : sectionTitle("显示 实验", "Show experiments", lang)}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2">
              <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700">
                <div className="font-semibold text-zinc-900">
                  {sectionTitle("关键指标", "Key results", lang)}
                </div>
                <div className="mt-1 whitespace-pre-wrap">{metrics || "-"}</div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </CardContent>

      <CardFooter className="flex flex-wrap gap-2">
        {paper.paper_url ? (
          <Button asChild variant="secondary" size="sm" className="h-8 text-xs">
            <a href={paper.paper_url} target="_blank" rel="noreferrer">
              {sectionTitle("论文", "Paper", lang)}
            </a>
          </Button>
        ) : null}
        {paper.pdf_url ? (
          <Button asChild variant="secondary" size="sm" className="h-8 text-xs">
            <a href={paper.pdf_url} target="_blank" rel="noreferrer">
              PDF
            </a>
          </Button>
        ) : null}
        {paper.code_url ? (
          <Button asChild variant="outline" size="sm" className="h-8 text-xs">
            <a href={paper.code_url} target="_blank" rel="noreferrer">
              {sectionTitle("官方代码", "Code", lang)}
            </a>
          </Button>
        ) : null}
        {paper.project_url ? (
          <Button asChild variant="outline" size="sm" className="h-8 text-xs">
            <a href={paper.project_url} target="_blank" rel="noreferrer">
              {sectionTitle("复现", "Reproduce", lang)}
            </a>
          </Button>
        ) : null}
      </CardFooter>
    </Card>
  );
}

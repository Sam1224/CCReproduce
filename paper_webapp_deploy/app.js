let db = null;
let jsonData = null;
let lang = "zh";

const I18N = {
  zh: {
    date: "日期",
    minScore: "最低分",
    method: "方法概览",
    story: "故事线",
    innovation: "创新点",
    breakdown: "评分明细",
    rationale: "评分依据",
    toggleFigure: "显示/隐藏 Method Figure",
    toggleExp: "显示/隐藏 亮眼实验",
    inspection: "巡检日期",
    published: "论文日期",
    source: "来源",
    authors: "作者",
    affiliations: "机构",
    dayKicker: "每日论文巡检",
    paperCount: "论文数",
    filteredCount: "当前显示",
    window: "巡检窗口",
    linkPaper: "论文",
    linkPdf: "PDF",
    linkCode: "开源代码",
    linkProject: "项目页",
    linkDataset: "数据集",
    linkReproduce: "复现代码",
    loading: "Loading…",
    empty: "当前筛选条件下没有论文。",
  },
  en: {
    date: "Date",
    minScore: "Min score",
    method: "Method overview",
    story: "Problem → solution",
    innovation: "Key innovations",
    breakdown: "Score breakdown",
    rationale: "Score rationale",
    toggleFigure: "Show/Hide methodology figure",
    toggleExp: "Show/Hide highlights",
    inspection: "Inspection",
    published: "Published",
    source: "Source",
    authors: "Authors",
    affiliations: "Affiliations",
    dayKicker: "Daily paper inspection",
    paperCount: "Papers",
    filteredCount: "Showing",
    window: "Inspection window",
    linkPaper: "Paper",
    linkPdf: "PDF",
    linkCode: "Code",
    linkProject: "Project",
    linkDataset: "Dataset",
    linkReproduce: "Reproduce",
    loading: "Loading…",
    empty: "No papers match the current filter.",
  },
};

// Score dimensions (key, max, labels) used to render the per-paper breakdown.
// `alt` lists alternate JSON keys so both the legacy (results/exp_quality) and the
// newer (metrics/quality) papers.json score schemas render correctly.
const DIMS = [
  { key: "innovation", alt: [], max: 30, zh: "创新性", en: "Innovation" },
  { key: "results", alt: ["metrics"], max: 15, zh: "实验指标", en: "Results" },
  { key: "exp_quality", alt: ["quality"], max: 15, zh: "实验质量", en: "Exp. quality" },
  { key: "efficiency", alt: [], max: 10, zh: "效率", en: "Efficiency" },
  { key: "generalization", alt: [], max: 5, zh: "泛化", en: "Generalization" },
  { key: "relevance", alt: [], max: 25, zh: "相关性", en: "Relevance" },
];

function dimValue(breakdown, d) {
  if (typeof breakdown[d.key] === "number") return breakdown[d.key];
  for (const k of d.alt || []) {
    if (typeof breakdown[k] === "number") return breakdown[k];
  }
  return undefined;
}

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("status").textContent = text;
}

function setLang(nextLang) {
  lang = nextLang;
  const t = I18N[lang];
  $("label-date").textContent = t.date;
  $("label-score").textContent = t.minScore;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    el.textContent = t[key] || el.textContent;
  });
  $("langToggle").textContent = lang === "zh" ? "EN" : "中文";
}

function query(sql, params = []) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

function parseJsonMaybe(v) {
  if (v === null || v === undefined) return null;
  if (typeof v !== "string") return v;
  try {
    return JSON.parse(v);
  } catch {
    return v;
  }
}

function getJsonDays() {
  return Array.isArray(jsonData?.days) ? jsonData.days : [];
}

function getJsonPapers() {
  return Array.isArray(jsonData?.papers) ? jsonData.papers : [];
}

function listDates() {
  if (jsonData) {
    const dayDates = getJsonDays()
      .map((day) => day?.inspection_date)
      .filter(Boolean);
    const paperDates = getJsonPapers()
      .map((paper) => paper?.inspection_date)
      .filter(Boolean);
    return [...new Set([...dayDates, ...paperDates])].sort((a, b) => b.localeCompare(a));
  }

  try {
    return query("SELECT inspection_date AS d FROM days ORDER BY inspection_date DESC").map((r) => r.d);
  } catch {
    return query("SELECT DISTINCT inspection_date AS d FROM papers ORDER BY inspection_date DESC").map((r) => r.d);
  }
}

function getDayMeta(date) {
  if (jsonData) {
    return (
      getJsonDays().find((day) => day && day.inspection_date === date) ||
      {
        inspection_date: date,
        paper_count: getJsonPapers().filter((paper) => paper && paper.inspection_date === date).length,
      }
    );
  }

  try {
    const rows = query(
      `SELECT inspection_date, paper_count, notes_zh, notes_en, window_start, window_end, repo_branch
       FROM days WHERE inspection_date = ?`,
      [date]
    );
    if (rows.length) return rows[0];
  } catch {
    // Older generated databases only had the papers table; keep the static app usable.
  }
  return { inspection_date: date, paper_count: null };
}

function getPapersForDate(date, minScore) {
  if (jsonData) {
    return getJsonPapers()
      .filter((paper) => paper && paper.inspection_date === date && Number(paper.score_total || 0) >= minScore)
      .sort((a, b) => Number(b.score_total || 0) - Number(a.score_total || 0) || String(a.title || "").localeCompare(String(b.title || "")));
  }

  return query(
    `SELECT 
      inspection_date, paper_id, title, authors, affiliations, source, published, links, tags,
      score_total, score_breakdown, rationale_zh, rationale_en,
      method_overview_zh, method_overview_en,
      story_zh, story_en,
      innovation_zh, innovation_en,
      key_metrics_zh, key_metrics_en, reproduce_url, figure_path, exp_figure_path
    FROM papers
    WHERE inspection_date = ? AND score_total >= ?
    ORDER BY score_total DESC, title ASC`,
    [date, minScore]
  );
}

function formatWindow(day) {
  if (!day || (!day.window_start && !day.window_end)) return "";
  const start = day.window_start || "?";
  const end = day.window_end || "?";
  return `${I18N[lang].window}: ${start} → ${end}`;
}

function renderDayHeader(day, visibleCount) {
  const header = $("dayHeader");
  if (!header || !day || !day.inspection_date) return;
  header.classList.remove("hidden");
  $("dayKicker").textContent = I18N[lang].dayKicker;
  $("dayTitle").textContent = day.inspection_date;

  const total = Number.isFinite(Number(day.paper_count)) ? Number(day.paper_count) : visibleCount;
  $("dayCount").textContent = `${I18N[lang].filteredCount} ${visibleCount} / ${I18N[lang].paperCount} ${total}`;
  $("dayWindow").textContent = formatWindow(day);
  $("dayWindow").style.display = $("dayWindow").textContent ? "" : "none";

  const note = (lang === "zh" ? day.notes_zh : day.notes_en) || day.notes_zh || day.notes_en || "";
  $("dayNote").textContent = note;
  $("dayNote").style.display = note ? "" : "none";
}

function addLink(items, seen, name, url) {
  if (!url || seen.has(url)) return;
  seen.add(url);
  items.push({ name, url });
}

function renderPapers(papers) {
  const grid = $("grid");
  grid.innerHTML = "";

  if (!papers.length) {
    setStatus(I18N[lang].empty);
    return;
  }

  setStatus("");

  const tpl = $("paperCardTpl");
  for (const p of papers) {
    const node = tpl.content.cloneNode(true);
    const card = node.querySelector(".card");

    card.querySelector(".title").textContent = p.title;

    const authors = parseJsonMaybe(p.authors) || [];
    const affiliations = parseJsonMaybe(p.affiliations) || [];
    const t = I18N[lang];
    const metaParts = [
      `${p.inspection_date ? `${t.inspection}: ${p.inspection_date}` : ""}`,
      `${p.published ? `${t.published}: ${p.published}` : ""}`,
      `${p.source ? `${t.source}: ${p.source}` : ""}`,
      `${authors.length ? `${t.authors}: ${authors.join(", ")}` : ""}`,
      `${affiliations.length ? `${t.affiliations}: ${affiliations.join(" / ")}` : ""}`,
    ].filter(Boolean);
    card.querySelector(".meta").textContent = metaParts.join(" · ");

    card.querySelector(".score-num").textContent = `${p.score_total}`;

    const tags = parseJsonMaybe(p.tags) || [];
    const tagWrap = card.querySelector(".tags");
    for (const tag of tags) {
      const el = document.createElement("span");
      el.className = "tag";
      el.textContent = tag;
      tagWrap.appendChild(el);
    }

    const links = parseJsonMaybe(p.links) || {};
    const linksWrap = card.querySelector(".links");
    const linkItems = [];
    const seenLinks = new Set();
    const paperUrl = links.abs || links.arxiv || links.paper;
    addLink(linkItems, seenLinks, paperUrl && paperUrl.includes("arxiv.org") ? "arXiv" : t.linkPaper, paperUrl);
    addLink(linkItems, seenLinks, t.linkPdf, links.pdf);
    addLink(linkItems, seenLinks, t.linkCode, links.code || links.github || links.repo);
    addLink(linkItems, seenLinks, t.linkProject, links.project || links.project_page);
    addLink(linkItems, seenLinks, t.linkDataset, links.dataset || links.data);
    addLink(linkItems, seenLinks, t.linkReproduce, p.reproduce_url);

    for (const it of linkItems) {
      const a = document.createElement("a");
      a.className = "link";
      a.href = it.url;
      a.target = "_blank";
      a.rel = "noreferrer";
      a.textContent = it.name;
      linksWrap.appendChild(a);
    }

    const methodText = lang === "zh" ? p.method_overview_zh : p.method_overview_en;
    const storyText = lang === "zh" ? p.story_zh : p.story_en;
    const innovationText = lang === "zh" ? p.innovation_zh : p.innovation_en;

    card.querySelector(".method").textContent = methodText || "";
    card.querySelector(".story").textContent = storyText || "";
    card.querySelector(".innovation").textContent = innovationText || "";
    card.querySelector(".rationale").textContent = (lang === "zh" ? p.rationale_zh : p.rationale_en) || p.rationale_zh || "";

    // score breakdown (per-dimension bars)
    const breakdown = parseJsonMaybe(p.score_breakdown) || {};
    const bWrap = card.querySelector(".breakdown");
    bWrap.innerHTML = "";
    let hasBreakdown = false;
    for (const d of DIMS) {
      const v = dimValue(breakdown, d);
      if (typeof v !== "number") continue;
      hasBreakdown = true;
      const row = document.createElement("div");
      row.className = "bd-row";

      const label = document.createElement("span");
      label.className = "bd-label";
      label.textContent = lang === "zh" ? d.zh : d.en;

      const track = document.createElement("div");
      track.className = "bd-track";
      const fill = document.createElement("div");
      fill.className = "bd-fill";
      fill.style.width = `${Math.max(0, Math.min(100, Math.round((v / d.max) * 100)))}%`;
      track.appendChild(fill);

      const val = document.createElement("span");
      val.className = "bd-val mono";
      val.textContent = `${v}/${d.max}`;

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      bWrap.appendChild(row);
    }
    if (!hasBreakdown) {
      const sec = bWrap.closest(".section");
      if (sec) sec.style.display = "none";
    }

    const figWrap = card.querySelector(".figure-wrap");
    const expWrap = card.querySelector(".exp-wrap");
    const figBtn = card.querySelector(".toggle-figure");
    const expBtn = card.querySelector(".toggle-exp");

    if (p.figure_path) {
      card.querySelector(".figure").src = p.figure_path;
    } else {
      figBtn.disabled = true;
      figBtn.classList.remove("btn-secondary");
    }

    // experiment / results figure (optional, shown above the highlight text)
    const expFig = card.querySelector(".exp-figure");
    if (p.exp_figure_path) {
      expFig.src = p.exp_figure_path;
    } else {
      expFig.remove();
    }

    const expText = lang === "zh" ? p.key_metrics_zh : p.key_metrics_en;
    card.querySelector(".exp").textContent = expText || "";

    if (!p.exp_figure_path && !expText) {
      expBtn.disabled = true;
      expBtn.classList.remove("btn-secondary");
    }

    figBtn.addEventListener("click", () => {
      figWrap.classList.toggle("hidden");
    });

    expBtn.addEventListener("click", () => {
      expWrap.classList.toggle("hidden");
    });

    // localize the cloned card's static labels (titles / buttons) for current lang
    node.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (I18N[lang][key]) el.textContent = I18N[lang][key];
    });

    grid.appendChild(node);
  }
}

function refresh() {
  const date = $("dateSelect").value;
  const minScore = Number($("scoreSlider").value);

  const rows = getPapersForDate(date, minScore);
  const day = getDayMeta(date);
  renderDayHeader(day, rows.length);
  renderPapers(rows);
}

function loadInlineData() {
  const inlineData = window.__PAPER_DATA__;
  if (
    !inlineData ||
    typeof inlineData !== "object" ||
    (!Array.isArray(inlineData.days) && !Array.isArray(inlineData.papers))
  ) {
    throw new Error("Inline data bundle is unavailable");
  }
  jsonData = inlineData;
}

async function loadJsonFallback() {
  const resp = await fetch("data/papers.json");
  if (!resp.ok) {
    throw new Error(`JSON fallback load failed: ${resp.status}`);
  }
  jsonData = await resp.json();
}

async function loadSqliteDatabase() {
  if (typeof window.initSqlJs !== "function") {
    throw new Error("sql.js is unavailable");
  }

  const SQL = await window.initSqlJs({
    locateFile: (file) => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${file}`,
  });

  const resp = await fetch("data/papers.sqlite");
  if (!resp.ok) {
    throw new Error(`SQLite load failed: ${resp.status}`);
  }

  const buf = await resp.arrayBuffer();
  db = new SQL.Database(new Uint8Array(buf));
}

async function init() {
  setLang("zh");
  setStatus(I18N[lang].loading);

  const protocol = window.location.protocol;
  const loaders = protocol === "file:"
    ? [loadInlineData, loadJsonFallback]
    : [loadJsonFallback, loadInlineData];

  let lastError = null;
  for (const load of loaders) {
    try {
      await load();
      lastError = null;
      break;
    } catch (error) {
      console.warn("Data load attempt failed.", error);
      db = null;
      jsonData = null;
      lastError = error;
    }
  }

  if (lastError) {
    throw lastError;
  }

  const dates = listDates();
  const dateSelect = $("dateSelect");
  dateSelect.innerHTML = "";
  for (const d of dates) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    dateSelect.appendChild(opt);
  }

  if (!dates.length) {
    renderPapers([]);
    return;
  }

  $("scoreValue").textContent = $("scoreSlider").value;

  $("scoreSlider").addEventListener("input", (e) => {
    $("scoreValue").textContent = e.target.value;
    refresh();
  });

  dateSelect.addEventListener("change", refresh);

  $("langToggle").addEventListener("click", () => {
    setLang(lang === "zh" ? "en" : "zh");
    refresh();
  });

  refresh();
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error("Failed to initialize paper webapp.", error);
    setStatus("加载失败，请检查 data/papers-inline.js / data/papers.json 是否可访问。");
  });
});

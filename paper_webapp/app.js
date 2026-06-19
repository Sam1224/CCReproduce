let db = null;
let lang = "zh";

const I18N = {
  zh: {
    date: "日期",
    minScore: "最低分",
    method: "方法概览",
    innovation: "创新点",
    breakdown: "评分明细",
    rationale: "评分依据",
    toggleFigure: "显示/隐藏 Method Figure",
    toggleExp: "显示/隐藏 亮眼实验",
    loading: "Loading…",
    empty: "当前筛选条件下没有论文。",
  },
  en: {
    date: "Date",
    minScore: "Min score",
    method: "Method overview",
    innovation: "Key innovations",
    breakdown: "Score breakdown",
    rationale: "Score rationale",
    toggleFigure: "Show/Hide methodology figure",
    toggleExp: "Show/Hide highlights",
    loading: "Loading…",
    empty: "No papers match the current filter.",
  },
};

// Score dimensions (key, max, labels) used to render the per-paper breakdown.
const DIMS = [
  { key: "innovation", max: 30, zh: "创新性", en: "Innovation" },
  { key: "results", max: 15, zh: "实验指标", en: "Results" },
  { key: "exp_quality", max: 15, zh: "实验质量", en: "Exp. quality" },
  { key: "efficiency", max: 10, zh: "效率", en: "Efficiency" },
  { key: "generalization", max: 5, zh: "泛化", en: "Generalization" },
  { key: "relevance", max: 25, zh: "相关性", en: "Relevance" },
];

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
  if (v == null) return null;
  if (typeof v !== "string") return v;
  try {
    return JSON.parse(v);
  } catch {
    return v;
  }
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
    const metaParts = [
      `${p.published ? `Published: ${p.published}` : ""}`,
      `${p.source || ""}`,
      `${authors.length ? `Authors: ${authors.join(", ")}` : ""}`,
      `${affiliations.length ? `Affiliations: ${affiliations.join(" / ")}` : ""}`,
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
    if (links.abs) linkItems.push({ name: "arXiv", url: links.abs });
    if (links.pdf) linkItems.push({ name: "PDF", url: links.pdf });
    if (links.code) linkItems.push({ name: "Code", url: links.code });
    if (links.project) linkItems.push({ name: "Project", url: links.project });
    if (p.reproduce_url) linkItems.push({ name: "Reproduce", url: p.reproduce_url });

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
    const innovationText = lang === "zh" ? p.innovation_zh : p.innovation_en;

    card.querySelector(".method").textContent = methodText || "";
    card.querySelector(".innovation").textContent = innovationText || "";
    card.querySelector(".rationale").textContent = p.rationale_zh || "";

    // score breakdown (per-dimension bars)
    const breakdown = parseJsonMaybe(p.score_breakdown) || {};
    const bWrap = card.querySelector(".breakdown");
    bWrap.innerHTML = "";
    let hasBreakdown = false;
    for (const d of DIMS) {
      const v = breakdown[d.key];
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

  const rows = query(
    `SELECT 
      inspection_date, paper_id, title, authors, affiliations, source, published, links, tags,
      score_total, score_breakdown, rationale_zh,
      method_overview_zh, method_overview_en,
      innovation_zh, innovation_en,
      key_metrics_zh, key_metrics_en, reproduce_url, figure_path, exp_figure_path
    FROM papers
    WHERE inspection_date = ? AND score_total >= ?
    ORDER BY score_total DESC, title ASC`,
    [date, minScore]
  );

  renderPapers(rows);
}

async function init() {
  setLang("zh");
  setStatus(I18N[lang].loading);

  const SQL = await initSqlJs({
    locateFile: (file) => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.10.2/${file}`,
  });

  const buf = await fetch("data/papers.sqlite").then((r) => r.arrayBuffer());
  db = new SQL.Database(new Uint8Array(buf));

  const dates = query(
    "SELECT DISTINCT inspection_date AS d FROM papers ORDER BY inspection_date DESC"
  ).map((r) => r.d);

  const dateSelect = $("dateSelect");
  dateSelect.innerHTML = "";
  for (const d of dates) {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    dateSelect.appendChild(opt);
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

window.addEventListener("DOMContentLoaded", init);

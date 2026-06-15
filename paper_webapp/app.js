let db = null;
let lang = "zh";

const I18N = {
  zh: {
    date: "日期",
    minScore: "最低分",
    method: "方法概览",
    innovation: "创新点",
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
    rationale: "Score rationale",
    toggleFigure: "Show/Hide methodology figure",
    toggleExp: "Show/Hide highlights",
    loading: "Loading…",
    empty: "No papers match the current filter.",
  },
};

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

    const expText = lang === "zh" ? p.key_metrics_zh : p.key_metrics_en;
    card.querySelector(".exp").textContent = expText || "";

    figBtn.addEventListener("click", () => {
      figWrap.classList.toggle("hidden");
    });

    expBtn.addEventListener("click", () => {
      expWrap.classList.toggle("hidden");
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
      key_metrics_zh, key_metrics_en, reproduce_url, figure_path
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

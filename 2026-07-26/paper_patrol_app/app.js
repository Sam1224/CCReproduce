const REPORT_DATE = "2026-07-26";
const DB_NAME = "paper_patrol_local_db";
const DB_VERSION = 1;
const STORE = "daily_reports";

const sourceClasses = ["全部", "今日热度高", "业务强相关", "补充强相关"];
const sourceClassEn = {
  "全部": "All",
  "今日热度高": "High Buzz Today",
  "业务强相关": "Business Critical",
  "补充强相关": "Supplemental Relevant"
};

const i18n = {
  zh: {
    eyebrow: "2026-07-26 巡检日报",
    heroTitle: "今日 paper 巡检结果分享台",
    heroDesc: "聚焦内容治理、多模态、Agent 与数据治理方向，用轻量本地数据库持久化每日巡检，支持按日期、分数、来源与标签快速筛选。",
    viewPapers: "查看论文卡片",
    papersCount: "论文数",
    avgScore: "平均分",
    topSource: "最高相关",
    reportDate: "Report date",
    scoreThreshold: "分数阈值",
    sourceClass: "Source 分类",
    tagFilter: "标签过滤",
    tagHint: "点击标签组合筛选，再次点击可取消。",
    filteredResult: "筛选结果",
    papersUnit: "篇论文",
    resetFilters: "重置过滤",
    emptyTitle: "没有匹配论文",
    emptyDesc: "请降低分数阈值，或取消部分标签 / 来源过滤。",
    paperDate: "Paper date",
    reportDateCard: "Report date",
    totalScore: "总分",
    scoreParts: "分项评分",
    rationale: "打分依据",
    overview: "方法概览",
    innovation: "创新点评",
    keyMetric: "关键实验",
    figure: "Methodology figure",
    experiment: "实验亮点",
    showFigure: "显示 methodology figure",
    hideFigure: "隐藏 methodology figure",
    showExperiment: "显示实验亮点",
    hideExperiment: "隐藏实验亮点",
    code: "代码链接",
    noCode: "暂无本地代码",
    dbReady: "IndexedDB · 本地数据库已就绪",
    dbFallback: "IndexedDB · 只读演示模式",
    all: "全部",
    scoreLabels: ["创新", "相关", "实验", "复现"]
  },
  en: {
    eyebrow: "2026-07-26 Patrol Digest",
    heroTitle: "Today’s Paper Patrol Shareboard",
    heroDesc: "A local-first research dashboard for content safety, multimodal systems, agents, and data governance. Filter daily patrols by date, score, source class, and tags.",
    viewPapers: "View paper cards",
    papersCount: "Papers",
    avgScore: "Average",
    topSource: "Top source",
    reportDate: "Report date",
    scoreThreshold: "Score threshold",
    sourceClass: "Source class",
    tagFilter: "Tag filter",
    tagHint: "Click tags to combine filters; click again to remove.",
    filteredResult: "Filtered result",
    papersUnit: "papers",
    resetFilters: "Reset filters",
    emptyTitle: "No matching papers",
    emptyDesc: "Lower the score threshold, or remove selected source / tag filters.",
    paperDate: "Paper date",
    reportDateCard: "Report date",
    totalScore: "Total score",
    scoreParts: "Score parts",
    rationale: "Score rationale",
    overview: "Method overview",
    innovation: "Innovation note",
    keyMetric: "Key experiment",
    figure: "Methodology figure",
    experiment: "Experiment highlights",
    showFigure: "Show methodology figure",
    hideFigure: "Hide methodology figure",
    showExperiment: "Show highlights",
    hideExperiment: "Hide highlights",
    code: "Code link",
    noCode: "No local code",
    dbReady: "IndexedDB · local database ready",
    dbFallback: "IndexedDB · read-only demo mode",
    all: "All",
    scoreLabels: ["Novelty", "Relevance", "Evidence", "Repro"]
  }
};

const seedReport = {
  reportDate: REPORT_DATE,
  updatedAt: new Date().toISOString(),
  papers: [
    {
      id: "dec_ob_definition_blindness",
      title: "RETHINKING OPEN-WORLD VIDEO ANOMALY DETECTION: DIAGNOSING DEFINITION BLINDNESS",
      authors: "Research team",
      institutions: "Open-world VAD benchmark authors",
      paperDate: "2026-07-24",
      reportDate: REPORT_DATE,
      sourceClass: "业务强相关",
      score: 93,
      scores: { novelty: 95, relevance: 97, evidence: 92, reproducibility: 88 },
      tags: ["内容治理", "视频异常检测", "评测", "多模态"],
      overview: {
        zh: "该工作指出开放世界视频异常检测存在“定义盲区”：模型能找到异常，但不随用户定义改变而改变排序。作者提出 DC-Disc/DC-DetΔ/DC-SelΔ 三个定义条件指标，并用 DeCoS 通过减去跨定义共享的泛异常证据来强化 definition following。方法保持对异常支撑，同时把分数质量从“是否异常”转向“是否符合当前定义”。",
        en: "The paper diagnoses “definition blindness” in open-world video anomaly detection: models can localize anomalies, yet their ranking barely changes when the queried abnormality definition changes. It introduces three definition-conditioned probes and proposes DeCoS, which subtracts anomaly evidence shared across definitions to improve true definition following while preserving anomaly support."
      },
      innovation: {
        zh: "创新点在于先拆解评测失真，再用 definition-contrastive scoring 进行最小干预式修正，兼顾评测与建模。",
        en: "Its key innovation is to expose an evaluation shortcut first, then fix it with a minimal definition-contrastive scoring rule rather than a heavy re-architecture."
      },
      keyMetric: "DC-DetΔ 提升 15.5–28.3 AUROC，DC-Disc 提升 7.3–16.0 AUROC。",
      scoreRationale: "创新与业务相关性双高，适合达人直播/短视频违规巡检。",
      experimentHighlights: {
        zh: "对比普通异常分数、定义条件分数与 DeCoS 后的 definition-contrastive 分数，核心提升集中在跨定义排序敏感度，而不是单纯异常召回。",
        en: "The key comparison separates generic anomaly scoring, definition-conditioned scoring, and DeCoS contrastive scoring, with gains concentrated on cross-definition ranking sensitivity rather than generic anomaly recall."
      },
      codeLink: { localPath: null, githubUrl: "https://github.com/Sam1224/CCReproduce/tree/aime/1785065085-daily-paper-patrol/2026-07-26/DeCoS_OWVAD" }
    },
    {
      id: "xuanwu_vl_2b",
      title: "Xuanwu: Evolving General Multimodal Models into an Industrial-Grade Foundation for Content Ecosystems",
      authors: "Xuanwu VL team",
      institutions: "Industrial content ecosystem research",
      paperDate: "2026-03-31",
      reportDate: REPORT_DATE,
      sourceClass: "补充强相关",
      score: 89,
      scores: { novelty: 86, relevance: 98, evidence: 90, reproducibility: 70 },
      tags: ["内容治理", "多模态", "对抗OCR", "工业落地"],
      overview: {
        zh: "玄武 VL-2B 采用 InternViT-300M 与 Qwen3-1.7B 的轻量组合，通过预训练、中训、后训练三阶段把通用多模态模型进化为工业级内容生态底座。系统重点强化细粒度视觉感知、长尾噪声鲁棒性和对抗 OCR 能力。",
        en: "Xuanwu VL-2B combines a compact InternViT-300M encoder with Qwen3-1.7B and uses a progressive three-stage pipeline to evolve a general multimodal model into an industrial-grade foundation for content ecosystems. The design emphasizes fine-grained visual perception, robustness to long-tail noise, and adversarial OCR understanding."
      },
      innovation: {
        zh: "亮点在于在 2B 级参数预算内，把多模态通用能力与工业审核任务对齐，并显式兼顾部署成本。",
        en: "The main innovation is balancing industrial moderation alignment, fine-grained perception, and deployment cost under a compact 2B-scale budget."
      },
      keyMetric: "对抗 OCR 场景加权综合召回 82.82%，高于 Gemini-2.5-Pro 的 76.72%。",
      scoreRationale: "业务相关性拉满，工业落地信号非常强，但实现复杂度极高。",
      experimentHighlights: {
        zh: "实验重点覆盖对抗 OCR、长尾噪声内容与多模态审核场景，证明小参数模型仍可通过阶段化训练获得工业可用能力。",
        en: "Experiments emphasize adversarial OCR, long-tail noisy content, and multimodal moderation, showing that a compact model can become industrially useful through staged training."
      },
      codeLink: { localPath: null, githubUrl: "https://github.com/Sam1224/CCReproduce/tree/aime/1785065085-daily-paper-patrol/2026-07-26/Xuanwu_VL_2B" }
    },
    {
      id: "evolving_user_intent",
      title: "LLMs Get Lost in Evolving User Intent",
      authors: "Dynamic intent evaluation authors",
      institutions: "Agent evaluation research",
      paperDate: "2026-07-24",
      reportDate: REPORT_DATE,
      sourceClass: "今日热度高",
      score: 81,
      scores: { novelty: 83, relevance: 84, evidence: 80, reproducibility: 78 },
      tags: ["Agent", "多轮交互", "评测", "个性化"],
      overview: {
        zh: "论文研究用户意图在多轮交互中不断披露、修正与转向时，LLM 代理如何失去追踪能力。作者把静态基准回溯改造成动态意图场景，发现顶尖模型在多次意图切换后准确率明显衰减。",
        en: "This paper studies how LLM agents lose track of evolving user intent when intent is gradually disclosed, revised, or redirected across a conversation. By transforming static benchmarks into dynamic intent scenarios, it shows that even frontier models degrade sharply after several intent shifts."
      },
      innovation: {
        zh: "其价值在于把“动态用户意图”单独抽象为评测维度，贴近真实导购/协作场景。",
        en: "It elevates evolving user intent into a first-class evaluation target, making agent assessment closer to real shopping-assistant and collaborative workflows."
      },
      keyMetric: "GPT-5.5 在数学任务里经历 6 次意图跳转后准确率从 99.0% 降至 80.5%。",
      scoreRationale: "对个性化导购、达人协作和长链路 Agent 都有直接启发。",
      experimentHighlights: {
        zh: "将静态任务包装为多轮变更轨迹，衡量模型是否保留最新约束并丢弃过期目标，适合作为 Agent 记忆与规划链路压力测试。",
        en: "Static tasks are wrapped into multi-turn change trajectories to measure whether agents preserve the latest constraints and discard obsolete goals."
      },
      codeLink: { localPath: null, githubUrl: "https://github.com/Sam1224/CCReproduce/tree/aime/1785065085-daily-paper-patrol/2026-07-26/Evolving_User_Intent" }
    },
    {
      id: "plane_meta_planning",
      title: "PlanE: Meta Planning of Data, Tuning, and Inference for Extractive-based LLMs",
      authors: "PlanE authors",
      institutions: "Extractive LLM systems research",
      paperDate: "2026-07-24",
      reportDate: REPORT_DATE,
      sourceClass: "业务强相关",
      score: 77,
      scores: { novelty: 78, relevance: 82, evidence: 76, reproducibility: 72 },
      tags: ["信息抽取", "Agent", "元规划", "评测"],
      overview: {
        zh: "PlanE 把抽取式 LLM 的构建流程拆成数据分解、对齐微调和推理规划三部分，并用 DTI Planner 预测最优组合。它的目标不是只优化单个模型，而是优化“如何构建模型”的策略。",
        en: "PlanE decomposes extractive LLM construction into data decomposition, tuning, and inference planning, then uses a DTI planner to predict the best combination. Its focus is not only the model itself, but the strategy for building the model efficiently."
      },
      innovation: {
        zh: "亮点是把工程配置搜索元学习化，适合平台治理中大量抽取任务的低成本定制。",
        en: "The novelty lies in turning engineering configuration search into a meta-planning problem for efficient task-specific extraction systems."
      },
      keyMetric: "在达到相近 F1 的同时，相比网格搜索节省 55 万秒以上搜索时间。",
      scoreRationale: "更偏方法工程提效，业务贴合中高，但不是最亮眼的治理模型。",
      experimentHighlights: {
        zh: "将数据、微调与推理策略作为组合动作空间，由 Planner 直接预测高性价比路径，避免昂贵网格搜索。",
        en: "Data, tuning, and inference choices form a combined action space; the planner predicts cost-effective paths without expensive grid search."
      },
      codeLink: { localPath: null, githubUrl: null }
    },
    {
      id: "influence_matching_distillation",
      title: "Dataset Distillation by Influence Matching",
      authors: "Inf-Match authors",
      institutions: "Dataset distillation research",
      paperDate: "2026-07-18",
      reportDate: REPORT_DATE,
      sourceClass: "今日热度高",
      score: 77,
      scores: { novelty: 82, relevance: 78, evidence: 76, reproducibility: 74 },
      tags: ["数据蒸馏", "训练加速", "模型治理", "多模态"],
      overview: {
        zh: "Inf-Match 直接对齐数据对最终模型参数的影响，而不是模仿训练轨迹。通过一阶泰勒近似高效估计样本影响，它能学习出更小但更有代表性的合成数据集。",
        en: "Inf-Match aligns synthetic data with the influence that real data exerts on final model parameters instead of matching the whole training trajectory. With a first-order Taylor approximation, it efficiently learns compact yet representative distilled datasets."
      },
      innovation: {
        zh: "提供了比梯度/轨迹匹配更“终局导向”的蒸馏视角，适合大规模数据清洗与打标成本优化。",
        en: "It introduces a more outcome-oriented objective for dataset distillation by matching final-parameter influence instead of intermediate optimization paths."
      },
      keyMetric: "Tiny-ImageNet IPC=10 准确率 31.5%，较 NCFM 提升 4.7%。",
      scoreRationale: "与数据治理链路高度相关，且 toy 复现友好。",
      experimentHighlights: {
        zh: "通过影响函数近似将真实样本与合成样本对最终参数的贡献对齐，减少对完整训练轨迹的依赖。",
        en: "Influence approximations align real and synthetic data contributions to final parameters, reducing reliance on full trajectory matching."
      },
      codeLink: { localPath: null, githubUrl: null }
    },
    {
      id: "clickguard_clickbait",
      title: "CLICKGUARD: DETECTING AND SPOILING CLICKBAIT NEWS",
      authors: "ClickGuard authors",
      institutions: "User experience and news moderation research",
      paperDate: "2026-07-24",
      reportDate: REPORT_DATE,
      sourceClass: "业务强相关",
      score: 75,
      scores: { novelty: 73, relevance: 86, evidence: 75, reproducibility: 72 },
      tags: ["内容治理", "标题党检测", "LLM", "用户体验"],
      overview: {
        zh: "ClickGuard 是一个浏览器扩展，结合 LLM 语义嵌入、人工语言学特征与 XGBoost 检测标题党，并用生成式摘要“剧透”正文来减少误点。",
        en: "ClickGuard is a browser extension that combines LLM embeddings, handcrafted linguistic features, and XGBoost to detect clickbait, then uses generated spoiler summaries to reduce curiosity-driven clicks."
      },
      innovation: {
        zh: "创新主要体现在“检测+剧透抑制点击”的完整治理闭环，而不是单点分类器。",
        en: "Its novelty is the full intervention loop—detect, then neutralize the click incentive with a spoiler-style summary."
      },
      keyMetric: "混合公开数据集上 F1 达到 91%。",
      scoreRationale: "非常贴合内容生态治理，但模型核心偏工程整合。",
      experimentHighlights: {
        zh: "混合语义嵌入和人工特征，检测后直接给出正文摘要，治理目标从分类准确扩展到用户行为干预。",
        en: "The system mixes semantic embeddings and handcrafted features, then uses generated summaries to shift from classification to behavioral intervention."
      },
      codeLink: { localPath: null, githubUrl: null }
    },
    {
      id: "abuse_detection_pipeline",
      title: "Large Language Models in the Abuse Detection Pipeline",
      authors: "Survey authors",
      institutions: "Abuse detection and governance research",
      paperDate: "2026-03-31",
      reportDate: REPORT_DATE,
      sourceClass: "补充强相关",
      score: 73,
      scores: { novelty: 66, relevance: 88, evidence: 73, reproducibility: 66 },
      tags: ["内容治理", "审核流程", "综述", "LLM"],
      overview: {
        zh: "这篇综述从标签生成、检测、人工复核到审计与治理四个阶段，系统分析了 LLM 如何嵌入 abuse detection pipeline。它更像一张内容治理全流程地图。",
        en: "This survey analyzes how LLMs fit into the abuse detection pipeline across label generation, detection, human review, and governance stages. It functions as a process map for end-to-end content safety systems."
      },
      innovation: {
        zh: "不是算法创新，而是生命周期框架梳理，适合作为治理体系设计参考。",
        en: "The contribution is not a new algorithm but a lifecycle framework for designing practical governance pipelines."
      },
      keyMetric: "文中引用 GPT-4 零样本毒性分类 F1 可超过 0.75。",
      scoreRationale: "体系价值强，但论文本身为综述，方法创新有限。",
      experimentHighlights: {
        zh: "按标签、检测、复核、审计四阶段梳理风险与能力边界，可作为业务治理流程查漏补缺清单。",
        en: "The four-stage map—labeling, detection, review, and audit—acts as a checklist for operational governance design."
      },
      codeLink: { localPath: null, githubUrl: null }
    }
  ]
};

let state = {
  lang: localStorage.getItem("paperPatrolLang") || "zh",
  reportDate: REPORT_DATE,
  minScore: Number(localStorage.getItem("paperPatrolMinScore") || 75),
  source: "全部",
  tags: new Set(),
  report: seedReport
};

function openDb() {
  return new Promise((resolve, reject) => {
    if (!("indexedDB" in window)) return reject(new Error("IndexedDB unavailable"));
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "reportDate" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function txStore(db, mode = "readonly") {
  return db.transaction(STORE, mode).objectStore(STORE);
}

function getReport(db, reportDate) {
  return new Promise((resolve, reject) => {
    const req = txStore(db).get(reportDate);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function putReport(db, report) {
  return new Promise((resolve, reject) => {
    const req = txStore(db, "readwrite").put(report);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function ensureSeed() {
  const status = document.getElementById("dbStatus");
  try {
    const db = await openDb();
    const existing = await getReport(db, REPORT_DATE);
    if (!existing) await putReport(db, seedReport);
    state.report = (await getReport(db, state.reportDate)) || { reportDate: state.reportDate, papers: [] };
    status.textContent = i18n[state.lang].dbReady;
    status.classList.remove("warn");
  } catch (error) {
    console.warn("IndexedDB failed, using seed data only", error);
    state.report = state.reportDate === REPORT_DATE ? seedReport : { reportDate: state.reportDate, papers: [] };
    status.textContent = i18n[state.lang].dbFallback;
    status.classList.add("warn");
  }
}

async function loadReportForDate(date) {
  state.reportDate = date;
  try {
    const db = await openDb();
    state.report = (await getReport(db, date)) || { reportDate: date, papers: [] };
  } catch (_) {
    state.report = date === REPORT_DATE ? seedReport : { reportDate: date, papers: [] };
  }
  render();
}

function t(key) { return i18n[state.lang][key]; }
function localizeSource(source) { return state.lang === "zh" ? source : (sourceClassEn[source] || source); }
function localizeScoreKey(key) {
  const map = { novelty: 0, relevance: 1, evidence: 2, reproducibility: 3 };
  return t("scoreLabels")[map[key]];
}

function getAllTags(papers = state.report.papers) {
  return [...new Set(papers.flatMap(p => p.tags))].sort((a, b) => a.localeCompare(b, "zh-Hans-CN"));
}

function filteredPapers() {
  return (state.report.papers || []).filter(paper => {
    const scorePass = paper.score >= state.minScore;
    const sourcePass = state.source === "全部" || paper.sourceClass === state.source;
    const tagPass = state.tags.size === 0 || [...state.tags].every(tag => paper.tags.includes(tag));
    return scorePass && sourcePass && tagPass;
  });
}

function methodologyFigure(paper) {
  const steps = figureSteps(paper.id);
  const palette = ["#87975a", "#c29a45", "#6e8493", "#c66f4b"];
  const nodes = steps.map((step, index) => {
    const x = 18 + index * 86;
    const fill = palette[index % palette.length];
    const arrow = index < steps.length - 1 ? `<path d="M${x + 60} 74 C${x + 70} 74 ${x + 72} 74 ${x + 82} 74" stroke="#9a927f" stroke-width="2" stroke-linecap="round"/><path d="M${x + 79} 69 L${x + 86} 74 L${x + 79} 79" fill="none" stroke="#9a927f" stroke-width="2" stroke-linecap="round"/>` : "";
    return `<g>
      <rect x="${x}" y="36" width="62" height="76" rx="14" fill="${fill}" opacity="0.92"/>
      <circle cx="${x + 31}" cy="59" r="13" fill="rgba(255,255,255,.32)"/>
      <text x="${x + 31}" y="64" text-anchor="middle" font-size="14" font-weight="800" fill="#fff">${index + 1}</text>
      <text x="${x + 31}" y="90" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">${step[0]}</text>
      <text x="${x + 31}" y="102" text-anchor="middle" font-size="9" font-weight="700" fill="#fff">${step[1] || ""}</text>
      ${arrow}
    </g>`;
  }).join("");
  return `<svg class="method-svg" viewBox="0 0 360 150" role="img" aria-label="${paper.title} methodology figure">
    <defs><filter id="soft"><feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#535f45" flood-opacity="0.12"/></filter></defs>
    <rect x="8" y="10" width="344" height="128" rx="20" fill="#fffdf6" stroke="rgba(83,96,72,.14)"/>
    <g filter="url(#soft)">${nodes}</g>
    <text x="180" y="130" text-anchor="middle" font-size="10" font-weight="800" fill="#65733d">${state.lang === "zh" ? "简化方法流程图" : "Simplified methodology flow"}</text>
  </svg>`;
}

function figureSteps(id) {
  const steps = {
    dec_ob_definition_blindness: [["Query", "Definition"], ["Generic", "Evidence"], ["Subtract", "Shared"], ["Rank", "Follow"]],
    xuanwu_vl_2b: [["Vision", "Encoder"], ["3-stage", "Training"], ["OCR", "Robust"], ["Moderate", "Deploy"]],
    evolving_user_intent: [["Static", "Task"], ["Intent", "Shift"], ["Memory", "Stress"], ["Accuracy", "Drop"]],
    plane_meta_planning: [["Data", "Split"], ["Tune", "Align"], ["Infer", "Plan"], ["DTI", "Select"]],
    influence_matching_distillation: [["Real", "Data"], ["Influence", "Approx"], ["Synthetic", "Set"], ["Train", "Fast"]],
    clickguard_clickbait: [["Title", "Signals"], ["LLM", "Embed"], ["XGBoost", "Detect"], ["Spoiler", "Summary"]],
    abuse_detection_pipeline: [["Label", "Stage"], ["Detect", "LLM"], ["Human", "Review"], ["Audit", "Govern"]]
  };
  return steps[id] || [["Input", ""], ["Model", ""], ["Score", ""], ["Decision", ""]];
}

function renderStaticText() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    if (t(key)) el.textContent = t(key);
  });
  document.querySelectorAll(".lang-btn").forEach(btn => btn.classList.toggle("is-active", btn.dataset.lang === state.lang));
  document.getElementById("dbStatus").textContent = document.getElementById("dbStatus").classList.contains("warn") ? t("dbFallback") : t("dbReady");
}

function renderSourceFilter() {
  const box = document.getElementById("sourceFilter");
  box.innerHTML = sourceClasses.map(source => `<button class="source-chip ${state.source === source ? "is-active" : ""}" data-source="${source}">${source === "全部" ? t("all") : localizeSource(source)}</button>`).join("");
}

function renderTags() {
  const box = document.getElementById("tagCloud");
  const tags = getAllTags();
  box.innerHTML = tags.map(tag => `<button class="tag-chip ${state.tags.has(tag) ? "is-active" : ""}" data-tag="${tag}">${tag}</button>`).join("");
}

function renderStats(papers) {
  const all = state.report.papers || [];
  const avg = all.length ? (all.reduce((sum, p) => sum + p.score, 0) / all.length).toFixed(1) : "0.0";
  document.getElementById("statTotal").textContent = all.length;
  document.getElementById("statAvg").textContent = avg;
  document.getElementById("statTopSource").textContent = state.lang === "zh" ? "业务强相关" : "Business Critical";
  document.getElementById("resultCount").textContent = papers.length;
  const bars = document.getElementById("scoreBars");
  bars.innerHTML = all.map(p => `<div class="bar-wrap" title="${p.title}: ${p.score}"><div class="bar" style="height:${Math.max(18, p.score)}px"></div><span class="bar-label">${p.score}</span></div>`).join("");
}

function renderCards(papers) {
  const grid = document.getElementById("paperGrid");
  const empty = document.getElementById("emptyState");
  empty.hidden = papers.length !== 0;
  grid.innerHTML = papers.map((paper, index) => {
    const overview = paper.overview[state.lang];
    const innovation = paper.innovation[state.lang];
    const highlight = paper.experimentHighlights[state.lang];
    const scoreParts = Object.entries(paper.scores).map(([key, value]) => `
      <div class="score-part"><span>${localizeScoreKey(key)}</span><div class="track"><div class="fill" style="width:${value}%"></div></div><strong>${value}</strong></div>
    `).join("");
    const code = paper.codeLink?.githubUrl || paper.codeLink?.localPath;
    return `<article class="paper-card" style="--score:${paper.score}; animation-delay:${index * 45}ms">
      <div class="card-head">
        <span class="source-badge">${localizeSource(paper.sourceClass)}</span>
        <div class="score-ring"><strong>${paper.score}</strong></div>
      </div>
      <h3 class="paper-title">${paper.title}</h3>
      <div class="meta-row">
        <span>${t("paperDate")}: ${paper.paperDate}</span>
        <span>${t("reportDateCard")}: ${paper.reportDate}</span>
      </div>
      <div class="meta-row">
        <span>${paper.authors}</span>
        <span>${paper.institutions}</span>
      </div>
      <div class="tags">${paper.tags.map(tag => `<span class="card-tag">${tag}</span>`).join("")}</div>
      <div>
        <p class="section-title">${t("scoreParts")}</p>
        <div class="score-parts">${scoreParts}</div>
      </div>
      <div class="rationale">
        <p class="section-title">${t("rationale")}</p>
        <p class="card-text">${paper.scoreRationale}</p>
      </div>
      <div>
        <p class="section-title">${t("overview")}</p>
        <p class="card-text">${overview}</p>
      </div>
      <div>
        <p class="section-title">${t("innovation")}</p>
        <p class="card-text">${innovation}</p>
      </div>
      <div>
        <p class="section-title">${t("keyMetric")}</p>
        <p class="metric-line">${paper.keyMetric}</p>
      </div>
      <div class="figure" id="fig-${paper.id}">${methodologyFigure(paper)}</div>
      <div class="experiment-panel" id="exp-${paper.id}"><strong>${t("experiment")} · </strong>${highlight}</div>
      <div class="card-actions">
        <button class="toggle-btn" data-toggle="fig-${paper.id}" data-open="${t("hideFigure")}" data-close="${t("showFigure")}">${t("showFigure")}</button>
        <button class="toggle-btn" data-toggle="exp-${paper.id}" data-open="${t("hideExperiment")}" data-close="${t("showExperiment")}">${t("showExperiment")}</button>
        ${code ? `<a class="code-link" href="${code}" target="_blank" rel="noreferrer">${t("code")}</a>` : `<span class="toggle-btn" aria-disabled="true">${t("noCode")}</span>`}
      </div>
    </article>`;
  }).join("");
}

function render() {
  renderStaticText();
  document.getElementById("reportDate").value = state.reportDate;
  document.getElementById("scoreThreshold").value = state.minScore;
  document.getElementById("scoreValue").textContent = state.minScore;
  const papers = filteredPapers();
  renderSourceFilter();
  renderTags();
  renderStats(papers);
  renderCards(papers);
}

function bindEvents() {
  document.querySelectorAll(".lang-btn").forEach(btn => btn.addEventListener("click", () => {
    state.lang = btn.dataset.lang;
    localStorage.setItem("paperPatrolLang", state.lang);
    render();
  }));
  document.getElementById("reportDate").addEventListener("change", e => loadReportForDate(e.target.value));
  document.getElementById("scoreThreshold").addEventListener("input", e => {
    state.minScore = Number(e.target.value);
    localStorage.setItem("paperPatrolMinScore", String(state.minScore));
    render();
  });
  document.getElementById("sourceFilter").addEventListener("click", e => {
    const btn = e.target.closest("[data-source]");
    if (!btn) return;
    state.source = btn.dataset.source;
    render();
  });
  document.getElementById("tagCloud").addEventListener("click", e => {
    const btn = e.target.closest("[data-tag]");
    if (!btn) return;
    const tag = btn.dataset.tag;
    state.tags.has(tag) ? state.tags.delete(tag) : state.tags.add(tag);
    render();
  });
  document.getElementById("paperGrid").addEventListener("click", e => {
    const btn = e.target.closest("[data-toggle]");
    if (!btn) return;
    const panel = document.getElementById(btn.dataset.toggle);
    panel.classList.toggle("is-open");
    btn.textContent = panel.classList.contains("is-open") ? btn.dataset.open : btn.dataset.close;
  });
  document.getElementById("resetFilters").addEventListener("click", () => {
    state.minScore = 75;
    state.source = "全部";
    state.tags.clear();
    localStorage.setItem("paperPatrolMinScore", "75");
    render();
  });
}

(async function init() {
  bindEvents();
  await ensureSeed();
  render();
})();

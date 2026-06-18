#!/usr/bin/env python3
"""Generate research-style methodology SVG figures for the 2026-06-18 papers."""
import os

OUT = os.path.join(os.path.dirname(__file__), "assets", "figures")
os.makedirs(OUT, exist_ok=True)

W, H = 980, 430
PAL = {
    "bg": "#ffffff", "panel": "#f7f8fb", "line": "#cbd5e1", "text": "#0f172a",
    "muted": "#64748b", "blue": "#2563eb", "green": "#10b981", "amber": "#f59e0b",
    "violet": "#7c3aed", "rose": "#e11d48", "cyan": "#0891b2",
}


def header():
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter, system-ui, sans-serif">'
        f'<defs><marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">'
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{PAL["muted"]}"/></marker>'
        f'<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.10"/></filter></defs>'
        f'<rect width="{W}" height="{H}" fill="{PAL["bg"]}"/>'
    )


def box(x, y, w, h, fill, stroke, rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#sh)"/>'


def txt(x, y, s, size=13, fill=None, weight="400", anchor="middle"):
    fill = fill or PAL["text"]
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>'


def mlines(x, y, lines, size=12, fill=None, weight="400", anchor="middle", dy=16):
    out = ""
    for i, ln in enumerate(lines):
        out += txt(x, y + i * dy, ln, size, fill, weight, anchor)
    return out


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{PAL["muted"]}" stroke-width="1.8" marker-end="url(#arr)"/>'


def title_bar(t, sub):
    s = txt(28, 34, t, 17, PAL["text"], "700", "start")
    s += txt(28, 54, sub, 12, PAL["muted"], "400", "start")
    s += f'<line x1="28" y1="66" x2="{W-28}" y2="66" stroke="{PAL["line"]}" stroke-width="1"/>'
    return s


def save(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(header() + body + "</svg>")
    print("wrote", name)


# ---------- 1. OmniAgent (2606.19341) ----------
def omniagent():
    b = title_bar("OmniAgent — Native Active Perception (POMDP / OTA cycle)",
                  "Long video + audio  ->  on-demand Observe-Think-Act  ->  persistent textual memory  ->  answer")
    b += box(40, 100, 150, 120, PAL["panel"], PAL["cyan"])
    b += txt(115, 124, "Input (hours)", 13, PAL["cyan"], "700")
    b += mlines(115, 150, ["🎞 frames", "🔊 audio", "🎬 AV clips"], 12, PAL["muted"])
    b += txt(115, 208, "query: 'why did", 11, PAL["muted"])
    # OTA cycle boxes
    cx = [260, 470, 680]
    labels = [("Observe", "browse / listen /", "watch on-demand", PAL["blue"]),
              ("Think", "reason over", "memory + percept", PAL["violet"]),
              ("Act", "next action or", "STOP & answer", PAL["green"])]
    for x, (t, l1, l2, c) in zip(cx, labels):
        b += box(x, 110, 150, 80, "#ffffff", c)
        b += txt(x + 75, 134, t, 14, c, "700")
        b += txt(x + 75, 154, l1, 11, PAL["muted"])
        b += txt(x + 75, 170, l2, 11, PAL["muted"])
    b += arrow(190, 150, 258, 150)
    b += arrow(410, 150, 468, 150)
    b += arrow(620, 150, 678, 150)
    # loop back
    b += f'<path d="M755 190 q 30 60 -250 60 q -280 0 -245 -60" fill="none" stroke="{PAL["amber"]}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arr)"/>'
    b += txt(490, 268, "iterate until sufficient evidence (test-time scaling)", 12, PAL["amber"], "600")
    # memory + output
    b += box(330, 300, 300, 56, PAL["panel"], PAL["violet"])
    b += txt(480, 324, "Persistent Textual Memory", 13, PAL["violet"], "700")
    b += txt(480, 342, "state depends on reasoning, not video length", 11, PAL["muted"])
    b += box(700, 300, 240, 56, "# eff6ff".replace(" ", ""), PAL["green"])
    b += txt(820, 324, "Answer / grounding", 13, PAL["green"], "700")
    b += txt(820, 342, "LVBench 50.5% (>72B)", 11, PAL["muted"])
    b += arrow(630, 328, 698, 328)
    # training note
    b += txt(40, 392, "Train: Agentic SFT (best-of-N) → Agentic RL w/ TAURA (turn-level entropy credit assignment)", 12, PAL["muted"], "500", "start")
    return b


# ---------- 2. ProductConsistency (2606.19103) ----------
def productconsistency():
    b = title_bar("ProductConsistency — identity-preserving product image editing (SFT + RL)",
                  "Cyclic Consistency reward: description → edit → re-caption → similarity")
    b += box(40, 100, 150, 110, PAL["panel"], PAL["cyan"])
    b += txt(115, 124, "Product image", 13, PAL["cyan"], "700")
    b += txt(115, 148, "🧴 logo + text", 11, PAL["muted"])
    b += txt(115, 168, "desc: 'BrandX", 10, PAL["muted"])
    b += txt(115, 182, "serum 30ml'", 10, PAL["muted"])
    b += txt(115, 200, "+ edit instruction", 10, PAL["muted"])
    b += box(250, 110, 160, 90, "#ffffff", PAL["blue"])
    b += txt(330, 138, "Editor (SFT)", 14, PAL["blue"], "700")
    b += txt(330, 158, "Qwen-Image-Edit /", 11, PAL["muted"])
    b += txt(330, 174, "Flux.1-Kontext", 11, PAL["muted"])
    b += box(470, 110, 150, 90, PAL["panel"], PAL["green"])
    b += txt(545, 138, "Edited image", 13, PAL["green"], "700")
    b += txt(545, 160, "new scene,", 11, PAL["muted"])
    b += txt(545, 176, "same identity", 11, PAL["muted"])
    b += arrow(190, 155, 248, 155)
    b += arrow(410, 155, 468, 155)
    # cyclic reward loop
    b += box(470, 250, 150, 70, "#ffffff", PAL["violet"])
    b += txt(545, 276, "Re-caption", 13, PAL["violet"], "700")
    b += txt(545, 296, "(VLM/OCR)", 11, PAL["muted"])
    b += box(250, 250, 160, 70, "#ffffff", PAL["rose"])
    b += txt(330, 272, "Cyclic Consistency", 12, PAL["rose"], "700")
    b += txt(330, 292, "sim(desc, caption)", 11, PAL["muted"])
    b += txt(330, 308, "→ RL reward", 11, PAL["muted"])
    b += arrow(545, 200, 545, 248)
    b += arrow(468, 285, 412, 285)
    b += f'<path d="M250 285 q -70 0 -70 -85 q 0 -45 70 -45" fill="none" stroke="{PAL["rose"]}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#arr)"/>'
    b += txt(120, 250, "reward → policy update", 11, PAL["rose"], "600")
    b += box(700, 150, 240, 110, PAL["panel"], PAL["amber"])
    b += txt(820, 176, "Outcome", 14, PAL["amber"], "700")
    b += mlines(820, 200, ["OCR CER ↓ ~5×", "Seg CLIP-I / DINO-I ↑", "MLLM judge ↑", "(brand+text fidelity)"], 11, PAL["muted"])
    b += arrow(620, 175, 698, 175)
    b += txt(40, 400, "Data: 87k SFT + 869 RL synthetic products w/ unique brand & verifiable text; benchmark 174 imgs / 870 samples", 12, PAL["muted"], "500", "start")
    return b


# ---------- 3. SAMA (2606.18780) ----------
def sama():
    b = title_bar("SAMA — Semantic Anchor-aligned Multimodal Augmentation (MNER/MRE/MEE)",
                  "Anchors from labels guide unified multi-expert text + anchor-preserving image synthesis")
    b += box(40, 100, 150, 90, PAL["panel"], PAL["cyan"])
    b += txt(115, 126, "Labeled pair", 13, PAL["cyan"], "700")
    b += txt(115, 148, "text + image", 11, PAL["muted"])
    b += txt(115, 168, "+ GT labels", 11, PAL["muted"])
    b += box(230, 110, 140, 70, "#ffffff", PAL["rose"])
    b += txt(300, 138, "Semantic", 13, PAL["rose"], "700")
    b += txt(300, 156, "Anchors", 13, PAL["rose"], "700")
    b += arrow(190, 145, 228, 145)
    # CME-MLLM experts
    b += box(420, 96, 200, 120, "#ffffff", PAL["blue"])
    b += txt(520, 118, "CME-MLLM", 14, PAL["blue"], "700")
    b += txt(520, 136, "Universal Adapter", 11, PAL["muted"])
    for i, e in enumerate(["MNER", "MRE", "MEE"]):
        b += box(432 + i * 62, 150, 56, 50, PAL["panel"], PAL["violet"], 8)
        b += txt(460 + i * 62, 180, e, 11, PAL["violet"], "700")
    b += arrow(370, 145, 418, 145)
    # diffusion
    b += box(420, 250, 200, 64, PAL["panel"], PAL["green"])
    b += txt(520, 276, "Anchor-Preserving", 12, PAL["green"], "700")
    b += txt(520, 294, "Diffusion (image)", 12, PAL["muted"])
    b += arrow(300, 180, 300, 282)
    b += arrow(370, 282, 418, 282)
    # filter
    b += box(680, 150, 130, 110, "#ffffff", PAL["amber"])
    b += txt(745, 178, "Dual-", 13, PAL["amber"], "700")
    b += txt(745, 196, "Constraint", 13, PAL["amber"], "700")
    b += txt(745, 218, "Filter", 13, PAL["amber"], "700")
    b += txt(745, 240, "auto, no human", 10, PAL["muted"])
    b += arrow(620, 156, 678, 156)
    b += arrow(620, 282, 678, 240)
    b += box(840, 165, 110, 80, PAL["panel"], PAL["green"])
    b += txt(895, 195, "Synthetic", 12, PAL["green"], "700")
    b += txt(895, 213, "train data", 12, PAL["muted"])
    b += txt(895, 231, "F1 ↑ 1-3", 11, PAL["muted"])
    b += arrow(810, 205, 838, 205)
    b += txt(40, 400, "Unified anchors close the cross-modal gap and let MNER/MRE/MEE share semantics; gains larger in low-resource", 12, PAL["muted"], "500", "start")
    return b


# ---------- 4. APT (2606.18586) ----------
def apt():
    b = title_bar("APT — Atomic Physical Transitions for causal video understanding",
                  "Event label says WHAT; APT chain (typed, timestamped) explains WHY")
    b += box(40, 100, 150, 95, PAL["panel"], PAL["cyan"])
    b += txt(115, 126, "Video clip", 13, PAL["cyan"], "700")
    b += txt(115, 148, "event: 'bounce'", 11, PAL["muted"])
    b += txt(115, 168, "16 frames", 11, PAL["muted"])
    # APT chain
    chain = [("free_fall", PAL["blue"]), ("contact_init", PAL["violet"]), ("elastic_rebound", PAL["green"]), ("sliding_arrest", PAL["amber"])]
    x0 = 230
    for i, (t, c) in enumerate(chain):
        x = x0 + i * 180
        b += box(x, 110, 150, 56, "#ffffff", c, 10)
        b += txt(x + 75, 134, "t%d" % i, 11, c, "700")
        b += txt(x + 75, 152, t, 11, PAL["muted"])
        if i < 3:
            b += arrow(x + 150, 138, x + 180 + 0, 138)
    b += arrow(190, 140, 228, 140)
    b += txt(490, 196, "ordered causal transition chain (mechanism-typed)", 12, PAL["muted"], "600")
    # training
    b += box(120, 250, 220, 80, "#ffffff", PAL["rose"])
    b += txt(230, 276, "APT-Tune (11M LoRA)", 13, PAL["rose"], "700")
    b += mlines(230, 298, ["format-conditional co-training", "image-pad-aware supervision"], 10, PAL["muted"])
    b += box(420, 250, 200, 80, PAL["panel"], PAL["blue"])
    b += txt(520, 276, "Qwen3-VL-2B", 13, PAL["blue"], "700")
    b += txt(520, 296, "no event-level forgetting", 11, PAL["muted"])
    b += arrow(340, 290, 418, 290)
    b += box(700, 250, 240, 80, PAL["panel"], PAL["green"])
    b += txt(820, 276, "Results", 13, PAL["green"], "700")
    b += mlines(820, 298, ["APT recall 10→38.1%", "SSv2 MC +15pp, →PhysBench"], 11, PAL["muted"])
    b += arrow(620, 290, 698, 290)
    b += txt(40, 400, "Data: human + simulator GT, 14 transition types, 27,303 timed instances over 1,246 trials", 12, PAL["muted"], "500", "start")
    return b


# ---------- 5. DSG (2606.18947) ----------
def dsg():
    b = title_bar("DSG — Decoupled Search Grounding (vendor-agnostic MCP gateway)",
                  "Move grounding OUT of the reasoning model; retrieval becomes a controllable interface")
    b += box(40, 120, 150, 90, PAL["panel"], PAL["blue"])
    b += txt(115, 150, "LLM Agent", 13, PAL["blue"], "700")
    b += txt(115, 172, "(interchangeable", 10, PAL["muted"])
    b += txt(115, 186, "reasoning model)", 10, PAL["muted"])
    b += box(260, 100, 220, 230, "#ffffff", PAL["violet"])
    b += txt(370, 124, "DSG Gateway (MCP)", 13, PAL["violet"], "700")
    feats = ["Provider routing", "Source-aware context", "Configured fallback", "Retrieval-depth ctrl", "Exact + semantic cache"]
    for i, fch in enumerate(feats):
        b += box(276, 142 + i * 35, 188, 28, PAL["panel"], PAL["line"], 7)
        b += txt(370, 161 + i * 35, fch, 11, PAL["text"])
    b += arrow(190, 165, 258, 165)
    b += f'<line x1="190" y1="185" x2="258" y2="185" stroke="{PAL["muted"]}" stroke-width="1.8" marker-end="url(#arr)"/>'
    # providers
    for i, p in enumerate(["Search A", "Search B", "Search C"]):
        b += box(540, 110 + i * 50, 110, 40, PAL["panel"], PAL["cyan"], 8)
        b += txt(595, 135 + i * 50, p, 12, PAL["cyan"], "700")
        b += arrow(480, 130 + i * 30, 538, 130 + i * 50)
    b += box(720, 130, 220, 140, PAL["panel"], PAL["green"])
    b += txt(830, 156, "Outcomes", 14, PAL["green"], "700")
    b += mlines(830, 182, ["SimpleQA 86.1% vs 87.7%", "search cost −91% (−98% QIU)", "warm-cache hit 99.4%", "latency −68%", "no Search-Induced Verbosity"], 11, PAL["muted"])
    b += arrow(660, 175, 718, 175)
    b += txt(40, 400, "E-commerce Query Intent Understanding (QIU) production workload: matches native search at >98% lower cost", 12, PAL["muted"], "500", "start")
    return b


# ---------- 6. AI-Gen text-rich image detection (2606.19259) ----------
def aigen():
    b = title_bar("Benchmark: detecting AI-generated text-rich images (GPT-Image-2)",
                  "8,602 imgs · 6 domains · zero-shot eval of 5 detectors + 1 VLM")
    b += box(40, 100, 170, 150, PAL["panel"], PAL["cyan"])
    b += txt(125, 124, "GPT-Image-2", 13, PAL["cyan"], "700")
    cats = ["Commercial poster", "Infographic", "Academic poster", "Receipt", "Table", "UI screenshot"]
    for i, c in enumerate(cats):
        b += txt(125, 148 + i * 16, "• " + c, 10, PAL["muted"])
    b += box(270, 130, 180, 90, "#ffffff", PAL["violet"])
    b += txt(360, 158, "Privacy-preserving", 12, PAL["violet"], "700")
    b += txt(360, 176, "prompt synthesis", 12, PAL["muted"])
    b += txt(360, 196, "(no real text/identity)", 10, PAL["muted"])
    b += arrow(210, 175, 268, 175)
    b += box(500, 130, 170, 90, PAL["panel"], PAL["blue"])
    b += txt(585, 158, "Zero-shot eval", 13, PAL["blue"], "700")
    b += txt(585, 178, "5 detectors", 11, PAL["muted"])
    b += txt(585, 196, "+ 1 multimodal VLM", 11, PAL["muted"])
    b += arrow(450, 175, 498, 175)
    b += box(720, 110, 220, 150, "#ffffff", PAL["amber"])
    b += txt(830, 136, "Findings", 14, PAL["amber"], "700")
    b += mlines(830, 162, ["strongly domain-dependent", "fragile to JPEG compression", "VLM helps overall but", "unreliable on structured", "text-rich images"], 11, PAL["muted"])
    b += arrow(670, 175, 718, 175)
    b += txt(40, 400, "Motivates text-/layout-aware detection for content authenticity & governance", 12, PAL["muted"], "500", "start")
    return b


save("2606.19341.svg", omniagent())
save("2606.19103.svg", productconsistency())
save("2606.18780.svg", sama())
save("2606.18586.svg", apt())
save("2606.18947.svg", dsg())
save("2606.19259.svg", aigen())
print("done")

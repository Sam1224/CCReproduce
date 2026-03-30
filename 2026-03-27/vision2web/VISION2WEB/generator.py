from __future__ import annotations

from spec import WebSpec


def plan_hierarchy(spec: WebSpec) -> dict:
    # A minimal “hierarchical plan”: layout stage then components stage.
    layout = {
        "nav": spec.has_nav,
        "search": spec.has_search,
        "footer": spec.has_footer,
    }
    components = {
        "cards": spec.num_cards,
    }
    return {"layout": layout, "components": components}


def render_html(spec: WebSpec, template_id: int) -> str:
    plan = plan_hierarchy(spec)

    # Template controls broad page structure; spec controls component counts.
    body_parts = []

    tmpl_has_nav = template_id in (0, 1)
    tmpl_has_search = template_id in (0, 2)

    if tmpl_has_nav:
        body_parts.append('<nav class="nav">\n  <div class="brand">DemoShop</div>\n</nav>')

    if template_id == 1:
        body_parts.append('<div class="sidebar">\n  <a href="#">Home</a>\n  <a href="#">Explore</a>\n</div>')

    main_parts = []
    if tmpl_has_search:
        main_parts.append('<input class="search" type="search" placeholder="Search..." />')

    cards = []
    for i in range(plan["components"]["cards"]):
        cards.append(f'<div class="card"><div class="title">Item {i+1}</div><button>Buy</button></div>')
    main_parts.append('<div class="grid">' + "".join(cards) + "</div>")

    body_parts.append('<main class="main">' + "".join(main_parts) + "</main>")

    if plan["layout"]["footer"]:
        body_parts.append('<footer class="footer">© DemoShop</footer>')

    css = """
    :root { --bg: #f6f8fb; --card: #ffffff; --text: #1f2a37; --muted: #6b7280; --border: #e5e7eb; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system; background: var(--bg); color: var(--text); }
    .nav { padding: 12px 18px; background: #ffffff; border-bottom: 1px solid var(--border); position: sticky; top: 0; }
    .brand { font-weight: 700; letter-spacing: 0.2px; }
    .sidebar { position: fixed; top: 52px; left: 0; width: 160px; bottom: 0; padding: 14px; background: #ffffff; border-right: 1px solid var(--border); }
    .sidebar a { display: block; color: var(--muted); text-decoration: none; margin: 8px 0; }
    .main { padding: 18px; max-width: 1100px; margin: 0 auto; }
    .search { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: #fff; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; box-shadow: 0 1px 1px rgba(0,0,0,0.02); }
    .card .title { font-size: 14px; margin-bottom: 10px; }
    .card button { border: 1px solid var(--border); background: #111827; color: #fff; border-radius: 10px; padding: 8px 10px; cursor: pointer; }
    .footer { padding: 16px 18px; color: var(--muted); }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    """

    # If sidebar template, shift main.
    if template_id == 1:
        css += "\n.main { margin-left: 180px; }"

    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>""".format(css=css, body="\n".join(body_parts))

    return html

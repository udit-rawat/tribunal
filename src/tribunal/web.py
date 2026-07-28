"""Minimal FastAPI web UI. Run: `uvicorn tribunal.web:app --reload` then open http://127.0.0.1:8000"""

from __future__ import annotations

import html

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from .graph import verify_claim

app = FastAPI(title="Tribunal")

_COLORS = {
    "True": "#1a7f37",
    "False": "#cf222e",
    "Misleading": "#bf8700",
    "Unverifiable": "#57606a",
}

_PAGE = """<!doctype html><meta charset="utf-8"><title>Tribunal</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;max-width:720px;margin:3rem auto;padding:0 1rem;color:#1f2328}}
 h1{{font-size:1.6rem}} input{{width:100%;padding:.7rem;font-size:1rem;box-sizing:border-box}}
 button{{margin-top:.6rem;padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}}
 .card{{margin-top:2rem;padding:1.2rem 1.4rem;border:1px solid #d0d7de;border-radius:10px}}
 .verdict{{font-size:1.3rem;font-weight:700}} .muted{{color:#57606a}} .cite{{margin:.5rem 0}}
</style>
<h1>⚖️ Tribunal</h1>
<p class="muted">Adversarial multi-agent fact-checker with RAG grounding.</p>
<form method="post" action="/verify">
 <input name="claim" placeholder="Enter a claim to fact-check…" value="{claim}" autofocus>
 <button type="submit">Verify</button>
</form>
{result}
"""


def _render_card(r: dict) -> str:
    color = _COLORS.get(r["verdict"], "#57606a")
    cites = "".join(
        f'<div class="cite">{"➕" if c["supports"] else "➖"} '
        f'“{html.escape(c["quote"][:200])}” '
        f'<a href="{html.escape(c.get("source_url", ""))}" class="muted">source</a></div>'
        for c in r.get("citations", [])
    )
    return f"""<div class="card">
      <div class="verdict" style="color:{color}">{r['verdict']}
        <span class="muted">· {r['confidence']:.0%} confidence</span></div>
      <p><b>{html.escape(r['summary'])}</b></p>
      <p>{html.escape(r['reasoning'])}</p>
      {cites}
    </div>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE.format(claim="", result="")


@app.post("/verify", response_class=HTMLResponse)
def verify(claim: str = Form(...)) -> str:
    result = verify_claim(claim)
    return _PAGE.format(claim=html.escape(claim), result=_render_card(result))

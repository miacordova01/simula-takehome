#!/usr/bin/env bash
# Render the markdown docs to styled HTML and open them in your browser.
#
#   ./view.sh          -> render everything, open README + the 3 filming tabs
#   ./view.sh --all    -> render everything, open every report too
#
# Output goes to _rendered/ (gitignored). Re-run any time the reports change.
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python - "$@" <<'PY'
import sys, webbrowser
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
ROOT = Path.cwd()
OUT = ROOT / "_rendered"
OUT.mkdir(exist_ok=True)

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font: 17px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 1000px; margin: 0 auto; padding: 3rem 2rem 6rem;
  color: #1a1a1a; background: #fff;
}
h1 { font-size: 2.1rem; margin: 0 0 1.5rem; padding-bottom: .5rem; border-bottom: 3px solid #2563eb; }
h2 { font-size: 1.5rem; margin: 2.5rem 0 1rem; padding-bottom: .3rem; border-bottom: 1px solid #e5e7eb; }
h3 { font-size: 1.2rem; margin: 2rem 0 .75rem; color: #374151; }
p, li { font-size: 1.05rem; }
code { font: 15px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
       background: #f3f4f6; padding: .15em .4em; border-radius: 4px; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
      padding: 1rem 1.25rem; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 14.5px; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin: 1.25rem 0; font-size: 1rem; }
th, td { border: 1px solid #e5e7eb; padding: .55rem .85rem; text-align: left; }
th { background: #f9fafb; font-weight: 600; }
tr:nth-child(even) td { background: #fcfcfd; }
blockquote { border-left: 4px solid #f59e0b; background: #fffbeb; margin: 1.5rem 0;
             padding: .85rem 1.25rem; border-radius: 0 6px 6px 0; }
strong { color: #111827; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 2.5rem 0; }
a { color: #2563eb; }
.nav { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #e5e7eb;
       margin: -3rem -2rem 2rem; padding: .75rem 2rem; font-size: .9rem; }
.nav a { margin-right: 1rem; text-decoration: none; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1117; color: #e5e7eb; }
  h3 { color: #cbd5e1; } strong { color: #f9fafb; }
  code { background: #1f2430; }
  pre { background: #161a22; border-color: #2a2f3a; }
  th, td { border-color: #2a2f3a; } th { background: #1a1f29; }
  tr:nth-child(even) td { background: #141821; }
  blockquote { background: #251f0f; }
  .nav { background: #0f1117; border-color: #2a2f3a; }
}
"""

docs = [ROOT / "README.md"] + sorted((ROOT / "reports").glob("*.md"))
docs = [d for d in docs if d.exists()]

# Code files worth showing on screen, rendered with syntax highlighting.
CODE = [ROOT / "scripts/debug_staleness.py", ROOT / "simula/ranker.py",
        ROOT / "scripts/demo.py", ROOT / "simula/features.py"]
CODE = [c for c in CODE if c.exists()]

def label_for(p: Path) -> str:
    return "README" if p.name == "README.md" else p.stem

nav = " ".join(f'<a href="{label_for(d)}.html">{label_for(d)}</a>' for d in docs)
nav += ' &nbsp;|&nbsp; ' + " ".join(
    f'<a href="{c.stem}.html">{c.stem}.py</a>' for c in CODE
)

written = {}
for d in docs:
    html = markdown.markdown(
        d.read_text(),
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    name = label_for(d)
    page = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{name}</title><style>{CSS}</style></head><body>"
            f"<div class='nav'>{nav}</div>{html}</body></html>")
    p = OUT / f"{name}.html"
    p.write_text(page)
    written[name] = p
    print(f"  rendered {d.name:<24} -> _rendered/{name}.html")

# --- code files, syntax highlighted with line numbers ------------------
try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    fmt = HtmlFormatter(linenos="table", style="friendly", cssclass="hl")
    hl_css = fmt.get_style_defs(".hl")
    for c in CODE:
        body = highlight(c.read_text(), PythonLexer(), fmt)
        page = (f"<!doctype html><html><head><meta charset='utf-8'>"
                f"<title>{c.stem}.py</title><style>{CSS}"
                f".hl {{ font-size: 14px; line-height: 1.5; }}"
                f".hl pre {{ margin:0; border:none; background:none; }}"
                f".hl table {{ border:none; }} .hl td {{ border:none; padding:0 .5rem; }}"
                f".linenos {{ color:#9ca3af; user-select:none; }}"
                f"{hl_css}</style></head><body>"
                f"<div class='nav'>{nav}</div><h1>{c.relative_to(ROOT)}</h1>"
                f"{body}</body></html>")
        p = OUT / f"{c.stem}.html"
        p.write_text(page)
        written[c.stem] = p
        print(f"  rendered {c.name:<24} -> _rendered/{c.stem}.html")
except ImportError:
    print("  (pip install pygments for syntax-highlighted code views)")

open_all = "--all" in sys.argv
order = (["README", "signal_audit", "drift", "debug_staleness", "model_results"]
         if not open_all else list(written))
for name in order:
    if name in written:
        webbrowser.open(f"file://{written[name]}")

print(f"\nOpened {len([n for n in order if n in written])} tab(s) in your browser.")
print("Re-run ./view.sh after regenerating reports.")
PY

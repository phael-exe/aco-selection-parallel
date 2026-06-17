#!/usr/bin/env python3
"""Converte um .md em .pdf via python-markdown -> HTML -> Google Chrome headless.

Imagens com caminho relativo são resolvidas a partir da pasta do .md (o HTML
intermediário é escrito ali). Uso:
    python3 scripts/md_to_pdf.py results/EP05_CDC_grid_report.md
    python3 scripts/md_to_pdf.py entrada.md -o saida.pdf
"""
import argparse
import pathlib
import shutil
import subprocess

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt; line-height: 1.5; color: #1f2328; max-width: 100%; margin: 0;
}
h1 { font-size: 21pt; border-bottom: 2px solid #d0d7de; padding-bottom: .2em; }
h2 { font-size: 15pt; border-bottom: 1px solid #d0d7de; padding-bottom: .2em; margin-top: 1.4em; }
h3 { font-size: 12.5pt; }
code { background: #f0f1f2; padding: .1em .35em; border-radius: 4px; font-size: 90%; }
pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
blockquote {
  border-left: 4px solid #d0a215; background: #fff8e6; margin: 1em 0;
  padding: .6em 1em; color: #4d3b00; border-radius: 0 6px 6px 0;
}
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #d0d7de; padding: 5px 9px; }
th { background: #f0f3f6; }
tr:nth-child(even) td { background: #fafbfc; }
img { max-width: 100%; height: auto; display: block; margin: .8em auto; }
h2, h3 { page-break-after: avoid; }
table, img, pre { page-break-inside: avoid; }
a { color: #0969da; text-decoration: none; }
hr { border: none; border-top: 1px solid #d0d7de; }
"""


def find_chrome():
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(c)
        if path:
            return path
    raise SystemExit("Chrome/Chromium não encontrado.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    md_path = pathlib.Path(args.md).resolve()
    out = pathlib.Path(args.out).resolve() if args.out else md_path.with_suffix(".pdf")

    body = markdown.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = ("<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")

    # HTML na mesma pasta do .md → resolve imagens relativas (figs/...).
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    chrome = find_chrome()
    subprocess.run(
        [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
         "--allow-file-access-from-files", "--virtual-time-budget=10000",
         "--no-pdf-header-footer", f"--print-to-pdf={out}", html_path.as_uri()],
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    html_path.unlink(missing_ok=True)
    print(f"PDF gerado: {out}")


if __name__ == "__main__":
    main()
